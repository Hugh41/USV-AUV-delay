"""
GPU-accelerated Stackelberg leader optimization.

All FIM computation and follower-response stay on GPU tensors.
AUV states are precomputed once per solve() call.
Multi-start Adam replaces scipy DE.
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# GPU FIM determinant  (exact port of env.calcnegdetJ_USV, batched over B)
# ---------------------------------------------------------------------------

def _neg_det_J_batch(usv_xy: torch.Tensor,          # (B, 2)
                     auv_positions: torch.Tensor,    # (M, 2)
                     tide_h: float                   # Python float
                     ) -> torch.Tensor:              # (B,)
    B = usv_xy.shape[0]
    M = auv_positions.shape[0]
    dev = usv_xy.device
    eps = 1e-6

    # 3-D positions: USV at tide height, AUVs at z=0
    tide_col = torch.full((B, 1), tide_h, dtype=torch.float32, device=dev)
    usv_3d = torch.cat([usv_xy, tide_col], dim=1)                   # (B, 3)
    auv_3d = torch.cat([auv_positions,
                         torch.zeros(M, 1, device=dev)], dim=1)     # (M, 3)

    # S[b,i] = ||USV_b - AUV_i||  — clamp to avoid div-by-zero
    diff = usv_3d.unsqueeze(1) - auv_3d.unsqueeze(0)                # (B, M, 3)
    S = torch.norm(diff, dim=2).clamp(min=0.5)                      # (B, M)

    # p[i] = ||AUV_i||
    p = torch.norm(auv_3d, dim=1).clamp(min=eps)                    # (M,)

    A = (p**4 - 2.0 * S**2 * p**2) / (2.0 * S**6 + eps)            # (B, M)

    det_J1 = torch.sum(S**(-2), dim=1)                               # (B,)
    det_J2 = torch.sum(2.0 * A + S**(-2), dim=1)                    # (B,)

    # sin² cross-product terms  (B, M, M)
    vi = auv_positions.unsqueeze(0) - usv_xy.unsqueeze(1)           # (B, M, 2)
    vi_norm = torch.norm(vi, dim=2).clamp(min=eps)                  # (B, M)

    cross = (vi[:, :, 0].unsqueeze(2) * vi[:, :, 1].unsqueeze(1)
             - vi[:, :, 1].unsqueeze(2) * vi[:, :, 0].unsqueeze(1))  # (B,M,M)
    norm_prod = vi_norm.unsqueeze(2) * vi_norm.unsqueeze(1) + eps   # (B,M,M)
    sin2 = (cross / norm_prod)**2                                    # (B,M,M)

    mask = torch.triu(torch.ones(M, M, device=dev, dtype=torch.bool), diagonal=1)
    Aij = A.unsqueeze(2) * A.unsqueeze(1)                           # (B,M,M)
    det_J3 = torch.sum(4.0 * Aij * sin2 * mask, dim=(1, 2))         # (B,)

    return -(det_J1 * det_J2 + det_J3)                              # (B,)


# ---------------------------------------------------------------------------
# GPU Stackelberg solver
# ---------------------------------------------------------------------------

class StackelbergGPUSolver:
    """
    Multi-start Adam on GPU.
    API-compatible with StackelbergGame (expose solve_stackelberg()).
    """

    def __init__(self, env, agents,
                 lambda_J: float = 1.0,
                 lambda_u: float = 0.05,
                 n_restarts: int = 20,
                 n_steps: int = 100,
                 lr: float = 1.0):
        self.env = env
        self.agents = agents
        self.lambda_J = lambda_J
        self.lambda_u = lambda_u
        self.n_restarts = n_restarts
        self.n_steps = n_steps
        self.lr = lr
        self._prev_usv: torch.Tensor | None = None

        # Device from actor weights (TD3: agent.actor, DSAC: agent.networks)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if agents:
            a0 = agents[0]
            try:
                if hasattr(a0, 'actor'):
                    self.device = next(a0.actor.parameters()).device
                elif hasattr(a0, 'networks'):
                    self.device = next(a0.networks.parameters()).device
            except StopIteration:
                pass

    # ------------------------------------------------------------------
    # Precompute AUV states once (independent of USV candidate position)
    # ------------------------------------------------------------------

    def _precompute_states(self, auv_t: torch.Tensor) -> torch.Tensor:
        env = self.env
        M = auv_t.shape[0]
        border_norm = float(np.linalg.norm([env.X_max, env.Y_max]))
        dev = self.device
        phase = (env.Ft % env.usv_update_frequency) / env.usv_update_frequency

        rows = []
        for i in range(M):
            parts = []
            for j in range(M):
                if j == i:
                    continue
                parts.append((auv_t[j] - auv_t[i]) / border_norm)
            tgt = torch.tensor(
                env.target_Pcenter[i] if hasattr(env, 'target_Pcenter') else [0., 0.],
                dtype=torch.float32, device=dev)
            parts.append((tgt - auv_t[i]) / border_norm)
            parts.append(auv_t[i] / border_norm)
            fx = float(env.FX[i]) / env.epi_len if i < len(env.FX) else 0.
            n_do = float(env.N_DO) / env.N_POI
            parts.append(torch.tensor([fx, n_do, phase],
                                      dtype=torch.float32, device=dev))
            rows.append(torch.cat(parts))

        return torch.stack(rows)  # (M, state_dim)

    # ------------------------------------------------------------------
    # Unified actor forward (TD3 and DSAC)
    # ------------------------------------------------------------------

    def _actor_forward(self, agent, state: torch.Tensor) -> torch.Tensor:
        """Return action (2,) on GPU.  Works for TD3 and DSAC agents."""
        s = state.unsqueeze(0)
        if hasattr(agent, 'actor'):                        # TD3
            return agent.actor(s).squeeze(0)
        elif hasattr(agent, 'networks'):                   # DSAC
            logits = agent.networks.policy(s)
            action_dist = agent.networks.create_action_distributions(logits)
            action, _ = action_dist.sample()
            return action.squeeze(0).to(self.device)
        else:
            return torch.zeros(2, device=self.device)

    # ------------------------------------------------------------------
    # Batched follower best response (GPU tensors, no_grad)
    # ------------------------------------------------------------------

    def _best_response(self, states: torch.Tensor,
                        auv_t: torch.Tensor) -> torch.Tensor:
        env = self.env
        M = auv_t.shape[0]
        dev = self.device

        with torch.no_grad():
            actions = torch.stack([
                self._actor_forward(self.agents[i], states[i])
                for i in range(M)
            ])  # (M, 2)

        v = 0.5 * (actions[:, 0] + 1.0) * (env.V_max - env.V_min) + env.V_min
        dx = v * torch.cos(actions[:, 1] * np.pi)
        dy = v * torch.sin(actions[:, 1] * np.pi)
        new_pos = auv_t + torch.stack([dx, dy], dim=1)
        hi = torch.tensor([float(env.X_max), float(env.Y_max)],
                          dtype=torch.float32, device=dev)
        return torch.clamp(new_pos, torch.zeros(2, device=dev), hi).detach()

    # ------------------------------------------------------------------
    # Main solve — multi-start Adam, everything on GPU
    # ------------------------------------------------------------------

    def solve_stackelberg(self,
                          current_auv_positions: np.ndarray,
                          current_states,
                          init_guess: np.ndarray | None = None):
        env = self.env
        dev = self.device

        # Search bounds
        if env.Ft == 0 or self._prev_usv is None:
            lo = np.array([0.0, 0.0])
            hi = np.array([float(env.X_max), float(env.Y_max)])
        else:
            prev_np = self._prev_usv.cpu().numpy()
            r = float(env.USV_SHIFT_MAX)
            lo = np.clip(prev_np - r, [0., 0.], [env.X_max, env.Y_max])
            hi = np.clip(prev_np + r, [0., 0.], [env.X_max, env.Y_max])

        lo_t = torch.tensor(lo, dtype=torch.float32, device=dev)
        hi_t = torch.tensor(hi, dtype=torch.float32, device=dev)

        # Precompute constants (once per solve call)
        auv_t = torch.tensor(current_auv_positions,
                             dtype=torch.float32, device=dev)
        mid = (lo + hi) / 2.0
        tide_h = float(env.tidewave.get_tideHeight(
            mid[0] / env.X_max, mid[1] / env.Y_max, env.Ft))

        states_gpu = self._precompute_states(auv_t)
        pred_auv = self._best_response(states_gpu, auv_t)   # (M, 2), detached

        prev_gpu = self._prev_usv.clone() if self._prev_usv is not None else None

        # ---- Fully batched: all K restarts optimised in parallel ----
        K = self.n_restarts
        x0_np = lo + np.random.rand(K, 2) * (hi - lo)          # (K, 2)
        x = torch.tensor(x0_np, dtype=torch.float32,
                         device=dev, requires_grad=True)
        opt = torch.optim.Adam([x], lr=self.lr)

        for _ in range(self.n_steps):
            opt.zero_grad()
            xc = torch.clamp(x, lo_t, hi_t)                     # (K, 2)

            # One batched forward for all K candidates
            neg_det = _neg_det_J_batch(xc, pred_auv, tide_h)    # (K,)
            losses = self.lambda_J * neg_det
            if prev_gpu is not None:
                motion = self.lambda_u * torch.sum(
                    (xc - prev_gpu.unsqueeze(0))**2, dim=1)      # (K,)
                losses = losses + motion

            total = losses.sum()
            if torch.isnan(total) or torch.isinf(total):
                break
            total.backward()
            torch.nn.utils.clip_grad_norm_([x], max_norm=100.0)
            opt.step()
            with torch.no_grad():
                x.clamp_(lo_t, hi_t)

        # Pick best valid result
        with torch.no_grad():
            xf = torch.clamp(x, lo_t, hi_t)                     # (K, 2)
            nd_f = _neg_det_J_batch(xf, pred_auv, tide_h)
            fl = self.lambda_J * nd_f
            if prev_gpu is not None:
                fl = fl + self.lambda_u * torch.sum(
                    (xf - prev_gpu.unsqueeze(0))**2, dim=1)

        valid = ~(torch.isnan(fl) | torch.isinf(fl))
        if valid.any():
            best_idx = fl[valid].argmin()
            best_pos = xf[valid][best_idx].detach().clone()
        else:
            best_pos = xf[0].detach().clone()

        self._prev_usv = best_pos.clone()
        return best_pos.cpu().numpy(), pred_auv.cpu().numpy(), None
