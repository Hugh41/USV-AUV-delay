import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def set_device(gpu_id: int = -1):
    """Set global device. gpu_id=-1 means auto (cuda if available, else cpu)."""
    global device
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DDPG
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        # ---- Q1 ----
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)
        # ---- Q2 ----
        self.fc3 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        Q1 = self.fc_out(x)
        # Q2
        x = F.relu(self.fc3(cat))
        x = F.relu(self.fc4(x))
        Q2 = self.fc_out2(x)
        return Q1, Q2

    # only return Q1 value(https://github.com/sfujim/TD3)
    def Q1(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        Q1 = self.fc_out(x)
        return Q1


class ReplayBuffer:
    """GPU-resident replay buffer. All data lives on device; sampling is pure GPU."""

    def __init__(self, capacity, state_dim=None, action_dim=None):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        # Dimensions resolved lazily on first push when not provided
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._initialised = False

    def _init_buffers(self, state_dim, action_dim):
        cap = self.capacity
        dev = device
        self._states      = torch.zeros(cap, state_dim,  dtype=torch.float32, device=dev)
        self._actions     = torch.zeros(cap, action_dim, dtype=torch.float32, device=dev)
        self._rewards     = torch.zeros(cap, 1,          dtype=torch.float32, device=dev)
        self._next_states = torch.zeros(cap, state_dim,  dtype=torch.float32, device=dev)
        self._dones       = torch.zeros(cap, 1,          dtype=torch.float32, device=dev)
        self._initialised = True

    def push(self, state, action, reward, next_state, done):
        state      = np.asarray(state,      dtype=np.float32)
        action     = np.asarray(action,     dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if not self._initialised:
            self._init_buffers(state.shape[-1], action.shape[-1])
        i = self.ptr
        self._states[i]      = torch.from_numpy(state)
        self._actions[i]     = torch.from_numpy(action)
        self._rewards[i, 0]  = float(reward)
        self._next_states[i] = torch.from_numpy(next_state)
        self._dones[i, 0]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=device)
        return (
            self._states[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_states[idx],
            self._dones[idx],
        )

    def __len__(self):
        return self.size


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        discount=0.97,
        tau=0.001,
        lr=1e-3,
        batch_size=64,
    ):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.discount = discount
        self.tau = tau
        self.replay_buffer = ReplayBuffer(20000)
        self.policy_freq = 2
        self.total_it = 0
        self.batch_size = batch_size
        # AMP: enabled only on CUDA; GradScaler is a no-op on CPU
        self._use_amp = (device.type == "cuda")
        self._scaler  = torch.amp.GradScaler("cuda", enabled=self._use_amp)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).detach().cpu().numpy().flatten()  # 1-dim

    def store_transition(self, state, action, reward, next_state, done=False):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        # sample() returns GPU tensors directly — no conversion needed
        state, action, reward, next_state, done = self.replay_buffer.sample(
            batch_size=self.batch_size
        )
        amp_ctx = torch.amp.autocast("cuda", enabled=self._use_amp)

        # Critic update
        with amp_ctx:
            with torch.no_grad():
                noise = (torch.randn_like(action) * 0.1).clamp(-1.0, 1.0)
                next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (1 - done) * self.discount * target_Q
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )
        self.critic_optimizer.zero_grad()
        self._scaler.scale(critic_loss).backward()
        self._scaler.step(self.critic_optimizer)
        self._scaler.update()

        # Delayed policy updates
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            with amp_ctx:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            self._scaler.scale(actor_loss).backward()
            self._scaler.step(self.actor_optimizer)
            self._scaler.update()
            with torch.no_grad():
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

        self.total_it += 1
        a_val = actor_loss.detach().cpu().float().numpy() if actor_loss is not None else 0
        return a_val, critic_loss.detach().cpu().float().numpy()

    # save model
    def save(self, filename, ep, idx):
        ep = "_" + str(ep)
        idx = "_" + str(idx)
        torch.save(
            self.critic.state_dict(), filename + "TD3" + idx + ep + "_critic.pth"
        )
        torch.save(
            self.critic_optimizer.state_dict(),
            filename + "TD3" + idx + ep + "_critic_optimizer.pth",
        )
        torch.save(self.actor.state_dict(), filename + "TD3" + idx + ep + "_actor.pth")
        torch.save(
            self.actor_optimizer.state_dict(),
            filename + "TD3" + idx + ep + "_actor_optimizer.pth",
        )

    # load model
    def load(self, filename, ep, idx):
        ep = "_" + str(ep)
        idx = "_" + str(idx)
        self.critic.load_state_dict(
            torch.load(filename + "TD3" + idx + ep + "_critic.pth")
        )
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "TD3" + idx + ep + "_critic_optimizer.pth")
        )
        self.actor.load_state_dict(
            torch.load(filename + "TD3" + idx + ep + "_actor.pth")
        )
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "TD3" + idx + ep + "_actor_optimizer.pth")
        )
        # target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
