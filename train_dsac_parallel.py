"""
Parallel DSAC-T training (Ape-X style).

Architecture:
  - 1 Learner (GPU): RL training + Stackelberg
  - N Workers (CPU): environment simulation, experience collection

Workers create local CPU policy networks, receive weight updates from the
learner via a shared queue, and push transitions for the learner's replay buffer.

Usage:
  python3 -u train_dsac_parallel.py --N_AUV 2 --usv_update_frequency 5 \
      --save_model_freq 5 --gpu 2 --n_workers 2 --load_ep 110 --start_episode 111
"""

import os
import sys
import argparse
import copy
import math
import time
import numpy as np

import torch
import torch.multiprocessing as mp

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DSAC_PATH = os.path.join(_ROOT, "DSAC-v2")
_GYM_PATH  = os.path.join(_DSAC_PATH, "env_gym")

for _p in [_DSAC_PATH, _GYM_PATH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--env_id",                   type=str,   default="gym_usv_auv_multi")
parser.add_argument("--algorithm",                type=str,   default="DSAC_V2")
parser.add_argument("--enable_cuda",              default=False)
parser.add_argument("--gpu",                      type=int,   default=-1)
parser.add_argument("--seed",                     default=None)
parser.add_argument("--n_workers",                type=int,   default=2)
# env
parser.add_argument("--reward_scale",             type=float, default=1.0)
parser.add_argument("--action_type",              type=str,   default="continu")
parser.add_argument("--R_dc",                     type=float, default=6.0)
parser.add_argument("--border_x",                 type=float, default=200.0)
parser.add_argument("--border_y",                 type=float, default=200.0)
parser.add_argument("--n_s",                      type=int,   default=30)
parser.add_argument("--N_AUV",                    type=int,   default=2)
parser.add_argument("--Q",                        type=float, default=2.0)
parser.add_argument("--alpha",                    type=float, default=0.05)
parser.add_argument("--episode_length",           type=int,   default=1000)
parser.add_argument("--usv_update_frequency",     type=int,   default=5)
# networks
parser.add_argument("--value_func_name",          type=str,   default="ActionValueDistri")
parser.add_argument("--value_func_type",          type=str,   default="MLP")
parser.add_argument("--value_hidden_sizes",       type=list,  default=[256, 256, 256])
parser.add_argument("--value_hidden_activation",  type=str,   default="gelu")
parser.add_argument("--value_output_activation",  type=str,   default="linear")
parser.add_argument("--value_min_log_std",        type=int,   default=-8)
parser.add_argument("--value_max_log_std",        type=int,   default=8)
parser.add_argument("--policy_func_name",         type=str,   default="StochaPolicy")
parser.add_argument("--policy_func_type",         type=str,   default="MLP")
parser.add_argument("--policy_act_distribution",  type=str,   default="TanhGaussDistribution")
parser.add_argument("--policy_hidden_sizes",      type=list,  default=[256, 256, 256])
parser.add_argument("--policy_hidden_activation", type=str,   default="gelu")
parser.add_argument("--policy_output_activation", type=str,   default="linear")
parser.add_argument("--policy_min_log_std",       type=int,   default=-20)
parser.add_argument("--policy_max_log_std",       type=float, default=0.5)
# RL
parser.add_argument("--value_learning_rate",      type=float, default=0.0001)
parser.add_argument("--policy_learning_rate",     type=float, default=0.0001)
parser.add_argument("--alpha_learning_rate",      type=float, default=0.0003)
parser.add_argument("--gamma",                    type=float, default=0.97)
parser.add_argument("--tau",                      type=float, default=0.005)
parser.add_argument("--auto_alpha",               type=bool,  default=True)
parser.add_argument("--entropy_alpha",            type=float, default=0.2)
parser.add_argument("--delay_update",             type=int,   default=2)
parser.add_argument("--TD_bound",                 type=float, default=1.0)
parser.add_argument("--bound",                    default=True)
# buffer
parser.add_argument("--buffer_name",              type=str,   default="replay_buffer")
parser.add_argument("--buffer_warm_size",         type=int,   default=1000)
parser.add_argument("--buffer_max_size",          type=int,   default=500000)
parser.add_argument("--replay_batch_size",        type=int,   default=256)
parser.add_argument("--sample_batch_size",        type=int,   default=20)
# training
parser.add_argument("--episode_num",              type=int,   default=600)
parser.add_argument("--save_model_freq",          type=int,   default=5)
parser.add_argument("--models_dir",               type=str,   default="models_dsac")
parser.add_argument("--log_interval",             type=int,   default=1)
parser.add_argument("--load_ep",                  type=int,   default=None)
parser.add_argument("--start_episode",            type=int,   default=0)

_cli_args = parser.parse_args()


def _build_agent_kwargs(env, cli_args):
    """Convert parsed args into the kwargs dict expected by DSACAgent."""
    from utils.common_utils import seed_everything

    kw = vars(copy.deepcopy(cli_args))
    kw["trainer"] = "off_serial_trainer"
    kw["additional_info"] = {}
    kw["obsv_dim"]          = env.observation_space.shape[0]
    kw["action_dim"]        = env.action_space.shape[0]
    kw["action_high_limit"] = env.action_space.high.astype("float32")
    kw["action_low_limit"]  = env.action_space.low.astype("float32")
    kw["batch_size_per_sampler"] = kw.get("sample_batch_size", 20)
    kw["use_gpu"]  = False
    kw["enable_cuda"] = False
    seed = kw.get("seed")
    kw["seed"] = seed_everything(seed) if seed else np.random.randint(0, 10000)
    kw["cnn_shared"] = False
    # entropy_alpha → alpha (the SAC temperature)
    kw["alpha"] = kw.pop("entropy_alpha", 0.2)
    # remove CLI-only keys
    for k in ("gpu", "n_workers", "env_id", "load_ep", "start_episode",
              "models_dir", "log_interval", "save_model_freq", "episode_num"):
        kw.pop(k, None)
    return kw


# ─────────────────────────────────────────────────────────────────────────────
# DSACAgent (same as in train_dsac.py but importable)
# ─────────────────────────────────────────────────────────────────────────────

class DSACAgent:
    def __init__(self, agent_idx, **kwargs):
        self.agent_idx = agent_idx
        self.kwargs    = kwargs
        alg_name       = kwargs["algorithm"]
        module         = __import__(alg_name.lower())
        self.ApproxContainer = getattr(module, "ApproxContainer")
        alg_cls              = getattr(module, alg_name)
        self.networks = self.ApproxContainer(**kwargs)
        self.alg = alg_cls(**kwargs)
        self.alg.networks = self.networks
        from training.replay_buffer import ReplayBuffer
        self.buffer = ReplayBuffer(index=agent_idx, **kwargs)
        self.replay_batch_size = kwargs["replay_batch_size"]
        self.use_gpu = kwargs.get("use_gpu", False)
        if self.use_gpu:
            self.networks.cuda()

    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            t = torch.from_numpy(np.expand_dims(obs, 0).astype("float32"))
            if self.use_gpu:
                t = t.cuda()
            logits = self.networks.policy(t)
            dist = self.networks.create_action_distributions(logits)
            action = dist.mode() if deterministic else dist.sample()[0]
            return action.detach().cpu().numpy()[0]

    def store_transition(self, obs, action, reward, next_obs, done, logp=0.0):
        data = [(obs.copy(), {}, action, reward, next_obs.copy(), done, logp, {})]
        self.buffer.add_batch(data)

    def train_step(self, iteration):
        if self.buffer.size < self.kwargs["buffer_warm_size"]:
            return None
        batch = self.buffer.sample_batch(self.replay_batch_size)
        if self.use_gpu:
            for k, v in batch.items():
                batch[k] = v.cuda()
        return self.alg.local_update(batch, iteration)

    def save(self, path):
        torch.save(self.networks.state_dict(), path)

    def load(self, path):
        self.networks.load_state_dict(torch.load(path, map_location="cpu"))


# ─────────────────────────────────────────────────────────────────────────────
# Worker: CPU env simulation, no Stackelberg
# ─────────────────────────────────────────────────────────────────────────────

def _worker_fn(rank, agent_kwargs_list, shared_usv_xy,
               weights_queue, transition_queue, stop_flag):
    """
    Each worker:
      - creates a CPU-only DSAC policy for each AUV
      - runs the raw env.Env (no Stackelberg)
      - syncs USV position from learner
      - polls weights_queue for policy updates
      - pushes (auv_idx, s, a, r, s', done) to transition_queue
    """
    # Hide all GPUs — workers are CPU-only; prevents PyTorch from
    # creating a CUDA context (~256 MiB) on every visible GPU.
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    np.random.seed(rank * 1000 + int(time.time()) % 10000)

    # Set up sys.path for DSAC modules inside spawned process
    for _p in [_DSAC_PATH, _GYM_PATH]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import sys as _sys
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)

    from env import Env

    class _Args:
        pass

    def _make_args(kw):
        a = _Args()
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    first_kw = agent_kwargs_list[0]
    env_args = _make_args({
        "n_s": first_kw.get("n_s", 30),
        "N_AUV": first_kw.get("N_AUV", 2),
        "border_x": first_kw.get("border_x", 200.0),
        "border_y": first_kw.get("border_y", 200.0),
        "R_dc": first_kw.get("R_dc", 6.0),
        "episode_length": first_kw.get("episode_length", 1000),
        "usv_update_frequency": first_kw.get("usv_update_frequency", 5),
        "alpha": first_kw.get("alpha", 0.05),
        "Q": first_kw.get("Q", 2.0),
    })

    env = Env(env_args)
    N = env_args.N_AUV

    # Build local CPU policy agents
    alg_name = agent_kwargs_list[0]["algorithm"]
    module = __import__(alg_name.lower())
    ApproxContainer = getattr(module, "ApproxContainer")

    # Only need policy network for action selection
    local_policies = []
    for i in range(N):
        net = ApproxContainer(**agent_kwargs_list[i])
        net.eval()
        local_policies.append(net)

    noise = 0.5

    def select_cpu(i, obs):
        with torch.no_grad():
            t = torch.from_numpy(np.expand_dims(obs, 0).astype("float32"))
            logits = local_policies[i].policy(t)
            dist = local_policies[i].create_action_distributions(logits)
            action, _ = dist.sample()
            return action.detach().cpu().numpy()[0]

    mode   = [0] * N
    ht     = [0.0] * N
    hovers = [False] * N
    Q      = env_args.Q

    while not stop_flag.value:
        # Poll for weight updates
        while not weights_queue.empty():
            try:
                idx, sd = weights_queue.get_nowait()
                local_policies[idx].load_state_dict(sd)
            except Exception:
                break

        state = env.reset()
        env.usv_xy = np.array(shared_usv_xy[:])

        Ft = 0
        while Ft <= env_args.episode_length:
            if Ft % env_args.usv_update_frequency == 0:
                env.usv_xy = np.array(shared_usv_xy[:])

            act = []
            for i in range(N):
                a = select_cpu(i, state[i])
                a = np.clip(a + noise * np.random.randn(2), -1, 1)
                act.append(a)

            env.posit_change(act, hovers)
            state_, rewards, Done, data_rate, ec, _ = env.step_move(hovers)

            for i in range(N):
                if mode[i] == 0:
                    if not stop_flag.value:
                        transition_queue.put((
                            i,
                            state[i].copy(),
                            act[i].copy(),
                            float(rewards[i]),
                            state_[i].copy(),
                            False,
                        ))
                    state[i] = copy.deepcopy(state_[i])
                    if Done[i]:
                        ht[i] = Q * env.updata[i] / max(data_rate[i], 1e-9)
                        mode[i] = math.ceil(ht[i])
                        hovers[i] = True
                else:
                    mode[i] -= 1
                    if mode[i] == 0:
                        hovers[i] = False
                        state[i] = env.CHOOSE_AIM(idx=i, lamda=env_args.alpha)

            Ft += 1
            env.Ft = Ft

        noise = max(noise * (0.99998 ** env_args.episode_length), 0.1)
        mode   = [0] * N
        ht     = [0.0] * N
        hovers = [False] * N


# ─────────────────────────────────────────────────────────────────────────────
# Learner
# ─────────────────────────────────────────────────────────────────────────────

def _learner_train(start_ep, agents, env, models_dir, shared_usv_xy,
                   weights_queue, transition_queue, stop_flag, cli):
    N_AUV          = env.n_agents
    episode_num    = cli.episode_num
    episode_length = cli.episode_length
    save_freq      = cli.save_model_freq
    Q              = cli.Q

    ep_reward_history = []
    total_steps       = 0
    noise             = 0.5

    print(f"\n{'='*100}")
    header = f"{'Episode':<10}"
    for i in range(N_AUV):
        header += f" {'AUV'+str(i):<10}"
    header += f" {'Total':<12} {'Avg(100)':<12} {'DataRate':<12} {'Noise':<8}"
    print(header)
    print(f"{'='*100}")

    for ep in range(start_ep, episode_num):
        all_states = env.reset()
        states = [all_states[i].copy().astype(np.float32) for i in range(N_AUV)]

        # Broadcast USV to workers
        usv = env.env.usv_xy if hasattr(env.env, "usv_xy") else [100.0, 100.0]
        for k in range(2):
            shared_usv_xy[k] = float(usv[k]) if k < len(usv) else 100.0

        ep_rewards  = [0.0] * N_AUV
        ep_data_rate = 0.0
        step         = 0

        while step < episode_length:
            # Drain worker transitions
            drained = 0
            while not transition_queue.empty() and drained < 512:
                try:
                    i, s, a, r, ns, d = transition_queue.get_nowait()
                    agents[i].store_transition(s, a, r, ns, d)
                    drained += 1
                except Exception:
                    break

            # Learner's own step
            actions = []
            for i in range(N_AUV):
                a = agents[i].select_action(states[i])
                a = np.clip(a + noise * np.random.randn(2), -1, 1)
                actions.append(a)

            next_states, rewards, dones, infos = env.step(actions)

            for i in range(N_AUV):
                agents[i].store_transition(
                    states[i], actions[i], rewards[i],
                    next_states[i].astype(np.float32), dones[i]
                )
                if agents[i].buffer.size >= agents[i].kwargs["buffer_warm_size"]:
                    agents[i].train_step(total_steps)
                states[i] = next_states[i].copy().astype(np.float32)
                ep_rewards[i] += rewards[i]
                ep_data_rate  += infos[i].get("data_rate", 0)

            # Update USV broadcast
            usv = env.env.usv_xy if hasattr(env.env, "usv_xy") else [100.0, 100.0]
            for k in range(2):
                shared_usv_xy[k] = float(usv[k]) if k < len(usv) else 100.0

            # Broadcast policy weights to workers every 50 steps
            if step % 50 == 0:
                for i in range(N_AUV):
                    sd = {k: v.cpu() for k, v in
                          agents[i].networks.policy.state_dict().items()}
                    weights_queue.put((i, sd))

            step        += 1
            total_steps += 1
            noise        = max(noise * 0.99998, 0.1)

            if all(dones):
                break

        total_reward = sum(ep_rewards)
        ep_reward_history.append(total_reward)
        avg_reward = np.mean(ep_reward_history[-100:])

        log_path = os.path.join(models_dir, "training_log.txt")
        with open(log_path, "a") as f:
            f.write(f"{ep},{total_reward},{avg_reward},{ep_data_rate},{noise}\n")

        log_str = f"{ep:<10}"
        for i in range(N_AUV):
            log_str += f" {ep_rewards[i]:<10.2f}"
        log_str += f" {total_reward:<12.2f} {avg_reward:<12.2f} {ep_data_rate:<12.2f} {noise:<8.4f}"
        print(log_str)

        if ep % save_freq == 0 and ep > 0:
            print(f"\n[Episode {ep}] Saving models ...")
            for i in range(N_AUV):
                path = os.path.join(models_dir, f"DSAC_{i}_{ep}.pkl")
                agents[i].save(path)
            print(f"{'='*100}")

    stop_flag.value = 1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    cli = _cli_args

    # GPU setup
    gpu_id = cli.gpu
    if gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    # models dir
    models_dir = cli.models_dir
    if models_dir == "models_dsac":
        models_dir = f"models_dsac_{cli.N_AUV}AUV_{cli.usv_update_frequency}"
    os.makedirs(models_dir, exist_ok=True)

    # Build learner environment (gym wrapper with Stackelberg)
    from gym_usv_auv_multi_data import USVAUVMultiEnv
    env = USVAUVMultiEnv(**vars(cli))
    N_AUV = env.n_agents

    agent_kwargs = _build_agent_kwargs(env, cli)

    # Enable GPU for learner agents
    if gpu_id >= 0 and torch.cuda.is_available():
        agent_kwargs["use_gpu"]     = True
        agent_kwargs["enable_cuda"] = True

    # Create per-AUV kwargs (different seed)
    base_seed = agent_kwargs.get("seed", 0)
    agent_kwargs_list = []
    for i in range(N_AUV):
        kw = copy.deepcopy(agent_kwargs)
        kw["seed"] = base_seed + i * 100
        kw["use_gpu"]     = False   # workers always CPU
        kw["enable_cuda"] = False
        agent_kwargs_list.append(kw)

    # Create learner agents (GPU)
    learner_kwargs_list = []
    for i in range(N_AUV):
        kw = copy.deepcopy(agent_kwargs)
        kw["seed"] = base_seed + i * 100
        learner_kwargs_list.append(kw)

    agents = [DSACAgent(i, **learner_kwargs_list[i]) for i in range(N_AUV)]

    # Load checkpoint
    if cli.load_ep is not None:
        print(f"Resuming from episode {cli.load_ep} ...")
        for i in range(N_AUV):
            path = os.path.join(models_dir, f"DSAC_{i}_{cli.load_ep}.pkl")
            if os.path.exists(path):
                agents[i].load(path)
                print(f"  Loaded AUV{i} from {path}")
            else:
                print(f"  Checkpoint not found: {path}")

    # Enable Stackelberg for learner
    env.env.use_stackelberg = True
    env.set_agents(agents)

    # Shared state
    shared_usv_xy    = mp.Array("f", [100.0, 100.0])
    stop_flag        = mp.Value("i", 0)
    # weights_queue: (auv_idx, policy_state_dict)  learner → workers
    weights_queue    = mp.Queue(maxsize=N_AUV * 50)
    # transition_queue: (auv_idx, s, a, r, s', done)  workers → learner
    transition_queue = mp.Queue(maxsize=4000 * N_AUV)

    # Start workers
    workers = []
    for rank in range(cli.n_workers):
        p = mp.Process(
            target=_worker_fn,
            args=(rank, agent_kwargs_list, shared_usv_xy,
                  weights_queue, transition_queue, stop_flag),
            daemon=True,
        )
        p.start()
        workers.append(p)
    print(f"Started {cli.n_workers} worker process(es)  |  GPU {gpu_id}")

    # Learner (main process)
    try:
        _learner_train(
            cli.start_episode, agents, env, models_dir,
            shared_usv_xy, weights_queue, transition_queue, stop_flag, cli,
        )
    finally:
        stop_flag.value = 1
        for p in workers:
            p.terminate()
            p.join(timeout=5)
        print("All workers stopped.")
