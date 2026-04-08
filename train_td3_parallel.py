"""
Parallel TD3 training (Ape-X style).

Architecture:
  - 1 Learner process  (GPU): RL network training + Stackelberg
  - N Worker processes (CPU): environment simulation, experience collection

Workers use the learner's shared actor weights for action selection.
USV position is updated by the learner and broadcast to workers, so workers
never run Stackelberg themselves (no GPU contention).

Usage:
  python3 -u train_td3_parallel.py --N_AUV 2 --usv_update_frequency 5 \
      --save_model_freq 5 --gpu 2 --n_workers 3 --load_ep 395 --start_episode 396
"""

import math
import os
import copy
import argparse
import numpy as np
import time

import torch
import torch.multiprocessing as mp

from env import Env
from td3 import TD3, set_device, ReplayBuffer

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--lr",              type=float, default=0.001)
parser.add_argument("--gamma",           type=float, default=0.97)
parser.add_argument("--tau",             type=float, default=0.001)
parser.add_argument("--hidden_size",     type=int,   default=128)
parser.add_argument("--replay_capa",     type=int,   default=20000)
parser.add_argument("--batch_size",      type=int,   default=64)
parser.add_argument("--policy_freq",     type=int,   default=2)
parser.add_argument("--episode_num",     type=int,   default=602)
parser.add_argument("--episode_length",  type=int,   default=1000)
parser.add_argument("--save_model_freq", type=int,   default=5)
parser.add_argument("--gpu",             type=int,   default=-1)
parser.add_argument("--n_workers",       type=int,   default=2,
                    help="number of parallel environment workers")
parser.add_argument("--load_ep",         type=int,   default=None)
parser.add_argument("--start_episode",   type=int,   default=0)
parser.add_argument("--R_dc",            type=float, default=6.0)
parser.add_argument("--border_x",        type=float, default=200.0)
parser.add_argument("--border_y",        type=float, default=200.0)
parser.add_argument("--n_s",             type=int,   default=30)
parser.add_argument("--N_AUV",           type=int,   default=2)
parser.add_argument("--Q",               type=float, default=2.0)
parser.add_argument("--alpha",           type=float, default=0.05)
parser.add_argument("--usv_update_frequency", type=int, default=5)
args = parser.parse_args()

SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"models_td3_{args.N_AUV}AUV_{args.usv_update_frequency}/")
os.makedirs(SAVE_PATH, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Worker process: collects experiences, no Stackelberg, no GPU needed
# ─────────────────────────────────────────────────────────────────────────────

def _worker_fn(rank, args_ns, shared_actors, shared_usv_xy,
               transition_queue, stop_flag):
    """
    Worker runs the environment on CPU.
    - Uses shared_actors for action selection (CPU inference, weights shared
      in memory with learner — automatically sees latest policy).
    - Uses shared_usv_xy as the current USV position (learner keeps it updated).
    - Sends (state, action, reward, next_state) tuples via transition_queue.
    """
    # Hide all GPUs — workers are CPU-only; this prevents PyTorch from
    # creating a CUDA context (~256 MiB) on every visible GPU.
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    np.random.seed(rank * 1000 + int(time.time()) % 10000)

    env = Env(args_ns)
    # Workers do NOT call env.set_agents — USV is fixed at shared position,
    # so no Stackelberg solver needed here.

    N = args_ns.N_AUV
    noise = 0.8

    def select_action_cpu(actor, state):
        with torch.no_grad():
            s = torch.FloatTensor(state.reshape(1, -1))   # CPU
            return actor._orig_mod(s).numpy().flatten() if hasattr(actor, '_orig_mod') \
                else actor(s).numpy().flatten()

    mode   = [0] * N
    ht     = [0.0] * N
    hovers = [False] * N

    while not stop_flag.value:
        state = env.reset()

        # Sync USV position from learner
        env.usv_xy = np.array(shared_usv_xy[:])

        Ft = 0
        while Ft <= args_ns.episode_length:
            # Sync USV every usv_update_frequency steps
            if Ft % args_ns.usv_update_frequency == 0:
                env.usv_xy = np.array(shared_usv_xy[:])

            act = []
            for i in range(N):
                a = select_action_cpu(shared_actors[i], state[i])
                a = np.clip(a + noise * np.random.randn(2), -1, 1)
                act.append(a)

            env.posit_change(act, hovers)
            state_, rewards, Done, data_rate, ec, _ = env.step_move(hovers)

            for i in range(N):
                if mode[i] == 0:
                    if not stop_flag.value:
                        transition_queue.put((i, state[i].copy(),
                                              act[i].copy(),
                                              float(rewards[i]),
                                              state_[i].copy(),
                                              False))
                    state[i] = copy.deepcopy(state_[i])
                    if Done[i]:
                        ht[i] = args_ns.Q * env.updata[i] / max(data_rate[i], 1e-9)
                        mode[i] = math.ceil(ht[i])
                        hovers[i] = True
                else:
                    mode[i] -= 1
                    if mode[i] == 0:
                        hovers[i] = False
                        state[i] = env.CHOOSE_AIM(idx=i, lamda=args_ns.alpha)

            Ft += 1
            env.Ft = Ft

        noise = max(noise * (0.99998 ** args_ns.episode_length), 0.1)
        mode   = [0] * N
        ht     = [0.0] * N
        hovers = [False] * N


# ─────────────────────────────────────────────────────────────────────────────
# Learner: GPU training + Stackelberg
# ─────────────────────────────────────────────────────────────────────────────

def learner_train(start_episode: int, shared_actors, shared_usv_xy,
                  transition_queue, stop_flag, agents, env, N_AUV):
    noise = 0.8
    ep_reward_history = []

    print(f"\n{'='*100}")
    print(f"{'Episode':<10} {'Total Reward':<15} {'Avg Reward (100)':<20} "
          f"{'Steps':<10} {'Data Rate':<15} {'TD Error':<15}")
    print(f"{'='*100}")

    for ep in range(start_episode, args.episode_num):
        # ── learner's own episode (with full Stackelberg, GPU) ──────────────
        state_c = env.reset()
        state   = copy.deepcopy(state_c)

        # Broadcast initial USV position to workers
        usv = env.usv_xy if hasattr(env, 'usv_xy') else [100.0, 100.0]
        for k in range(len(shared_usv_xy)):
            shared_usv_xy[k] = float(usv[k]) if k < len(usv) else 100.0

        ep_r = ep_reward = 0
        idu = N_DO = DQ = 0
        FX = [0] * N_AUV
        sum_rate = 0
        Ec = TD_error = A_Loss = [0] * N_AUV
        Ht = [0] * N_AUV
        Ft = 0
        update_network = [0] * N_AUV
        crash = 0
        mode   = [0] * N_AUV
        ht     = [0.0] * N_AUV
        hovers = [False] * N_AUV

        while True:
            # ── drain worker transitions ────────────────────────────────────
            drained = 0
            while not transition_queue.empty() and drained < 512:
                try:
                    i, s, a, r, ns, d = transition_queue.get_nowait()
                    agents[i].store_transition(s, a, r, ns, d)
                    drained += 1
                except Exception:
                    break

            # ── learner's own step ──────────────────────────────────────────
            act = []
            for i in range(N_AUV):
                ia = agents[i].select_action(state[i])
                ia = np.clip(ia + noise * np.random.randn(2), -1, 1)
                act.append(ia)

            env.posit_change(act, hovers)
            state_, rewards, Done, data_rate, ec, cs = env.step_move(hovers)

            crash    += cs
            ep_reward += np.sum(rewards) / 1000

            for i in range(N_AUV):
                if mode[i] == 0:
                    agents[i].store_transition(
                        state[i], act[i], rewards[i], state_[i], False)
                    state[i] = copy.deepcopy(state_[i])
                    if Done[i]:
                        idu += 1
                        ht[i] = args.Q * env.updata[i] / max(data_rate[i], 1e-9)
                        mode[i] = math.ceil(ht[i])
                        hovers[i] = True
                        sum_rate += data_rate[i]
                else:
                    mode[i] -= 1
                    Ht[i] += 1
                    if mode[i] == 0:
                        hovers[i] = False
                        Ht[i] -= math.ceil(ht[i]) - ht[i]
                        state[i] = env.CHOOSE_AIM(idx=i, lamda=args.alpha)

                if len(agents[i].replay_buffer) > 20 * args.batch_size:
                    a_loss, td_err = agents[i].train()
                    noise = max(noise * 0.99998, 0.1)
                    update_network[i] += 1
                    TD_error[i] += td_err
                    A_Loss[i]   += a_loss

            # Broadcast updated USV position to workers
            usv = env.usv_xy if hasattr(env, 'usv_xy') else [100.0, 100.0]
            for k in range(len(shared_usv_xy)):
                shared_usv_xy[k] = float(usv[k]) if k < len(usv) else 100.0

            # Share updated actor weights with workers
            if Ft % 50 == 0:
                for i in range(N_AUV):
                    sd = agents[i].actor.state_dict()
                    shared_actors[i].load_state_dict(sd)

            Ft += 1
            env.Ft = Ft
            N_DO += env.N_DO
            FX    = np.array(FX) + np.array(env.FX)
            DQ   += sum(env.b_S / env.Fully_buffer)
            Ec    = np.array(Ec) + np.array(ec)

            if Ft > args.episode_length:
                for i in range(N_AUV):
                    if update_network[i]:
                        TD_error[i] /= update_network[i]
                        A_Loss[i]   /= update_network[i]
                N_DO /= Ft
                DQ   /= Ft * env.N_POI
                Ec    = np.sum(np.array(Ec) / (Ft - np.array(Ht))) / N_AUV

                ep_reward_history.append(ep_reward)
                avg_reward = np.mean(ep_reward_history[-100:])
                avg_td     = np.mean(TD_error)

                print(f"{ep:<10} {ep_reward:<15.2f} {avg_reward:<20.2f} "
                      f"{Ft:<10} {sum_rate:<15.2f} {avg_td:<15.4f}")

                log_path = os.path.join(SAVE_PATH, "training_log.txt")
                write_header = not os.path.exists(log_path)
                with open(log_path, "a") as f:
                    if write_header:
                        f.write("Episode,Total Reward,Avg Reward (100),"
                                "Steps,Data Rate,TD Error\n")
                    f.write(f"{ep},{ep_reward},{avg_reward},"
                            f"{Ft},{sum_rate},{avg_td}\n")
                break

        if ep % args.save_model_freq == 0 and ep != 0:
            for i in range(N_AUV):
                agents[i].save(SAVE_PATH, ep, idx=i)

    stop_flag.value = 1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    set_device(args.gpu)

    env = Env(args)
    N_AUV     = args.N_AUV
    state_dim = env.state_dim
    action_dim = 2

    agents = [TD3(state_dim, action_dim) for _ in range(N_AUV)]

    if args.load_ep is not None:
        print(f"Resuming from episode {args.load_ep} ...")
        for i in range(N_AUV):
            agents[i].load(SAVE_PATH, args.load_ep, idx=i)
            print(f"  Loaded AUV{i} weights from episode {args.load_ep}")

    env.use_stackelberg = True
    env.set_agents(agents)

    # ── Shared actor weights (CPU, read by workers) ─────────────────────────
    # Workers read these tensors directly — no serialisation overhead
    shared_actors = []
    for ag in agents:
        cpu_actor = copy.deepcopy(ag.actor).cpu()
        cpu_actor.share_memory()
        shared_actors.append(cpu_actor)

    # ── Shared USV position (2 floats) ──────────────────────────────────────
    shared_usv_xy = mp.Array("f", [100.0, 100.0])

    # ── Shared stop flag ────────────────────────────────────────────────────
    stop_flag = mp.Value("i", 0)

    # ── Transition queue (workers → learner) ─────────────────────────────────
    # maxsize limits memory; 4000 ≈ 4 episodes × 1000 steps
    transition_queue = mp.Queue(maxsize=4000 * N_AUV)

    # ── Launch workers ───────────────────────────────────────────────────────
    workers = []
    for rank in range(args.n_workers):
        p = mp.Process(
            target=_worker_fn,
            args=(rank, args, shared_actors, shared_usv_xy,
                  transition_queue, stop_flag),
            daemon=True,
        )
        p.start()
        workers.append(p)
    print(f"Started {args.n_workers} worker process(es)")

    # ── Learner (runs in main process) ───────────────────────────────────────
    try:
        learner_train(args.start_episode, shared_actors, shared_usv_xy,
                      transition_queue, stop_flag, agents, env, N_AUV)
    finally:
        stop_flag.value = 1
        for p in workers:
            p.terminate()
            p.join(timeout=5)
        print("All workers stopped.")
