"""
Multi-agent gym wrapper for the USV-AUV environment.
Adapts env.Env to the interface expected by train_dsac.py.
"""
import sys
import os
import copy
import math
import numpy as np
import gym
from gym import spaces

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from env import Env


class _EnvArgs:
    """Minimal args object built from a kwargs dict."""
    def __init__(self, **kwargs):
        self.n_s = kwargs.get("n_s", 30)
        self.N_AUV = kwargs.get("N_AUV", 2)
        self.border_x = kwargs.get("border_x", 200.0)
        self.border_y = kwargs.get("border_y", 200.0)
        self.R_dc = kwargs.get("R_dc", 6.0)
        self.episode_length = kwargs.get("episode_length", 1000)
        self.usv_update_frequency = kwargs.get("usv_update_frequency", 5)
        self.alpha = kwargs.get("alpha", 0.05)
        self.Q = kwargs.get("Q", 2.0)


class USVAUVMultiEnv:
    """
    Multi-agent wrapper around Env that provides:
      - observation_space / action_space  (gym.Space)
      - reset() -> list[np.ndarray]
      - step(actions) -> (list[obs], list[reward], list[done], list[info])
      - n_agents attribute
      - set_agents(agents) for Stackelberg follower prediction
    """

    def __init__(self, **kwargs):
        env_args = _EnvArgs(**kwargs)
        self.env = Env(env_args)
        self.n_agents = env_args.N_AUV
        self.alpha = env_args.alpha
        self.episode_length = env_args.episode_length
        self._step = 0

        state_dim = self.env.state_dim
        action_dim = 2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        # Per-AUV hover bookkeeping (mirrors train_td3 logic)
        self._hovers = [False] * self.n_agents
        self._modes = [0] * self.n_agents
        self._ht = [0.0] * self.n_agents

    def set_agents(self, agents):
        self.env.set_agents(agents)

    def reset(self):
        states = self.env.reset()
        self._step = 0
        self._hovers = [False] * self.n_agents
        self._modes = [0] * self.n_agents
        self._ht = [0.0] * self.n_agents
        return [s.astype(np.float32) for s in states]

    def step(self, actions):
        actions = [np.array(a, dtype=np.float64) for a in actions]

        self.env.posit_change(copy.deepcopy(actions), self._hovers)
        next_states, rewards, done_flags, data_rates, ec, _ = self.env.step_move(
            self._hovers
        )

        self._step += 1
        self.env.Ft = self._step

        infos = []
        for i in range(self.n_agents):
            info = {"data_rate": float(data_rates[i])}
            if self._modes[i] == 0:
                if done_flags[i]:
                    self._ht[i] = 2.0 * self.env.updata[i] / max(data_rates[i], 1e-9)
                    self._modes[i] = math.ceil(self._ht[i])
                    self._hovers[i] = True
            else:
                self._modes[i] -= 1
                if self._modes[i] == 0:
                    self._hovers[i] = False
                    next_states[i] = self.env.CHOOSE_AIM(idx=i, lamda=self.alpha)
            infos.append(info)

        episode_done = self._step >= self.episode_length
        dones = [bool(episode_done)] * self.n_agents

        return (
            [s.astype(np.float32) for s in next_states],
            [float(r) for r in rewards],
            dones,
            infos,
        )
