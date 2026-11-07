"""Deterministic CPU smoke check without training, checkpoints or datasets."""
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from env import Env
from td3 import PolicyNet
np.random.seed(7)
torch.manual_seed(7)
for count in (2, 3, 4):
    args = SimpleNamespace(n_s=10, N_AUV=count, border_x=200, border_y=200,
                           R_dc=30, episode_length=4, usv_update_frequency=5)
    env = Env(args)
    expected = 6 + 2 * (count - 1) + 1
    assert env.state_dim == expected
    state = torch.zeros((count, expected), dtype=torch.float32, device="cpu")
    model = PolicyNet(expected, 32, 2).cpu().eval()
    with torch.no_grad():
        action = model(state)
    assert action.shape == (count, 2)
    assert torch.isfinite(action).all() and (action.abs() <= 1).all()
print("OK: 2/3/4-AUV environment initialization and finite bounded CPU policy actions")
