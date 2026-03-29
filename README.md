# USV-AUV-colab: Communication-Aware Time-Scale-Separated Bi-Level Coordination

[![IEEE TMC](https://img.shields.io/badge/IEEE-TMC-blue)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7755)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official simulation toolkit for the paper:

> **Communication-Aware Time-Scale-Separated Bi-Level Coordination for USV-AUV Collaboration in Underwater Mobile Computing**
> Jingzehua Xu†, Hongmiaoyi Zhang†, Yubo Huang, Zixi Wang, Junhao Huang, Guanwen Xie, Xiaofan Li
> *IEEE Transactions on Mobile Computing*, 2026.
> *A preliminary version appeared at IROS 2025, Hangzhou, China.*

---

## Overview

This repository provides the open-source simulation toolkit for a **communication-aware time-scale-separated bi-level coordination framework** for USV–AUV collaboration in underwater mobile computing systems. The framework explicitly models realistic acoustic communication constraints—including **long propagation delay**, **Rayleigh-fading packet loss**, and **stale leader-side observations**—and integrates them directly into the coordination architecture.

<p align="center">
  <img src="README.assets/Snipaste_2024-10-15_10-26-38.png" width="600"/>
</p>

### Key Design Principles

| Layer | Role | Method |
|-------|------|--------|
| **USV (Leader)** | Slower timescale ($N_u = 5$ steps) | FIM-based geometry optimization via Differential Evolution, over *predicted* follower responses and *stale* leader-side information |
| **AUV (Followers)** | Faster timescale (every step) | TD3 / DSAC-T RL policies with **communication-phase-aware state** $\phi_t = (t \bmod N_u) / N_u$ |
| **Evaluation Layer** | Robustness test | Acoustic delay + Rayleigh-fading packet loss injected at evaluation time |

### Main Results (TD3, under acoustic delay & packet loss)

| Team Size | USV Motion Reduction | Tracking Error Reduction |
|-----------|:-------------------:|:-----------------------:|
| 2 AUVs | 3.59 → **0.91 km** (−75%) | 0.277 → **0.239 m** (−13.8%) |
| 3 AUVs | 3.59 → **0.88 km** (−75%) | 0.242 → **0.221 m** (−8.7%) |
| 4 AUVs | 3.56 → **0.86 km** (−76%) | 0.217 → **0.206 m** (−5.3%) |

---

## Repository Structure

```
USV-AUV-colab/
├── env.py                          # Simulation environment (task, sensors, dynamics)
├── td3.py                          # TD3 actor-critic implementation
├── tidewave_usbl.py                # USBL acoustic positioning model + tide model
├── stackelberg_game.py             # USV Stackelberg leader: FIM-based DE optimization
├── water_model.py                  # Acoustic delay & Rayleigh-fading packet loss model
├── colab.py                        # Collaboration utilities
│
├── train_td3.py                    # Train AUV followers with TD3
├── train_dsac.py                   # Train AUV followers with DSAC-T
├── eval_td3.py                     # Evaluate trained TD3 policies
│
├── compare_delay_stackelberg.py    # Run Table II/III experiments (with delay & packet loss)
├── compare_stackelberg.py          # Run comparison experiments (without delay)
├── run_delay_packetloss_exp.sh     # Shell script: batch experiments for all team sizes
│
├── visualize_env.py                # Animate environment with trained model
├── visualize_comparison.py         # Visualize comparison results
├── visualize_comparison_delay.py   # Visualize delay-condition comparison results
│
├── figures/                        # Paper figure reproduction scripts
│   ├── plot_episode_frontier_delay.py       # Fig. 1: Mobility–accuracy distribution
│   ├── plot_td3_auv_panels.py               # Fig. 2–4: Trajectories, FIM, tracking error
│   ├── plot_td3_usv_occupancy_heatmaps.py   # Fig. 5–6: USV occupancy heatmaps & stats
│   ├── plot_phasewise_tracking_advantage.py # Fig. 7: Phase-wise tracking reduction
│   ├── plot_delay_compensation_phase_map.py # Fig. 8–9: Lag–motion maps & phase-end error
│   └── examples/
│       ├── draw_tracking_error.py
│       └── draw_trajectory.py
│
├── DSAC-v2/                        # DSAC-T backbone (submodule, used for generalization study)
├── requirements.txt
└── README.md
```

---

## Environment

### Task Setup

- **Workspace**: 200 m × 200 m, water depth 100 m
- **Sensor nodes**: 30 SNs with Poisson data rates ∈ {3, 5, 8, 12} Mbps, 5000 Mbit buffer
- **AUVs**: 2–4 agents, speed range [1.2, 2.2] m/s, data collection radius 6 m
- **USV**: Single surface leader, updates every $N_u = 5$ steps
- **Episode**: 1000 steps; evaluation over 50 trials × 3 groups

### Acoustic Communication Model

| Parameter | Value |
|-----------|-------|
| Carrier frequency | 20 kHz |
| Packet size | 4096 bits |
| Sound speed | 1500 m/s |
| Fixed processing delay | 0.1 s |
| Sampling delay | U(0, 0.333) s |
| Source level | 135 dB |
| Noise level | 87 dB |
| Fading model | Rayleigh, σ_h = 1/√2 |
| Modulation | QPSK |

---

## Installation

```bash
git clone https://github.com/360ZMEM/USV-AUV-colab.git
cd USV-AUV-colab
pip install -r requirements.txt
```

For DSAC-T support, install the submodule:

```bash
cd DSAC-v2
pip install -e .
```

**Requirements**: Python 3.8+, PyTorch ≥ 1.12, NumPy, SciPy, Matplotlib

---

## Usage

### 1. Train AUV Policies (TD3)

```bash
# Train with 3 AUVs
python train_td3.py --N_AUV 3

# Train with 2 / 4 AUVs
python train_td3.py --N_AUV 2
python train_td3.py --N_AUV 4
```

Trained models are saved to `models_td3_{N_AUV}AUV_5/`.

### 2. Train with DSAC-T

```bash
python train_dsac.py --N_AUV 3
```

### 3. Run Comparison Experiments (Tables II & III)

```bash
# Single run with 3 AUVs, TD3 backbone
python compare_delay_stackelberg.py --N_AUV 3 --rl_type td3

# Batch run all settings (2/3/4 AUVs × TD3/DSAC)
bash run_delay_packetloss_exp.sh
```

Results are saved to `delay_comparison_results/`.

### 4. Visualize Environment

```bash
# Animate with trained TD3 model (3 AUVs, episode 500)
python visualize_env.py --N_AUV 3 --load_ep 500

# Random policy (no trained model needed)
python visualize_env.py --N_AUV 3 --random
```

### 5. Reproduce Paper Figures

All figure scripts are in `figures/`. Each script reads from `delay_comparison_results/` and outputs publication-quality PDFs.

```bash
# Fig. 1: Mobility–accuracy distribution + bar charts
python figures/plot_episode_frontier_delay.py

# Fig. 2–4: Trajectories, FIM determinant, tracking error curves
python figures/plot_td3_auv_panels.py

# Fig. 5–6: USV occupancy heatmaps + entropy/radial spread
python figures/plot_td3_usv_occupancy_heatmaps.py

# Fig. 7: Phase-wise tracking-error reduction
python figures/plot_phasewise_tracking_advantage.py

# Fig. 8–9: Delay-compensation lag–motion maps + phase-end error
python figures/plot_delay_compensation_phase_map.py
```

---

## Method Summary

### Communication-Phase-Aware Follower State

Each AUV follower appends the **communication phase** to its observation:

$$s^k_t = \left\{\Delta p^k_{j,t},\; \Delta p^k_{\text{tar},t},\; \tilde{p}_{k,t},\; b_{k,t},\; \rho_{\text{overflow},t},\; \underbrace{\phi_t = \frac{t \bmod N_u}{N_u}}_{\text{phase term}}\right\}^\top$$

This prevents **scheduler aliasing**: two steps with identical geometry but different phases would otherwise imply different next-step dynamics.

### Stackelberg Leader Optimization

At each leader update instant $t \in \{0, N_u, 2N_u, \ldots\}$, the USV solves:

$$p^*_{u,t} = \arg\max_{p_u \in \mathcal{Q}_t} \det J\!\left(p_u,\; P^{\text{br}}_{a,t+1}(p_u;\, \bar{P}_{a,t})\right)$$

where $\bar{P}_{a,t}$ is the **stale** (buffered) follower configuration and $P^{\text{br}}_{a,t+1}$ is the one-step predicted follower response. The optimization is solved by **Differential Evolution** ($P = 20$, $I = 100$) over a local bounded search region.

### Acoustic Delay & Packet Loss

```
Effective delay:  τ = τ_samp ~ U(0, 0.333s) + τ_proc (0.1s) + d/1500 s
Packet loss:      PER = 1 - (1 - BER(SNR_Rayleigh))^4096
                  SNR depends on USV-AUV distance via TL(d) = 20log10(d) + α(f)·d/1000
```

---

## Theoretical Guarantees

Six theoretical results explain the framework's properties (Section V of the paper):

| Result | Statement |
|--------|-----------|
| Lemma V.1 | Expected delay scales affinely with USV–AUV distance |
| Prop. V.2 | Buffered observation error resets on reception; packet loss is the accumulation driver |
| Prop. V.3 | Phase term is **necessary** to remove scheduler aliasing |
| Theorem V.4 | Stage-wise Stackelberg equilibrium **exists** (Weierstrass) |
| Theorem V.5 | Prediction mismatch loss is bounded by $2L_J\varepsilon_{\text{br}} + \eta_t$ |
| Prop. V.6 | Average USV motion ≤ $\Delta_u / N_u + \Delta_u / H$ (structural bound) |

---

## Citation

If you find this code useful, please cite:

```bibtex
@article{xu2026communication,
  title={Communication-Aware Time-Scale-Separated Bi-Level Coordination for {USV-AUV} Collaboration in Underwater Mobile Computing},
  author={Xu, Jingzehua and Zhang, Hongmiaoyi and Huang, Yubo and Wang, Zixi and Huang, Junhao and Xie, Guanwen and Li, Xiaofan},
  journal={IEEE Transactions on Mobile Computing},
  year={2026},
  publisher={IEEE}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The DSAC-v2 submodule is subject to its own license — please refer to `DSAC-v2/README.md`.

---

## Acknowledgements

- DSAC-T backbone: [DSAC-v2](https://github.com/Jingliang-Duan/DSAC-v2) by Duan et al.
- Baseline framework: *"Never Too Cocky to Cooperate"* (Xu et al., IEEE TMC 2026)
