# USV-AUV-delay

Training, evaluation and simulation code for the paper:

> **Communication-Aware Time-Scale-Separated Bi-Level Coordination for USV-AUV Collaboration in Underwater Mobile Computing**
> Jingzehua Xu†, Hongmiaoyi Zhang†, Yubo Huang, Zixi Wang, Junhao Huang, Guanwen Xie, Xiaofan Li
> *IEEE Transactions on Mobile Computing*, 2026.

---

## Demo

All animations are generated from real experiment data (50 trials × TD3/DSAC-T, acoustic delay + Rayleigh packet loss).

**Trajectory Comparison — Stackelberg vs Baseline**

*Left: Baseline. Right: Proposed (Stackelberg + Phase-Aware RL).*

**2 AUVs**
<p align="center"><img src="README.assets/trajectory_2auv.gif" width="860"/></p>

**3 AUVs**
<p align="center"><img src="README.assets/trajectory_3auv.gif" width="860"/></p>

**4 AUVs**
<p align="center"><img src="README.assets/trajectory_4auv.gif" width="860"/></p>

**Real-Time Metrics: Tracking Error / FIM / USV Motion (mean ± std, 50 episodes)**

**2 AUVs**
<p align="center"><img src="README.assets/metrics_2auv.gif" width="860"/></p>

**3 AUVs**
<p align="center"><img src="README.assets/metrics_3auv.gif" width="860"/></p>

**4 AUVs**
<p align="center"><img src="README.assets/metrics_4auv.gif" width="860"/></p>

**RL Backbone Comparison: TD3 vs DSAC-T (Stackelberg, 3 AUVs)**

<p align="center"><img src="README.assets/td3_vs_dsac_3auv.gif" width="860"/></p>

**Advantage Across Team Sizes (2 / 3 / 4 AUVs)**

<p align="center"><img src="README.assets/team_size_summary.gif" width="860"/></p>

---

## Requirements

- Python 3.8+
- See `requirements.txt`

## Installation

```bash
git clone https://github.com/Hugh41/USV-AUV-delay.git
cd USV-AUV-delay
pip install -r requirements.txt

# For DSAC-T support
cd DSAC-v2 && pip install -e . && cd ..
```

## Train

```bash
# TD3 (default)
python train_td3.py --N_AUV 3

# DSAC-T
python train_dsac.py --N_AUV 3
```

Models are saved to `models_{type}_{N_AUV}AUV_{Nu}/`.

## Run Experiments

```bash
# Single run: 3 AUVs, TD3, 50 trials
python compare_delay_stackelberg.py --N_AUV 3 --rl_type td3 --repeat_num 50

# Batch: all team sizes × both backbones
bash run_delay_packetloss_exp.sh
```

Results are saved to `delay_comparison_results/`.

## Visualise

```bash
# Animate trained policy in environment
python visualize_env.py --N_AUV 3 --load_ep 500

# Plot comparison results
python visualize_comparison_delay.py
```

---

## Citation

```bibtex
@article{xu2026communication,
  title={Communication-Aware Time-Scale-Separated Bi-Level Coordination
         for {USV-AUV} Collaboration in Underwater Mobile Computing},
  author={Xu, Jingzehua and Zhang, Hongmiaoyi and Huang, Yubo and
          Wang, Zixi and Huang, Junhao and Xie, Guanwen and Li, Xiaofan},
  journal={IEEE Transactions on Mobile Computing},
  year={2026},
  publisher={IEEE}
}
```

## Acknowledgements

- DSAC-T: [DSAC-v2](https://github.com/Jingliang-Duan/DSAC-v2) (Duan et al., TPAMI 2025)
- Baseline: *"Never Too Cocky to Cooperate"* (Xu et al., IEEE TMC 2026)
