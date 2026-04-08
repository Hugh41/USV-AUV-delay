# plot_figures

Plotting scripts for generating all figures and animations.  
All scripts read experiment data from `delay_comparison_results/` produced by `compare_delay_stackelberg.py`.  
Run all commands from the **repository root**.

---

## Animated GIFs

### Generate all GIFs for a given AUV count in one command

```bash
python plot_figures/plot_all_gifs.py \
    --data_dir delay_comparison_results \
    --out_dir  output_dir \
    --n_auv    3            # 2 / 3 / 4
```

This produces four GIFs per call:

| Output file | Content |
|---|---|
| `trajectory_<n>auv.gif` | Side-by-side trajectory: Baseline vs Proposed |
| `metrics_<n>auv.gif` | Tracking error / FIM det(J) / cumulative USV motion |
| `td3_vs_dsac_<n>auv.gif` | TD3 vs DSAC-T backbone comparison |
| `team_size_summary.gif` | Aggregate performance across 2 / 3 / 4 AUVs |

### Regenerate the full GIF set (2 / 3 / 4 AUVs)

```bash
DATA=delay_comparison_results
OUT=README.assets

for N in 2 3 4; do
    python plot_figures/plot_all_gifs.py --data_dir $DATA --out_dir $OUT --n_auv $N --skip 3
done
# team_size_summary is included in the n_auv=2 run above
```

### Run each script individually

```bash
# Trajectory comparison GIF
python plot_figures/plot_trajectory_comparison_gif.py \
    --result_dir delay_comparison_results --n_auv 3 \
    --output output_dir/trajectory_3auv.gif

# Metrics evolution GIF
python plot_figures/plot_metrics_gif.py \
    --result_dir delay_comparison_results --n_auv 3 \
    --output output_dir/metrics_3auv.gif
```

---

## Static Figures (PNG / PDF)

These scripts use fixed experiment directories hardcoded in `RESULT_DIR` at the top of each file. Run from the repository root.

### USV Occupancy Heatmaps

```bash
python plot_figures/plot_td3_usv_occupancy_heatmaps.py --output_dir output_dir
```

Outputs: `td3_usv_occupancy_heatmap_2_3_4auv.png`, `td3_usv_occupancy_hr_summary_2_3_4auv.png`

### Episode Frontier (Mobility–Accuracy Pareto)

```bash
python plot_figures/plot_episode_frontier_delay.py --output_dir output_dir
```

Output: `episode_frontier_with_bars.png`

### Phase-Wise Tracking Improvement

```bash
python plot_figures/plot_phasewise_tracking_advantage.py --output_dir output_dir
```

Output: `phasewise_tracking_reduction_reviewer.pdf`

### Delay-Compensation Phase Map

```bash
python plot_figures/plot_delay_compensation_phase_map.py --output_dir output_dir
```

Outputs: `delay_compensation_phase_map_heatmap_*.pdf`, `delay_compensation_phase_map_curve_only.pdf`

### TD3 AUV Trajectory / Tracking Error / FIM Panels

```bash
pip install ijson   # required dependency
python plot_figures/plot_td3_auv_panels.py --output_dir output_dir
```

Outputs: `td3_trajectory_2_3_4auv.pdf`, `td3_tracking_error_2_3_4auv.pdf`, `td3_fim_detj_2_3_4auv.pdf`
