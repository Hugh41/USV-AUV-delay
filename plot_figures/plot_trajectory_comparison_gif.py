"""
Create trajectory comparison GIF: Stackelberg (Proposed) vs Baseline

Loads trajectory data from delay_comparison_results/ and generates
an animated side-by-side comparison showing the USV motion compactness
and AUV tracking behavior advantages of the proposed framework.

Usage:
    python plot_figures/plot_trajectory_comparison_gif.py                        # auto-detect latest result
    python plot_figures/plot_trajectory_comparison_gif.py --result path/to.json  # specific result file
    python plot_figures/plot_trajectory_comparison_gif.py --n_auv 3              # filter by AUV count
    python plot_figures/plot_trajectory_comparison_gif.py --fps 15 --duration 8  # animation params
"""

import os
import json
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from glob import glob
from scipy.ndimage import gaussian_filter1d

# ── colour palette ──────────────────────────────────────────────────────────
C_BASELINE  = '#E74C3C'   # red  – Baseline / Traditional
C_PROPOSED  = '#2ECC71'   # green – Proposed / Stackelberg
C_USV       = '#F39C12'   # amber – USV surface vehicle
C_SN_LOW    = '#BDC3C7'   # light grey – sensor node (low data)
C_SN_HIGH   = '#8E44AD'   # purple     – sensor node (high data)
C_BG        = '#0A0F1E'   # dark navy background
C_GRID      = '#1A2340'
C_TEXT      = '#ECF0F1'


# ── helpers ─────────────────────────────────────────────────────────────────

def find_latest_result(result_dir: str, n_auv: int | None = None) -> str | None:
    """Return the path of the most-recent delay comparison JSON."""
    pattern = f"{result_dir}/**/delay_comparison_*.json"
    files = glob(pattern, recursive=True)
    if not files:
        return None
    if n_auv is not None:
        # prefer files containing f"{n_auv}AUV" in their path
        filtered = [f for f in files if f"{n_auv}AUV" in f]
        if filtered:
            files = filtered
    return max(files, key=os.path.getmtime)


def load_result(path: str) -> dict:
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_episode(result: dict, use_stackelberg: bool, delay_scenario: float = 1.0):
    """Extract one representative episode's trajectory data."""
    key = str(delay_scenario)
    if 'results' in result:
        data = result['results'].get(key, result['results'].get(list(result['results'].keys())[-1]))
    else:
        data = result
    episodes = data.get('stackelberg' if use_stackelberg else 'traditional', [])
    if not episodes:
        return None
    # pick the median-tracking-error episode
    errors = [
        np.mean([np.mean(e['tracking_error'][i]) for i in range(len(e['tracking_error']))])
        if 'tracking_error' in e else float('inf')
        for e in episodes
    ]
    idx = int(np.argsort(errors)[len(errors) // 2])
    return episodes[idx]


def smooth(arr, sigma=3):
    return gaussian_filter1d(np.array(arr, dtype=float), sigma=sigma)


# ── main animation ───────────────────────────────────────────────────────────

def build_trajectory_gif(result_path: str, output_path: str,
                          fps: int = 12, duration_s: float = 10.0,
                          trail_len: int = 80, delay_scenario: float = 1.0):
    """
    Side-by-side trajectory GIF:
        Left  : Baseline  (traditional, updates every step)
        Right : Proposed  (Stackelberg, updates every Nu=5 steps)
    """
    result = load_result(result_path)

    ep_base  = extract_episode(result, use_stackelberg=False, delay_scenario=delay_scenario)
    ep_stack = extract_episode(result, use_stackelberg=True,  delay_scenario=delay_scenario)

    if ep_base is None or ep_stack is None:
        print("⚠  Could not find both episodes in result file.")
        return False

    # ── unpack ──────────────────────────────────────────────────────────────
    def traj(ep, key):
        return np.array(ep[key]) if key in ep else None

    x_auv_b  = np.array(ep_base['x_auv'])    # (N_AUV, T)
    y_auv_b  = np.array(ep_base['y_auv'])
    x_usv_b  = np.array(ep_base['x_usv'])    # (T,)
    y_usv_b  = np.array(ep_base['y_usv'])
    te_b     = [smooth(ep_base['tracking_error'][i]) for i in range(len(ep_base['tracking_error']))]

    x_auv_s  = np.array(ep_stack['x_auv'])
    y_auv_s  = np.array(ep_stack['y_auv'])
    x_usv_s  = np.array(ep_stack['x_usv'])
    y_usv_s  = np.array(ep_stack['y_usv'])
    te_s     = [smooth(ep_stack['tracking_error'][i]) for i in range(len(ep_stack['tracking_error']))]

    sns     = np.array(ep_base.get('SoPcenter', ep_stack.get('SoPcenter', [])))
    lda_arr = ep_base.get('lda', [5] * len(sns))

    T = min(x_usv_b.shape[0], x_usv_s.shape[0])
    total_frames = int(fps * duration_s)
    step_per_frame = max(1, T // total_frames)
    frames = range(0, T, step_per_frame)

    N_AUV = x_auv_b.shape[0]
    AUV_COLORS = ['#3498DB', '#E67E22', '#1ABC9C', '#9B59B6'][:N_AUV]

    # ── figure layout ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7), facecolor=C_BG)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1, 1],
        height_ratios=[6, 1],
        hspace=0.08, wspace=0.06,
        left=0.04, right=0.96, top=0.88, bottom=0.08
    )
    ax_l = fig.add_subplot(gs[0, 0], facecolor=C_BG)   # left  – baseline
    ax_r = fig.add_subplot(gs[0, 1], facecolor=C_BG)   # right – proposed
    ax_b = fig.add_subplot(gs[1, :], facecolor=C_BG)   # bottom – progress bar

    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 200); ax.set_ylim(0, 200)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(colors=C_TEXT, labelsize=7)
        ax.set_xlabel('X (m)', color=C_TEXT, fontsize=8)
        ax.set_ylabel('Y (m)', color=C_TEXT, fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)
        ax.grid(color=C_GRID, linewidth=0.4, alpha=0.6)

    ax_b.axis('off')

    # titles
    ax_l.set_title('Baseline (Traditional FIM)', color=C_BASELINE, fontsize=11, fontweight='bold', pad=4)
    ax_r.set_title('Proposed (Stackelberg + Phase-Aware RL)', color=C_PROPOSED, fontsize=11, fontweight='bold', pad=4)

    # global title
    fig.suptitle('USV-AUV Collaboration: Stackelberg vs Baseline (Acoustic Delay + Packet Loss)',
                 color=C_TEXT, fontsize=12, fontweight='bold', y=0.96)

    # ── draw static sensor nodes ─────────────────────────────────────────────
    def draw_sns(ax):
        if len(sns) == 0:
            return
        max_lda = max(lda_arr)
        for (sx, sy), ld in zip(sns, lda_arr):
            alpha = 0.3 + 0.5 * (ld / max_lda)
            ax.scatter(sx, sy, s=18, c=[[0.6, 0.3, 0.8]], alpha=alpha, zorder=2, linewidths=0)
        # legend marker
        ax.scatter([], [], s=18, c=[[0.6, 0.3, 0.8]], alpha=0.7, label='Sensor Node', zorder=2)

    draw_sns(ax_l)
    draw_sns(ax_r)

    # ── animated artists ─────────────────────────────────────────────────────
    # Trails (line objects)
    trail_usv_b,  = ax_l.plot([], [], '-', color=C_USV, lw=1.2, alpha=0.5, zorder=3)
    trail_usv_s,  = ax_r.plot([], [], '-', color=C_USV, lw=1.2, alpha=0.5, zorder=3)

    trail_auv_b = [ax_l.plot([], [], '-', color=AUV_COLORS[i], lw=0.8, alpha=0.35, zorder=3)[0]
                   for i in range(N_AUV)]
    trail_auv_s = [ax_r.plot([], [], '-', color=AUV_COLORS[i], lw=0.8, alpha=0.35, zorder=3)[0]
                   for i in range(N_AUV)]

    # Current positions (scatter)
    dot_usv_b  = ax_l.scatter([], [], s=90,  c=[C_USV],        marker='D', zorder=6, linewidths=0.5, edgecolors='white')
    dot_usv_s  = ax_r.scatter([], [], s=90,  c=[C_USV],        marker='D', zorder=6, linewidths=0.5, edgecolors='white')
    dots_auv_b = [ax_l.scatter([], [], s=55, c=[AUV_COLORS[i]], marker='o', zorder=5, linewidths=0.3, edgecolors='white')
                  for i in range(N_AUV)]
    dots_auv_s = [ax_r.scatter([], [], s=55, c=[AUV_COLORS[i]], marker='o', zorder=5, linewidths=0.3, edgecolors='white')
                  for i in range(N_AUV)]

    # Metric text boxes
    def metric_box(ax, x, y, text, color):
        return ax.text(x, y, text, transform=ax.transAxes,
                       color=color, fontsize=7.5, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', fc='#0D1526', ec=color, alpha=0.85),
                       va='top', ha='left', zorder=10)

    txt_te_b = metric_box(ax_l, 0.02, 0.98, '', C_BASELINE)
    txt_tm_b = metric_box(ax_l, 0.02, 0.86, '', C_USV)
    txt_te_s = metric_box(ax_r, 0.02, 0.98, '', C_PROPOSED)
    txt_tm_s = metric_box(ax_r, 0.02, 0.86, '', C_USV)

    # progress bar
    prog_bg = ax_b.axhline(0.5, color=C_GRID, lw=4, transform=ax_b.transAxes, solid_capstyle='round')
    prog_line, = ax_b.plot([], [], '-', color=C_PROPOSED, lw=4, transform=ax_b.transAxes,
                           solid_capstyle='round', zorder=3)
    step_txt = ax_b.text(0.5, 0.5, '', transform=ax_b.transAxes,
                         color=C_TEXT, fontsize=8, ha='center', va='center')

    # legend
    handles = [
        mpatches.Patch(color=C_USV,  label='USV (surface leader)'),
        *[mpatches.Patch(color=AUV_COLORS[i], label=f'AUV {i+1}') for i in range(N_AUV)],
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#9B59B6',
               markersize=5, label='Sensor Node'),
    ]
    fig.legend(handles=handles, loc='lower right', ncol=len(handles),
               fontsize=7.5, framealpha=0.8, facecolor='#0D1526',
               labelcolor=C_TEXT, edgecolor=C_GRID,
               bbox_to_anchor=(0.97, 0.01))

    # ── update function ──────────────────────────────────────────────────────
    frame_list = list(frames)

    def update(fi):
        t = frame_list[fi]
        lo = max(0, t - trail_len)

        # --- baseline ---
        trail_usv_b.set_data(x_usv_b[lo:t+1], y_usv_b[lo:t+1])
        dot_usv_b.set_offsets([[x_usv_b[t], y_usv_b[t]]])
        for i in range(N_AUV):
            trail_auv_b[i].set_data(x_auv_b[i, lo:t+1], y_auv_b[i, lo:t+1])
            dots_auv_b[i].set_offsets([[x_auv_b[i, t], y_auv_b[i, t]]])
        avg_te_b = np.mean([te_b[i][t] if t < len(te_b[i]) else te_b[i][-1] for i in range(N_AUV)])
        usv_dist_b = np.sum(np.hypot(np.diff(x_usv_b[:t+1]), np.diff(y_usv_b[:t+1])))
        txt_te_b.set_text(f'Track Err: {avg_te_b:.3f} m')
        txt_tm_b.set_text(f'USV Dist:  {usv_dist_b/1000:.2f} km')

        # --- proposed ---
        trail_usv_s.set_data(x_usv_s[lo:t+1], y_usv_s[lo:t+1])
        dot_usv_s.set_offsets([[x_usv_s[t], y_usv_s[t]]])
        for i in range(N_AUV):
            trail_auv_s[i].set_data(x_auv_s[i, lo:t+1], y_auv_s[i, lo:t+1])
            dots_auv_s[i].set_offsets([[x_auv_s[i, t], y_auv_s[i, t]]])
        avg_te_s = np.mean([te_s[i][t] if t < len(te_s[i]) else te_s[i][-1] for i in range(N_AUV)])
        usv_dist_s = np.sum(np.hypot(np.diff(x_usv_s[:t+1]), np.diff(y_usv_s[:t+1])))
        txt_te_s.set_text(f'Track Err: {avg_te_s:.3f} m')
        txt_tm_s.set_text(f'USV Dist:  {usv_dist_s/1000:.2f} km')

        # progress bar
        prog_pct = t / max(T - 1, 1)
        prog_line.set_data([0.03, 0.03 + 0.94 * prog_pct], [0.5, 0.5])
        step_txt.set_text(f'Step {t}/{T}   |   Δ Track Err: {(avg_te_b - avg_te_s)*100/max(avg_te_b,1e-6):+.1f}%   |   Δ USV Dist: {(usv_dist_b - usv_dist_s)/1000:+.2f} km')

        return (trail_usv_b, trail_usv_s, dot_usv_b, dot_usv_s,
                *trail_auv_b, *trail_auv_s, *dots_auv_b, *dots_auv_s,
                txt_te_b, txt_tm_b, txt_te_s, txt_tm_s,
                prog_line, step_txt)

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_list),
        interval=1000 // fps, blit=True
    )

    writer = animation.PillowWriter(fps=fps, metadata={'loop': 0})
    ani.save(output_path, writer=writer, dpi=110)
    plt.close(fig)
    print(f'  ✓  Saved → {output_path}')
    return True


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result',    type=str,   default=None,  help='path to comparison JSON/pkl')
    ap.add_argument('--result_dir',type=str,   default='delay_comparison_results')
    ap.add_argument('--n_auv',     type=int,   default=None,  help='filter by AUV count (2/3/4)')
    ap.add_argument('--output',    type=str,   default='docs/trajectory_comparison.gif')
    ap.add_argument('--fps',       type=int,   default=12)
    ap.add_argument('--duration',  type=float, default=10.0,  help='GIF duration in seconds')
    ap.add_argument('--trail',     type=int,   default=80,    help='trail length (steps)')
    ap.add_argument('--delay_scenario', type=float, default=1.0,
                    help='which delay scenario to visualise (0=no-loss, 1=with-loss)')
    args = ap.parse_args()

    result_path = args.result
    if result_path is None:
        result_path = find_latest_result(args.result_dir, args.n_auv)
    if result_path is None:
        print('ERROR: No comparison result file found.')
        print(f'       Run compare_delay_stackelberg.py first, then re-run this script.')
        print(f'       Expected location: {args.result_dir}/')
        return

    print(f'Loading: {result_path}')
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    ok = build_trajectory_gif(
        result_path, args.output,
        fps=args.fps, duration_s=args.duration,
        trail_len=args.trail, delay_scenario=args.delay_scenario
    )
    if ok:
        size_kb = os.path.getsize(args.output) // 1024
        print(f'  GIF size: {size_kb} KB  ({args.output})')


if __name__ == '__main__':
    main()
