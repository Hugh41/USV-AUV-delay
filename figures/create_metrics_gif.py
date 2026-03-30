"""
Create real-time metrics comparison GIF: Stackelberg vs Baseline

Generates an animated multi-panel figure showing how three key metrics
evolve over the episode:
  Panel 1 – Average tracking error  (lower is better  → Proposed wins)
  Panel 2 – FIM det(J)              (more stable → Proposed wins)
  Panel 3 – Cumulative USV motion   (lower is better  → Proposed wins)
  Panel 4 – Summary bar chart       (final episode statistics)

Usage:
    python create_metrics_gif.py
    python create_metrics_gif.py --result path/to.json --n_auv 3
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
from glob import glob
from scipy.ndimage import gaussian_filter1d

C_BASELINE = '#E74C3C'
C_PROPOSED = '#2ECC71'
C_FIM      = '#F39C12'
C_USV      = '#3498DB'
C_BG       = '#0A0F1E'
C_GRID     = '#1A2340'
C_TEXT     = '#ECF0F1'
C_AXES     = '#2C3E50'


def find_latest_result(result_dir, n_auv=None):
    files = glob(f'{result_dir}/**/delay_comparison_*.json', recursive=True)
    if not files:
        return None
    if n_auv:
        f2 = [f for f in files if f'{n_auv}AUV' in f]
        if f2:
            files = f2
    return max(files, key=os.path.getmtime)


def load_result(path):
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_all_episodes(result, use_stackelberg, delay_scenario=1.0):
    key = str(delay_scenario)
    if 'results' in result:
        d = result['results'].get(key, result['results'].get(list(result['results'].keys())[-1]))
    else:
        d = result
    tag = 'stackelberg' if use_stackelberg else 'traditional'
    return d.get(tag, [])


def smooth(arr, sigma=4):
    return gaussian_filter1d(np.array(arr, dtype=float), sigma=sigma)


def aggregate_metric(episodes, key, reduce='mean'):
    """Aggregate a per-step metric across episodes → (mean, std) arrays."""
    seqs = []
    for ep in episodes:
        if key not in ep:
            continue
        val = ep[key]
        # tracking_error: list-of-lists (N_AUV, T) → average over AUVs
        if isinstance(val[0], list):
            arr = np.mean([np.array(v) for v in val], axis=0)
        else:
            arr = np.array(val)
        seqs.append(arr)
    if not seqs:
        return None, None
    T = min(len(s) for s in seqs)
    mat = np.stack([s[:T] for s in seqs], axis=0)
    return mat.mean(axis=0), mat.std(axis=0)


def cumulative_usv_motion(episode):
    xs = np.array(episode.get('x_usv', []))
    ys = np.array(episode.get('y_usv', []))
    if len(xs) < 2:
        return np.zeros(max(len(xs), 1))
    deltas = np.hypot(np.diff(xs), np.diff(ys))
    return np.concatenate([[0.0], np.cumsum(deltas)])


def build_metrics_gif(result_path, output_path, fps=10, duration_s=12.0, delay_scenario=1.0):
    result = load_result(result_path)
    eps_b = extract_all_episodes(result, False, delay_scenario)
    eps_s = extract_all_episodes(result, True,  delay_scenario)
    if not eps_b or not eps_s:
        print('⚠  Cannot find both method episodes.')
        return False

    # ── tracking error ───────────────────────────────────────────────────────
    te_b_mean, te_b_std = aggregate_metric(eps_b, 'tracking_error')
    te_s_mean, te_s_std = aggregate_metric(eps_s, 'tracking_error')

    # ── FIM det(J) ───────────────────────────────────────────────────────────
    dj_b_mean, dj_b_std = aggregate_metric(eps_b, 'detJ_values')
    dj_s_mean, dj_s_std = aggregate_metric(eps_s, 'detJ_values')

    # ── USV motion (single representative episode – median) ──────────────────
    def median_episode(eps):
        errs = [np.mean(ep['tracking_error']) if isinstance(ep.get('tracking_error', [[]])[0], float)
                else np.mean([np.mean(v) for v in ep['tracking_error']])
                for ep in eps if 'tracking_error' in ep]
        idx = int(np.argsort(errs)[len(errs) // 2]) if errs else 0
        return eps[idx]

    ep_b = median_episode(eps_b)
    ep_s = median_episode(eps_s)
    cum_b = cumulative_usv_motion(ep_b)
    cum_s = cumulative_usv_motion(ep_s)

    T = min(
        len(te_b_mean) if te_b_mean is not None else 9999,
        len(te_s_mean) if te_s_mean is not None else 9999,
        len(cum_b), len(cum_s)
    )
    if dj_b_mean is not None:
        T = min(T, len(dj_b_mean), len(dj_s_mean))

    # smooth
    if te_b_mean is not None:
        te_b_mean = smooth(te_b_mean[:T])
        te_s_mean = smooth(te_s_mean[:T])
    if dj_b_mean is not None:
        dj_b_mean = smooth(dj_b_mean[:T], sigma=6)
        dj_s_mean = smooth(dj_s_mean[:T], sigma=6)
    cum_b = cum_b[:T]
    cum_s = cum_s[:T]

    total_frames = int(fps * duration_s)
    step_per_frame = max(1, T // total_frames)
    frame_list = list(range(0, T, step_per_frame))

    # ── figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor=C_BG)
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32,
                          left=0.07, right=0.97, top=0.88, bottom=0.10)

    ax_te  = fig.add_subplot(gs[0, 0], facecolor=C_BG)  # tracking error
    ax_fim = fig.add_subplot(gs[0, 1], facecolor=C_BG)  # FIM det
    ax_usv = fig.add_subplot(gs[0, 2], facecolor=C_BG)  # USV motion
    ax_bar = fig.add_subplot(gs[1, :], facecolor=C_BG)  # summary bars

    def style_ax(ax, title, ylabel, xlabel='Time Step'):
        ax.set_title(title, color=C_TEXT, fontsize=9, fontweight='bold', pad=4)
        ax.set_ylabel(ylabel, color=C_TEXT, fontsize=8)
        ax.set_xlabel(xlabel, color=C_TEXT, fontsize=8)
        ax.tick_params(colors=C_TEXT, labelsize=7)
        ax.set_facecolor(C_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_GRID)
        ax.grid(color=C_GRID, linewidth=0.4, alpha=0.5)
        ax.set_xlim(0, T)

    style_ax(ax_te,  'Avg. Tracking Error',        'Error (m)')
    style_ax(ax_fim, 'FIM Determinant  det(J)',    'det(J)')
    style_ax(ax_usv, 'Cumulative USV Motion',       'Distance (km)')

    fig.suptitle('Stackelberg Framework vs Baseline — Real-Time Metrics\n'
                 '(under acoustic delay + Rayleigh-fading packet loss)',
                 color=C_TEXT, fontsize=11, fontweight='bold', y=0.96)

    # ── static shaded regions (full episode) ────────────────────────────────
    t_full = np.arange(T)

    if te_b_mean is not None:
        ax_te.fill_between(t_full, te_b_mean - (te_b_std[:T] if te_b_std is not None else 0),
                            te_b_mean + (te_b_std[:T] if te_b_std is not None else 0),
                            alpha=0.12, color=C_BASELINE)
        ax_te.fill_between(t_full, te_s_mean - (te_s_std[:T] if te_s_std is not None else 0),
                            te_s_mean + (te_s_std[:T] if te_s_std is not None else 0),
                            alpha=0.12, color=C_PROPOSED)
        ax_te.set_ylim(0, max(te_b_mean.max(), te_s_mean.max()) * 1.25)

    if dj_b_mean is not None:
        ax_fim.set_ylim(0, max(dj_b_mean.max(), dj_s_mean.max()) * 1.25)

    ax_usv.set_ylim(0, max(cum_b[-1], cum_s[-1]) / 1000 * 1.15)

    # ── animated lines ───────────────────────────────────────────────────────
    line_te_b,  = ax_te.plot([], [], color=C_BASELINE, lw=1.8, label='Baseline')
    line_te_s,  = ax_te.plot([], [], color=C_PROPOSED, lw=1.8, label='Proposed')
    vl_te_b     = ax_te.axvline(0, color=C_BASELINE, lw=0.6, alpha=0.5, ls='--')
    vl_te_s     = ax_te.axvline(0, color=C_PROPOSED, lw=0.6, alpha=0.5, ls='--')

    line_fim_b, = ax_fim.plot([], [], color=C_BASELINE, lw=1.8, label='Baseline')
    line_fim_s, = ax_fim.plot([], [], color=C_PROPOSED, lw=1.8, label='Proposed')

    line_usv_b, = ax_usv.plot([], [], color=C_BASELINE, lw=1.8, label='Baseline')
    line_usv_s, = ax_usv.plot([], [], color=C_PROPOSED, lw=1.8, label='Proposed')

    for ax in (ax_te, ax_fim, ax_usv):
        ax.legend(fontsize=7.5, facecolor='#0D1526', labelcolor=C_TEXT,
                  edgecolor=C_GRID, loc='upper right')

    # ── animated bar chart ───────────────────────────────────────────────────
    ax_bar.axis('off')
    bar_data = {
        'Track Err (m)':  (None, None),
        'USV Motion (km)': (None, None),
    }
    bar_ax = ax_bar.inset_axes([0.05, 0.05, 0.90, 0.90])
    bar_ax.set_facecolor(C_BG)
    bar_ax.tick_params(colors=C_TEXT, labelsize=8)
    for sp in bar_ax.spines.values():
        sp.set_edgecolor(C_GRID)
    bar_ax.set_title('Live Comparison  (episode so far)', color=C_TEXT, fontsize=8.5, fontweight='bold')

    metrics_labels = ['Avg Track Err (m)', 'USV Cum. Motion (km)']
    x_pos = np.arange(len(metrics_labels))
    bar_b = bar_ax.bar(x_pos - 0.2, [0, 0], width=0.35, color=C_BASELINE, alpha=0.85, label='Baseline')
    bar_s = bar_ax.bar(x_pos + 0.2, [0, 0], width=0.35, color=C_PROPOSED, alpha=0.85, label='Proposed')
    bar_ax.set_xticks(x_pos)
    bar_ax.set_xticklabels(metrics_labels, color=C_TEXT, fontsize=8)
    bar_ax.legend(fontsize=7.5, facecolor='#0D1526', labelcolor=C_TEXT, edgecolor=C_GRID)
    bar_ax.set_ylim(0, max(cum_b[-1] / 1000 * 1.4, 0.5))

    txt_pct = bar_ax.text(0.98, 0.95, '', transform=bar_ax.transAxes,
                          color=C_TEXT, fontsize=8, ha='right', va='top',
                          bbox=dict(boxstyle='round', fc='#0D1526', ec=C_GRID, alpha=0.9))

    # ── update ───────────────────────────────────────────────────────────────
    def update(fi):
        t = frame_list[fi]
        ts = np.arange(t + 1)

        # tracking error
        if te_b_mean is not None:
            line_te_b.set_data(ts, te_b_mean[:t+1])
            line_te_s.set_data(ts, te_s_mean[:t+1])
            vl_te_b.set_xdata([t, t])
            vl_te_s.set_xdata([t, t])

        # FIM
        if dj_b_mean is not None:
            line_fim_b.set_data(ts, dj_b_mean[:t+1])
            line_fim_s.set_data(ts, dj_s_mean[:t+1])

        # USV motion
        line_usv_b.set_data(ts, cum_b[:t+1] / 1000)
        line_usv_s.set_data(ts, cum_s[:t+1] / 1000)

        # bars
        te_now_b = te_b_mean[t] if te_b_mean is not None else 0
        te_now_s = te_s_mean[t] if te_s_mean is not None else 0
        usv_now_b = cum_b[t] / 1000
        usv_now_s = cum_s[t] / 1000

        vals_b = [te_now_b, usv_now_b]
        vals_s = [te_now_s, usv_now_s]
        bar_ax.set_ylim(0, max(usv_now_b * 1.35, te_now_b * 1.35, 0.1))

        for rect, v in zip(bar_b, vals_b):
            rect.set_height(v)
        for rect, v in zip(bar_s, vals_s):
            rect.set_height(v)

        # improvement %
        if te_now_b > 1e-6:
            te_imp = (te_now_b - te_now_s) / te_now_b * 100
            usv_imp = (usv_now_b - usv_now_s) / max(usv_now_b, 1e-6) * 100
            txt_pct.set_text(
                f'Track Err reduction: {te_imp:+.1f}%\n'
                f'USV motion reduction: {usv_imp:+.1f}%'
            )

        return (line_te_b, line_te_s, line_fim_b, line_fim_s,
                line_usv_b, line_usv_s,
                *bar_b, *bar_s, txt_pct, vl_te_b, vl_te_s)

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_list),
        interval=1000 // fps, blit=True
    )
    writer = animation.PillowWriter(fps=fps, metadata={'loop': 0})
    ani.save(output_path, writer=writer, dpi=100)
    plt.close(fig)
    print(f'  ✓  Saved → {output_path}')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result',    type=str,   default=None)
    ap.add_argument('--result_dir',type=str,   default='delay_comparison_results')
    ap.add_argument('--n_auv',     type=int,   default=None)
    ap.add_argument('--output',    type=str,   default='docs/metrics_comparison.gif')
    ap.add_argument('--fps',       type=int,   default=10)
    ap.add_argument('--duration',  type=float, default=12.0)
    ap.add_argument('--delay_scenario', type=float, default=1.0)
    args = ap.parse_args()

    result_path = args.result
    if result_path is None:
        result_path = find_latest_result(args.result_dir, args.n_auv)
    if result_path is None:
        print('ERROR: No comparison result file found.')
        print('       Run compare_delay_stackelberg.py first.')
        return

    print(f'Loading: {result_path}')
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    ok = build_metrics_gif(
        result_path, args.output,
        fps=args.fps, duration_s=args.duration,
        delay_scenario=args.delay_scenario
    )
    if ok:
        size_kb = os.path.getsize(args.output) // 1024
        print(f'  GIF size: {size_kb} KB  ({args.output})')


if __name__ == '__main__':
    main()
