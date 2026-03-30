"""
Generate demonstration GIFs from real experiment data.

Produces three GIFs saved to docs/:
  1. trajectory_comparison.gif  - side-by-side USV/AUV trajectories
  2. metrics_evolution.gif      - real-time metric curves across episodes
  3. team_size_summary.gif      - animated bar chart across 2/3/4 AUVs

Usage:
    python generate_gifs.py --data_dir /path/to/delay_comparison_results
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

# ── colour palette ────────────────────────────────────────────────────────────
C_TRAD     = '#E74C3C'   # red    – Traditional baseline
C_STACK    = '#2ECC71'   # green  – Stackelberg proposed
C_USV      = '#F39C12'   # amber  – USV
C_SN       = '#9B59B6'   # purple – sensor node
C_BG       = '#0B1120'   # dark background
C_PANEL    = '#111827'   # panel bg
C_GRID     = '#1F2D45'
C_TEXT     = '#ECF0F1'
C_SUB      = '#8899AA'
AUV_COLORS = ['#3498DB', '#E67E22', '#1ABC9C', '#9B59B6']


# ── data loading ─────────────────────────────────────────────────────────────

def load_result(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_files(data_dir: str, n_auv: int, model: str = 'td3') -> list[str]:
    pattern = os.path.join(data_dir, f'**/*_{model}', '*.json')
    files = glob.glob(pattern, recursive=True)
    out = []
    for f in files:
        try:
            with open(f) as fp:
                info = json.load(fp)['experiment_info']
            if info['N_AUV'] == n_auv:
                out.append(f)
        except Exception:
            pass
    return sorted(out)


def get_episodes(data: dict, method: str) -> list:
    """Extract episode list from the nested result format."""
    results_key = [k for k in data['results'] if 'delay' in k][-1]
    inner = data['results'][results_key]
    return inner[method]['results']   # list of dicts


def get_stats(data: dict, method: str) -> dict:
    results_key = [k for k in data['results'] if 'delay' in k][-1]
    return data['results'][results_key][method]['stats']


def pick_median_episode(episodes: list) -> dict:
    """Return the episode closest to the median tracking error."""
    errors = []
    for ep in episodes:
        te = np.array(ep['tracking_error'])        # (N_AUV, T)
        errors.append(float(np.mean(te)))
    idx = int(np.argsort(errors)[len(errors) // 2])
    return episodes[idx]


def align_usv(x_usv: np.ndarray, target_len: int) -> np.ndarray:
    """Subsample USV trajectory to match AUV length."""
    if len(x_usv) == target_len:
        return x_usv
    idx = np.round(np.linspace(0, len(x_usv) - 1, target_len)).astype(int)
    return x_usv[idx]


def smooth(arr, sigma=4):
    return gaussian_filter1d(np.asarray(arr, dtype=float), sigma)


# ═══════════════════════════════════════════════════════════════════════════════
# GIF 1 – Trajectory Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def make_trajectory_gif(data_file: str, output: str,
                         fps: int = 15, duration: float = 12.0,
                         trail: int = 100):
    print(f'\n[GIF 1] Trajectory Comparison → {output}')
    data = load_result(data_file)
    n_auv = data['experiment_info']['N_AUV']

    ep_t = pick_median_episode(get_episodes(data, 'traditional'))
    ep_s = pick_median_episode(get_episodes(data, 'stackelberg'))

    T_auv = np.array(ep_t['x_auv']).shape[1]   # 1001

    xu_t = align_usv(np.array(ep_t['x_usv']), T_auv)
    yu_t = align_usv(np.array(ep_t['y_usv']), T_auv)
    xu_s = align_usv(np.array(ep_s['x_usv']), T_auv)
    yu_s = align_usv(np.array(ep_s['y_usv']), T_auv)

    xa_t = np.array(ep_t['x_auv'])   # (N_AUV, T)
    ya_t = np.array(ep_t['y_auv'])
    xa_s = np.array(ep_s['x_auv'])
    ya_s = np.array(ep_s['y_auv'])

    te_t = [smooth(ep_t['tracking_error'][i]) for i in range(n_auv)]
    te_s = [smooth(ep_s['tracking_error'][i]) for i in range(n_auv)]

    sns = np.array(ep_t.get('SoPcenter', ep_s.get('SoPcenter', [])))
    lda = ep_t.get('lda', [5] * len(sns))

    T = T_auv
    total_frames = int(fps * duration)
    step = max(1, T // total_frames)
    frame_ts = list(range(0, T, step))

    # ── figure ──
    fig = plt.figure(figsize=(16, 8), facecolor=C_BG)
    fig.patch.set_facecolor(C_BG)
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[1, 1], height_ratios=[7, 0.7],
        hspace=0.05, wspace=0.05,
        left=0.03, right=0.97, top=0.89, bottom=0.07
    )
    ax_l = fig.add_subplot(gs[0, 0], facecolor=C_PANEL)
    ax_r = fig.add_subplot(gs[0, 1], facecolor=C_PANEL)
    ax_b = fig.add_subplot(gs[1, :], facecolor=C_BG)

    for ax in (ax_l, ax_r):
        ax.set_xlim(-5, 205); ax.set_ylim(-5, 205)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(colors=C_SUB, labelsize=7)
        ax.set_xlabel('X (m)', color=C_SUB, fontsize=8)
        ax.set_ylabel('Y (m)', color=C_SUB, fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_GRID)
        ax.grid(color=C_GRID, linewidth=0.4, alpha=0.5)

    ax_b.axis('off')

    ax_l.set_title(f'Baseline  (Traditional FIM, {n_auv} AUVs)',
                   color=C_TRAD, fontsize=11, fontweight='bold', pad=6)
    ax_r.set_title(f'Proposed  (Stackelberg + Phase-Aware RL, {n_auv} AUVs)',
                   color=C_STACK, fontsize=11, fontweight='bold', pad=6)
    fig.suptitle(f'USV–AUV Collaboration under Acoustic Delay & Rayleigh Packet Loss  ({n_auv} AUVs · 1000 Steps)',
                 color=C_TEXT, fontsize=12.5, fontweight='bold', y=0.96)

    # sensor nodes
    if len(sns) > 0:
        max_lda = max(lda) if max(lda) > 0 else 1
        for ax in (ax_l, ax_r):
            for (sx, sy), ld in zip(sns, lda):
                ax.scatter(sx, sy, s=22, c=[[0.58, 0.32, 0.80]],
                           alpha=0.3 + 0.5 * ld / max_lda,
                           zorder=2, linewidths=0)

    # ── animated artists ──
    col = AUV_COLORS[:n_auv]

    trail_u_t, = ax_l.plot([], [], '-', color=C_USV, lw=1.3, alpha=0.55, zorder=3)
    trail_u_s, = ax_r.plot([], [], '-', color=C_USV, lw=1.3, alpha=0.55, zorder=3)
    tr_a_t = [ax_l.plot([], [], '-', color=col[i], lw=0.9, alpha=0.35, zorder=3)[0] for i in range(n_auv)]
    tr_a_s = [ax_r.plot([], [], '-', color=col[i], lw=0.9, alpha=0.35, zorder=3)[0] for i in range(n_auv)]

    dot_u_t = ax_l.scatter([], [], s=100, c=[C_USV], marker='D', zorder=6, linewidths=0.6, edgecolors='white')
    dot_u_s = ax_r.scatter([], [], s=100, c=[C_USV], marker='D', zorder=6, linewidths=0.6, edgecolors='white')
    dots_t  = [ax_l.scatter([], [], s=60, c=[col[i]], marker='o', zorder=5, linewidths=0.4, edgecolors='white') for i in range(n_auv)]
    dots_s  = [ax_r.scatter([], [], s=60, c=[col[i]], marker='o', zorder=5, linewidths=0.4, edgecolors='white') for i in range(n_auv)]

    def mbox(ax, x, y, s, c):
        return ax.text(x, y, s, transform=ax.transAxes, color=c,
                       fontsize=8, fontweight='bold', va='top', ha='left', zorder=10,
                       bbox=dict(boxstyle='round,pad=0.3', fc='#080E1C', ec=c, alpha=0.88))

    tb_te = mbox(ax_l, 0.02, 0.97, '', C_TRAD)
    tb_mv = mbox(ax_l, 0.02, 0.84, '', C_USV)
    ts_te = mbox(ax_r, 0.02, 0.97, '', C_STACK)
    ts_mv = mbox(ax_r, 0.02, 0.84, '', C_USV)
    ts_imp = mbox(ax_r, 0.55, 0.97, '', '#F1C40F')

    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1)
    ax_b.fill_between([0, 1], [0.35, 0.35], [0.65, 0.65], color=C_GRID, alpha=0.6)
    prog_line, = ax_b.plot([], [], '-', color=C_STACK, lw=12,
                            solid_capstyle='round', zorder=3)
    step_txt = ax_b.text(0.5, 0.5, '', transform=ax_b.transAxes,
                          color=C_TEXT, fontsize=9, ha='center', va='center', zorder=4)

    # legend
    handles = [
        mpatches.Patch(color=C_USV, label='USV (leader)'),
        *[mpatches.Patch(color=col[i], label=f'AUV {i+1}') for i in range(n_auv)],
        Line2D([0],[0], marker='o', color='none', markerfacecolor=C_SN, markersize=6, label='Sensor Node'),
    ]
    fig.legend(handles=handles, loc='lower right', ncol=len(handles),
               fontsize=8, framealpha=0.85, facecolor='#0D1526',
               labelcolor=C_TEXT, edgecolor=C_GRID,
               bbox_to_anchor=(0.97, 0.01))

    def update(fi):
        t = frame_ts[fi]
        lo = max(0, t - trail)

        trail_u_t.set_data(xu_t[lo:t+1], yu_t[lo:t+1])
        trail_u_s.set_data(xu_s[lo:t+1], yu_s[lo:t+1])
        dot_u_t.set_offsets([[xu_t[t], yu_t[t]]])
        dot_u_s.set_offsets([[xu_s[t], yu_s[t]]])
        for i in range(n_auv):
            tr_a_t[i].set_data(xa_t[i, lo:t+1], ya_t[i, lo:t+1])
            tr_a_s[i].set_data(xa_s[i, lo:t+1], ya_s[i, lo:t+1])
            dots_t[i].set_offsets([[xa_t[i, t], ya_t[i, t]]])
            dots_s[i].set_offsets([[xa_s[i, t], ya_s[i, t]]])

        te_now_t = np.mean([te_t[i][min(t, len(te_t[i])-1)] for i in range(n_auv)])
        te_now_s = np.mean([te_s[i][min(t, len(te_s[i])-1)] for i in range(n_auv)])
        dist_t = float(np.sum(np.hypot(np.diff(xu_t[:t+1]), np.diff(yu_t[:t+1]))))
        dist_s = float(np.sum(np.hypot(np.diff(xu_s[:t+1]), np.diff(yu_s[:t+1]))))

        tb_te.set_text(f'Track Err: {te_now_t:.3f} m')
        tb_mv.set_text(f'USV Dist:  {dist_t/1000:.2f} km')
        ts_te.set_text(f'Track Err: {te_now_s:.3f} m')
        ts_mv.set_text(f'USV Dist:  {dist_s/1000:.2f} km')

        te_imp = (te_now_t - te_now_s) / max(te_now_t, 1e-6) * 100
        dist_red = (dist_t - dist_s) / max(dist_t, 1e-6) * 100
        ts_imp.set_text(f'↓ {te_imp:+.1f}% Err  |  ↓ {dist_red:.0f}% Dist')

        pct = t / max(T - 1, 1)
        prog_line.set_data([0.02, 0.02 + 0.96 * pct], [0.5, 0.5])
        step_txt.set_text(f'  Step {t:4d} / {T}  ')

        return (trail_u_t, trail_u_s, dot_u_t, dot_u_s,
                *tr_a_t, *tr_a_s, *dots_t, *dots_s,
                tb_te, tb_mv, ts_te, ts_mv, ts_imp,
                prog_line, step_txt)

    ani = animation.FuncAnimation(fig, update, frames=len(frame_ts),
                                   interval=1000//fps, blit=True)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    ani.save(output, writer=animation.PillowWriter(fps=fps, metadata={'loop': 0}), dpi=100)
    plt.close(fig)
    kb = os.path.getsize(output) // 1024
    print(f'  ✓  {output}  ({kb} KB)')


# ═══════════════════════════════════════════════════════════════════════════════
# GIF 2 – Metrics Evolution (mean ± band across 50 episodes)
# ═══════════════════════════════════════════════════════════════════════════════

def make_metrics_gif(data_file: str, output: str,
                      fps: int = 15, duration: float = 12.0):
    print(f'\n[GIF 2] Metrics Evolution → {output}')
    data = load_result(data_file)
    n_auv = data['experiment_info']['N_AUV']

    eps_t = get_episodes(data, 'traditional')
    eps_s = get_episodes(data, 'stackelberg')

    def collect(eps, key, agg='mean'):
        """Collect per-step metric across all episodes, return (mean, std) arrays."""
        seqs = []
        for ep in eps:
            arr = np.array(ep[key])
            if arr.ndim == 2:
                arr = np.mean(arr, axis=0)   # average over AUVs
            seqs.append(arr)
        min_len = min(len(s) for s in seqs)
        mat = np.stack([s[:min_len] for s in seqs], axis=0)  # (N_ep, T)
        return mat.mean(0), mat.std(0), min_len

    te_t_m, te_t_s, T_te = collect(eps_t, 'tracking_error')
    te_s_m, te_s_s, _    = collect(eps_s, 'tracking_error')

    dj_t_m, dj_t_s, T_dj = collect(eps_t, 'detJ_values')
    dj_s_m, dj_s_s, _    = collect(eps_s, 'detJ_values')

    # USV cumulative distance per episode → distribution
    def usv_cumdist(eps, T_target):
        cumdists = []
        for ep in eps:
            xu = align_usv(np.array(ep['x_usv']), T_target)
            yu = align_usv(np.array(ep['y_usv']), T_target)
            cd = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xu), np.diff(yu)))])
            cumdists.append(cd)
        mat = np.stack(cumdists, axis=0)
        return mat.mean(0), mat.std(0)

    T_traj = int(np.array(eps_t[0]['x_auv']).shape[1])
    cd_t_m, cd_t_s = usv_cumdist(eps_t, T_traj)
    cd_s_m, cd_s_s = usv_cumdist(eps_s, T_traj)

    T = min(T_te, T_dj, T_traj)
    te_t_m  = smooth(te_t_m[:T],  sigma=6)
    te_s_m  = smooth(te_s_m[:T],  sigma=6)
    dj_t_m  = smooth(dj_t_m[:T],  sigma=6)
    dj_s_m  = smooth(dj_s_m[:T],  sigma=6)
    cd_t_m  = cd_t_m[:T]
    cd_s_m  = cd_s_m[:T]

    total_frames = int(fps * duration)
    step = max(1, T // total_frames)
    frame_ts = list(range(0, T, step))

    # ── figure ──
    fig = plt.figure(figsize=(14, 8), facecolor=C_BG)
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35,
                           left=0.08, right=0.96, top=0.88, bottom=0.09)
    ax_te = fig.add_subplot(gs[0, 0], facecolor=C_PANEL)
    ax_dj = fig.add_subplot(gs[0, 1], facecolor=C_PANEL)
    ax_cd = fig.add_subplot(gs[1, 0], facecolor=C_PANEL)
    ax_br = fig.add_subplot(gs[1, 1], facecolor=C_PANEL)

    fig.suptitle(
        f'Real-Time Performance Metrics  ({n_auv} AUVs · TD3 · 1000 Steps · Acoustic Delay + Packet Loss)',
        color=C_TEXT, fontsize=11, fontweight='bold', y=0.95)

    for ax, title, ylab in [
        (ax_te, 'Tracking Error',             'Error (m)'),
        (ax_dj, 'FIM Determinant  det(J)',     'det(J)'),
        (ax_cd, 'Cumulative USV Motion',       'Distance (m)'),
        (ax_br, 'Final-Step Summary',          ''),
    ]:
        ax.set_title(title, color=C_TEXT, fontsize=9, fontweight='bold', pad=4)
        ax.set_ylabel(ylab, color=C_SUB, fontsize=8)
        ax.set_xlabel('Timestep', color=C_SUB, fontsize=8)
        ax.tick_params(colors=C_SUB, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_GRID)
        ax.grid(color=C_GRID, linewidth=0.4, alpha=0.5)

    ax_br.set_xlabel('', color=C_SUB)

    # static background shading (full range)
    t_ax = np.arange(T)
    for ax, ym_t, ys_t, ym_s, ys_s in [
        (ax_te, te_t_m, te_t_s[:T], te_s_m, te_s_s[:T]),
        (ax_dj, dj_t_m, dj_t_s[:T], dj_s_m, dj_s_s[:T]),
    ]:
        ax.fill_between(t_ax, np.maximum(0, ym_t - ys_t), ym_t + ys_t,
                        color=C_TRAD, alpha=0.10)
        ax.fill_between(t_ax, np.maximum(0, ym_s - ys_s), ym_s + ys_s,
                        color=C_STACK, alpha=0.10)

    ax_cd.fill_between(t_ax, cd_t_m - cd_t_s[:T], cd_t_m + cd_t_s[:T], color=C_TRAD,  alpha=0.10)
    ax_cd.fill_between(t_ax, cd_s_m - cd_s_s[:T], cd_s_m + cd_s_s[:T], color=C_STACK, alpha=0.10)

    # animated lines
    line_te_t, = ax_te.plot([], [], color=C_TRAD,  lw=1.8, label='Baseline')
    line_te_s, = ax_te.plot([], [], color=C_STACK, lw=1.8, label='Proposed')
    line_dj_t, = ax_dj.plot([], [], color=C_TRAD,  lw=1.8, label='Baseline')
    line_dj_s, = ax_dj.plot([], [], color=C_STACK, lw=1.8, label='Proposed')
    line_cd_t, = ax_cd.plot([], [], color=C_TRAD,  lw=1.8, label='Baseline')
    line_cd_s, = ax_cd.plot([], [], color=C_STACK, lw=1.8, label='Proposed')

    for ax in (ax_te, ax_dj, ax_cd):
        ax.set_xlim(0, T)
        ax.legend(fontsize=7.5, facecolor='#0D1526', labelcolor=C_TEXT,
                  edgecolor=C_GRID, framealpha=0.85, loc='upper right')

    ax_te.set_ylim(0, max(te_t_m.max(), te_s_m.max()) * 1.15)
    ax_dj.set_ylim(0, max(dj_t_m.max(), dj_s_m.max()) * 1.15)
    ax_cd.set_ylim(0, max(cd_t_m.max(), cd_s_m.max()) * 1.10)

    # bar chart (updated each frame)
    bar_labels  = ['Track\nErr (m)', 'USV Dist\n(km)', 'detJ\n(×1e-7)']
    bar_ax_br_t = ax_br.bar(np.arange(3) - 0.2, [0,0,0], 0.38,
                             color=C_TRAD,  alpha=0.88, label='Baseline')
    bar_ax_br_s = ax_br.bar(np.arange(3) + 0.2, [0,0,0], 0.38,
                             color=C_STACK, alpha=0.88, label='Proposed')
    ax_br.set_xticks(range(3))
    ax_br.set_xticklabels(bar_labels, color=C_TEXT, fontsize=8)
    ax_br.set_ylabel('Value', color=C_SUB, fontsize=8)
    ax_br.tick_params(colors=C_SUB, labelsize=7)
    ax_br.legend(fontsize=7.5, facecolor='#0D1526', labelcolor=C_TEXT,
                 edgecolor=C_GRID, framealpha=0.85)
    imp_txts = [ax_br.text(i, 0, '', ha='center', va='bottom',
                            color='#F1C40F', fontsize=7.5, fontweight='bold', zorder=5)
                for i in range(3)]

    # value labels on each individual bar: [te_t, te_s, cd_t, cd_s, dj_t, dj_s]
    val_fmts  = ['{:.3f}', '{:.3f}', '{:.2f}km', '{:.2f}km', '{:.2f}', '{:.2f}']
    val_xpos  = [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2]
    val_txts  = [ax_br.text(xp, 0, '', ha='center', va='bottom',
                             color='#F1C40F', fontsize=6.5, fontweight='bold', zorder=6)
                 for xp in val_xpos]

    vline_te = ax_te.axvline(0, color=C_TEXT, lw=0.7, alpha=0.5, ls='--')
    vline_dj = ax_dj.axvline(0, color=C_TEXT, lw=0.7, alpha=0.5, ls='--')
    vline_cd = ax_cd.axvline(0, color=C_TEXT, lw=0.7, alpha=0.5, ls='--')

    time_txt = fig.text(0.5, 0.01, '', ha='center', color=C_SUB, fontsize=8)

    def update(fi):
        t = frame_ts[fi]
        ts = np.arange(t + 1)

        line_te_t.set_data(ts, te_t_m[:t+1])
        line_te_s.set_data(ts, te_s_m[:t+1])
        line_dj_t.set_data(ts, dj_t_m[:t+1])
        line_dj_s.set_data(ts, dj_s_m[:t+1])
        line_cd_t.set_data(ts, cd_t_m[:t+1])
        line_cd_s.set_data(ts, cd_s_m[:t+1])

        for v in (vline_te, vline_dj, vline_cd):
            v.set_xdata([t, t])

        # bar values at current t
        te_vals = [te_t_m[t], te_s_m[t]]
        dj_vals = [dj_t_m[t], dj_s_m[t]]
        cd_vals = [cd_t_m[t] / 1000, cd_s_m[t] / 1000]   # convert to km

        bar_ax_br_t[0].set_height(te_vals[0])
        bar_ax_br_s[0].set_height(te_vals[1])
        bar_ax_br_t[1].set_height(cd_vals[0])
        bar_ax_br_s[1].set_height(cd_vals[1])
        bar_ax_br_t[2].set_height(dj_vals[0] * 1e7)
        bar_ax_br_s[2].set_height(dj_vals[1] * 1e7)

        max_bar = max(te_vals[0], cd_vals[0], dj_vals[0] * 1e7, 1e-6)
        ax_br.set_ylim(0, max_bar * 1.35)

        for i, (vt, vs) in enumerate(zip(
                [te_vals[0], cd_vals[0], dj_vals[0]*1e7],
                [te_vals[1], cd_vals[1], dj_vals[1]*1e7])):
            imp = (vt - vs) / max(abs(vt), 1e-12) * 100
            h = max(vt, vs)
            sign = '↓' if imp > 0 else '↑'
            color = '#2ECC71' if imp > 0 else '#E74C3C'
            imp_txts[i].set_text(f'{sign}{abs(imp):.0f}%')
            imp_txts[i].set_position((i, h * 1.05))
            imp_txts[i].set_color(color)

        # value labels on each bar
        bar_vals = [te_vals[0], te_vals[1],
                    cd_vals[0], cd_vals[1],
                    dj_vals[0]*1e7, dj_vals[1]*1e7]
        bar_heights = [bar_ax_br_t[0].get_height(), bar_ax_br_s[0].get_height(),
                       bar_ax_br_t[1].get_height(), bar_ax_br_s[1].get_height(),
                       bar_ax_br_t[2].get_height(), bar_ax_br_s[2].get_height()]
        for j, (txt, fmt, h) in enumerate(zip(val_txts, val_fmts, bar_heights)):
            label = fmt.format(bar_vals[j])
            txt.set_text(label)
            txt.set_position((val_xpos[j], h * 1.02))

        time_txt.set_text(f'Timestep: {t} / {T}  (mean ± std over 50 runs)')

        return (line_te_t, line_te_s, line_dj_t, line_dj_s,
                line_cd_t, line_cd_s, vline_te, vline_dj, vline_cd,
                *bar_ax_br_t, *bar_ax_br_s, *imp_txts, *val_txts, time_txt)

    ani = animation.FuncAnimation(fig, update, frames=len(frame_ts),
                                   interval=1000//fps, blit=True)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    ani.save(output, writer=animation.PillowWriter(fps=fps, metadata={'loop': 0}), dpi=100)
    plt.close(fig)
    kb = os.path.getsize(output) // 1024
    print(f'  ✓  {output}  ({kb} KB)')


# ═══════════════════════════════════════════════════════════════════════════════
# GIF 3 – Team Size Summary (2 / 3 / 4 AUVs)
# ═══════════════════════════════════════════════════════════════════════════════

def make_team_summary_gif(data_dir: str, output: str,
                           fps: int = 8, duration: float = 6.0,
                           model: str = 'td3'):
    print(f'\n[GIF 3] Team Size Summary → {output}')

    configs = [2, 3, 4]
    colors_n = ['#3498DB', '#E67E22', '#2ECC71']

    # collect stats
    te_t, te_s, te_t_std, te_s_std = [], [], [], []
    mv_t, mv_s, mv_t_std, mv_s_std = [], [], [], []
    dj_t, dj_s = [], []

    for n in configs:
        files = find_files(data_dir, n, model)
        if not files:
            print(f'  WARNING: no {model} files for N_AUV={n}, skipping')
            te_t.append(0); te_s.append(0)
            te_t_std.append(0); te_s_std.append(0)
            mv_t.append(0); mv_s.append(0)
            mv_t_std.append(0); mv_s_std.append(0)
            dj_t.append(0); dj_s.append(0)
            continue

        # merge stats from all runs with this N_AUV
        all_te_t, all_te_s, all_mv_t, all_mv_s, all_dj_t, all_dj_s = [], [], [], [], [], []
        for f in files:
            d = load_result(f)
            st = get_stats(d, 'traditional')
            ss = get_stats(d, 'stackelberg')
            all_te_t.append(st['avg_tracking_error']['mean'])
            all_te_s.append(ss['avg_tracking_error']['mean'])
            all_mv_t.append(st['total_usv_move']['mean'])
            all_mv_s.append(ss['total_usv_move']['mean'])
            all_dj_t.append(st['avg_detJ']['mean'])
            all_dj_s.append(ss['avg_detJ']['mean'])

        te_t.append(np.mean(all_te_t)); te_t_std.append(np.std(all_te_t))
        te_s.append(np.mean(all_te_s)); te_s_std.append(np.std(all_te_s))
        mv_t.append(np.mean(all_mv_t) / 1000); mv_t_std.append(np.std(all_mv_t) / 1000)
        mv_s.append(np.mean(all_mv_s) / 1000); mv_s_std.append(np.std(all_mv_s) / 1000)
        dj_t.append(np.mean(all_dj_t) * 1e7)
        dj_s.append(np.mean(all_dj_s) * 1e7)

    te_imp = [(t - s) / max(t, 1e-9) * 100 for t, s in zip(te_t, te_s)]
    mv_imp = [(t - s) / max(t, 1e-9) * 100 for t, s in zip(mv_t, mv_s)]
    dj_imp = [(s - t) / max(t, 1e-9) * 100 for t, s in zip(dj_t, dj_s)]  # higher FIM is better

    # ── build animated bar frames ──
    # Animate: bars grow from 0 → final value over ~half duration, then hold
    total_frames = int(fps * duration)
    grow_frames = total_frames * 2 // 3
    hold_frames = total_frames - grow_frames

    fig = plt.figure(figsize=(14, 7), facecolor=C_BG)
    fig.suptitle(
        f'Algorithm Advantage Across Team Sizes  (TD3 · 50 Episodes each · Delay + Packet Loss)',
        color=C_TEXT, fontsize=12, fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(1, 3, wspace=0.38, left=0.07, right=0.97, top=0.86, bottom=0.14)
    ax1 = fig.add_subplot(gs[0], facecolor=C_PANEL)
    ax2 = fig.add_subplot(gs[1], facecolor=C_PANEL)
    ax3 = fig.add_subplot(gs[2], facecolor=C_PANEL)

    subtitles = ['Tracking Error (m)', 'USV Total Motion (km)', 'FIM  det(J) (×10⁻⁷)']
    for ax, st in zip((ax1, ax2, ax3), subtitles):
        ax.set_title(st, color=C_TEXT, fontsize=10, fontweight='bold', pad=5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f'{n} AUVs' for n in configs], color=C_TEXT, fontsize=9)
        ax.tick_params(colors=C_SUB, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_GRID)
        ax.grid(color=C_GRID, linewidth=0.4, alpha=0.5, axis='y')
        ax.set_axisbelow(True)

    x = np.arange(3)
    bw = 0.35

    bars1_t = ax1.bar(x - bw/2, [0]*3, bw, color=C_TRAD,  alpha=0.88, label='Baseline',  zorder=3)
    bars1_s = ax1.bar(x + bw/2, [0]*3, bw, color=C_STACK, alpha=0.88, label='Proposed', zorder=3)
    bars2_t = ax2.bar(x - bw/2, [0]*3, bw, color=C_TRAD,  alpha=0.88, label='Baseline',  zorder=3)
    bars2_s = ax2.bar(x + bw/2, [0]*3, bw, color=C_STACK, alpha=0.88, label='Proposed', zorder=3)
    bars3_t = ax3.bar(x - bw/2, [0]*3, bw, color=C_TRAD,  alpha=0.88, label='Baseline',  zorder=3)
    bars3_s = ax3.bar(x + bw/2, [0]*3, bw, color=C_STACK, alpha=0.88, label='Proposed', zorder=3)

    for ax in (ax1, ax2, ax3):
        ax.legend(fontsize=8, facecolor='#0D1526', labelcolor=C_TEXT,
                  edgecolor=C_GRID, framealpha=0.85, loc='upper left')

    ax1.set_ylim(0, max(te_t + te_s) * 1.35)
    ax2.set_ylim(0, max(mv_t + mv_s) * 1.35)
    ax3.set_ylim(0, max(dj_t + dj_s) * 1.35)

    imp_txts1 = [ax1.text(i, 0, '', ha='center', va='bottom',
                           color='#F1C40F', fontsize=8.5, fontweight='bold', zorder=5)
                 for i in range(3)]
    imp_txts2 = [ax2.text(i, 0, '', ha='center', va='bottom',
                           color='#F1C40F', fontsize=8.5, fontweight='bold', zorder=5)
                 for i in range(3)]
    imp_txts3 = [ax3.text(i, 0, '', ha='center', va='bottom',
                           color='#F1C40F', fontsize=8.5, fontweight='bold', zorder=5)
                 for i in range(3)]

    def lerp(target, fi):
        frac = min(fi / max(grow_frames - 1, 1), 1.0)
        frac = frac ** 0.5          # ease-out
        return [v * frac for v in target]

    def update(fi):
        f = lerp(te_t, fi)
        for bar, v in zip(bars1_t, f): bar.set_height(v)
        f = lerp(te_s, fi)
        for bar, v in zip(bars1_s, f): bar.set_height(v)

        f = lerp(mv_t, fi)
        for bar, v in zip(bars2_t, f): bar.set_height(v)
        f = lerp(mv_s, fi)
        for bar, v in zip(bars2_s, f): bar.set_height(v)

        f = lerp(dj_t, fi)
        for bar, v in zip(bars3_t, f): bar.set_height(v)
        f = lerp(dj_s, fi)
        for bar, v in zip(bars3_s, f): bar.set_height(v)

        frac = min(fi / max(grow_frames - 1, 1), 1.0) ** 0.5
        for i in range(3):
            h_te = max(lerp(te_t, fi)[i], lerp(te_s, fi)[i])
            h_mv = max(lerp(mv_t, fi)[i], lerp(mv_s, fi)[i])
            h_dj = max(lerp(dj_t, fi)[i], lerp(dj_s, fi)[i])
            imp_txts1[i].set_text(f'↓{te_imp[i]*frac:.0f}%')
            imp_txts1[i].set_position((i, h_te * 1.03))
            imp_txts2[i].set_text(f'↓{mv_imp[i]*frac:.0f}%')
            imp_txts2[i].set_position((i, h_mv * 1.03))
            imp_txts3[i].set_text(f'↓{dj_imp[i]*frac:.0f}%')
            imp_txts3[i].set_position((i, h_dj * 1.03))

        return (*bars1_t, *bars1_s, *bars2_t, *bars2_s,
                *bars3_t, *bars3_s, *imp_txts1, *imp_txts2, *imp_txts3)

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                   interval=1000//fps, blit=True)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    ani.save(output, writer=animation.PillowWriter(fps=fps, metadata={'loop': 0}), dpi=100)
    plt.close(fig)
    kb = os.path.getsize(output) // 1024
    print(f'  ✓  {output}  ({kb} KB)')


# ═══════════════════════════════════════════════════════════════════════════════
# GIF 4 – TD3 vs DSAC-T Comparison (Stackelberg, fixed N_AUV)
# ═══════════════════════════════════════════════════════════════════════════════

def make_td3_dsac_gif(data_dir: str, output: str, n_auv: int = 3,
                      fps: int = 15, duration: float = 12.0, trail: int = 100):
    print(f'\n[GIF 4] TD3 vs DSAC-T → {output}')

    files_td3  = find_files(data_dir, n_auv, 'td3')
    files_dsac = find_files(data_dir, n_auv, 'dsac')
    if not files_td3 or not files_dsac:
        print(f'  SKIP: missing td3 or dsac files for N_AUV={n_auv}')
        return

    data_td3  = load_result(files_td3[0])
    data_dsac = load_result(files_dsac[0])

    ep_td3  = pick_median_episode(get_episodes(data_td3,  'stackelberg'))
    ep_dsac = pick_median_episode(get_episodes(data_dsac, 'stackelberg'))

    T_auv = np.array(ep_td3['x_auv']).shape[1]

    xu_td3  = align_usv(np.array(ep_td3['x_usv']),  T_auv)
    yu_td3  = align_usv(np.array(ep_td3['y_usv']),  T_auv)
    xu_dsac = align_usv(np.array(ep_dsac['x_usv']), T_auv)
    yu_dsac = align_usv(np.array(ep_dsac['y_usv']), T_auv)

    xa_td3  = np.array(ep_td3['x_auv']);  ya_td3  = np.array(ep_td3['y_auv'])
    xa_dsac = np.array(ep_dsac['x_auv']); ya_dsac = np.array(ep_dsac['y_auv'])

    te_td3  = [smooth(ep_td3['tracking_error'][i])  for i in range(n_auv)]
    te_dsac = [smooth(ep_dsac['tracking_error'][i]) for i in range(n_auv)]

    sns = np.array(ep_td3.get('SoPcenter', ep_dsac.get('SoPcenter', [])))
    lda = ep_td3.get('lda', [5] * len(sns))

    T = T_auv
    total_frames = int(fps * duration)
    step = max(1, T // total_frames)
    frame_ts = list(range(0, T, step))

    # collect mean/std over all episodes for bottom panels
    eps_td3  = get_episodes(data_td3,  'stackelberg')
    eps_dsac = get_episodes(data_dsac, 'stackelberg')

    def collect_mean(eps, key):
        seqs = []
        for ep in eps:
            arr = np.array(ep[key])
            if arr.ndim == 2: arr = arr.mean(0)
            seqs.append(arr)
        ml = min(len(s) for s in seqs)
        mat = np.stack([s[:ml] for s in seqs])
        return mat.mean(0), mat.std(0), ml

    te_td3_m,  te_td3_s,  T_te  = collect_mean(eps_td3,  'tracking_error')
    te_dsac_m, te_dsac_s, _     = collect_mean(eps_dsac, 'tracking_error')
    dj_td3_m,  dj_td3_s,  T_dj  = collect_mean(eps_td3,  'detJ_values')
    dj_dsac_m, dj_dsac_s, _     = collect_mean(eps_dsac, 'detJ_values')

    T2 = min(T_te, T_dj, T_auv)
    te_td3_m  = smooth(te_td3_m[:T2],  sigma=6)
    te_dsac_m = smooth(te_dsac_m[:T2], sigma=6)
    dj_td3_m  = smooth(dj_td3_m[:T2],  sigma=6)
    dj_dsac_m = smooth(dj_dsac_m[:T2], sigma=6)

    def usv_cumdist_mean(eps):
        mats = []
        for ep in eps:
            xu = align_usv(np.array(ep['x_usv']), T_auv)
            yu = align_usv(np.array(ep['y_usv']), T_auv)
            cd = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xu), np.diff(yu)))])
            mats.append(cd[:T2])
        mat = np.stack(mats)
        return mat.mean(0), mat.std(0)

    cd_td3_m,  cd_td3_s  = usv_cumdist_mean(eps_td3)
    cd_dsac_m, cd_dsac_s = usv_cumdist_mean(eps_dsac)

    # ── figure ──
    C_TD3  = '#3498DB'
    C_DSAC = '#F39C12'

    fig = plt.figure(figsize=(18, 9), facecolor=C_BG)
    fig.suptitle(
        f'Stackelberg Framework — TD3 vs DSAC-T  ({n_auv} AUVs · 1000 Steps · Acoustic Delay + Packet Loss)',
        color=C_TEXT, fontsize=12, fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.30,
                           left=0.05, right=0.97, top=0.90, bottom=0.08)
    ax_l  = fig.add_subplot(gs[0, 0], facecolor=C_PANEL)
    ax_r  = fig.add_subplot(gs[0, 1], facecolor=C_PANEL)
    ax_br_top = fig.add_subplot(gs[0, 2], facecolor=C_PANEL)
    ax_te = fig.add_subplot(gs[1, 0], facecolor=C_PANEL)
    ax_dj = fig.add_subplot(gs[1, 1], facecolor=C_PANEL)
    ax_cd = fig.add_subplot(gs[1, 2], facecolor=C_PANEL)

    col = AUV_COLORS[:n_auv]

    for ax, title, c in [(ax_l, f'TD3 Trajectory  ({n_auv} AUVs)', C_TD3),
                          (ax_r, f'DSAC-T Trajectory  ({n_auv} AUVs)', C_DSAC)]:
        ax.set_title(title, color=c, fontsize=10, fontweight='bold', pad=4)
        ax.set_xlim(-5, 205); ax.set_ylim(-5, 205)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(colors=C_SUB, labelsize=7)
        ax.set_xlabel('X (m)', color=C_SUB, fontsize=8)
        ax.set_ylabel('Y (m)', color=C_SUB, fontsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(C_GRID)
        ax.grid(color=C_GRID, linewidth=0.4, alpha=0.5)

    if len(sns) > 0:
        max_lda = max(lda) if max(lda) > 0 else 1
        for ax in (ax_l, ax_r):
            for (sx, sy), ld in zip(sns, lda):
                ax.scatter(sx, sy, s=22, c=[[0.58, 0.32, 0.80]],
                           alpha=0.3 + 0.5 * ld / max_lda, zorder=2, linewidths=0)

    trail_u_l,  = ax_l.plot([], [], '-', color=C_USV, lw=1.3, alpha=0.55, zorder=3)
    trail_u_r,  = ax_r.plot([], [], '-', color=C_USV, lw=1.3, alpha=0.55, zorder=3)
    tr_a_l = [ax_l.plot([], [], '-', color=col[i], lw=0.9, alpha=0.35, zorder=3)[0] for i in range(n_auv)]
    tr_a_r = [ax_r.plot([], [], '-', color=col[i], lw=0.9, alpha=0.35, zorder=3)[0] for i in range(n_auv)]
    dot_u_l = ax_l.scatter([], [], s=100, c=[C_USV], marker='D', zorder=6, linewidths=0.6, edgecolors='white')
    dot_u_r = ax_r.scatter([], [], s=100, c=[C_USV], marker='D', zorder=6, linewidths=0.6, edgecolors='white')
    dots_l = [ax_l.scatter([], [], s=60, c=[col[i]], marker='o', zorder=5, linewidths=0.4, edgecolors='white') for i in range(n_auv)]
    dots_r = [ax_r.scatter([], [], s=60, c=[col[i]], marker='o', zorder=5, linewidths=0.4, edgecolors='white') for i in range(n_auv)]

    def mbox(ax, x, y, s, c):
        return ax.text(x, y, s, transform=ax.transAxes, color=c,
                       fontsize=8, fontweight='bold', va='top', ha='left', zorder=10,
                       bbox=dict(boxstyle='round,pad=0.3', fc='#080E1C', ec=c, alpha=0.88))

    tb_te = mbox(ax_l, 0.02, 0.97, '', C_TD3)
    tb_mv = mbox(ax_l, 0.02, 0.84, '', C_USV)
    ts_te = mbox(ax_r, 0.02, 0.97, '', C_DSAC)
    ts_mv = mbox(ax_r, 0.02, 0.84, '', C_USV)
    ts_imp = mbox(ax_r, 0.50, 0.97, '', '#F1C40F')

    # bar chart top-right
    ax_br_top.set_title('Final Summary', color=C_TEXT, fontsize=9, fontweight='bold', pad=4)
    bar_lbls = ['Track\nErr (m)', 'USV Dist\n(km)', 'detJ\n(×1e-7)']
    bars_td3  = ax_br_top.bar(np.arange(3) - 0.2, [0]*3, 0.38, color=C_TD3,  alpha=0.88, label='TD3')
    bars_dsac = ax_br_top.bar(np.arange(3) + 0.2, [0]*3, 0.38, color=C_DSAC, alpha=0.88, label='DSAC-T')
    ax_br_top.set_xticks(range(3))
    ax_br_top.set_xticklabels(bar_lbls, color=C_TEXT, fontsize=8)
    ax_br_top.tick_params(colors=C_SUB, labelsize=7)
    for sp in ax_br_top.spines.values(): sp.set_edgecolor(C_GRID)
    ax_br_top.grid(color=C_GRID, linewidth=0.4, alpha=0.5)
    ax_br_top.legend(fontsize=7.5, facecolor='#0D1526', labelcolor=C_TEXT,
                     edgecolor=C_GRID, framealpha=0.85)
    br_imp_txts = [ax_br_top.text(i, 0, '', ha='center', va='bottom',
                                   color='#F1C40F', fontsize=7.5, fontweight='bold', zorder=5)
                   for i in range(3)]

    # bottom time-series
    t_ax2 = np.arange(T2)
    for ax, ym1, ys1, ym2, ys2 in [
        (ax_te, te_td3_m, te_td3_s[:T2], te_dsac_m, te_dsac_s[:T2]),
        (ax_dj, dj_td3_m, dj_td3_s[:T2], dj_dsac_m, dj_dsac_s[:T2]),
    ]:
        ax.fill_between(t_ax2, np.maximum(0, ym1 - ys1), ym1 + ys1, color=C_TD3,  alpha=0.10)
        ax.fill_between(t_ax2, np.maximum(0, ym2 - ys2), ym2 + ys2, color=C_DSAC, alpha=0.10)

    ax_cd.fill_between(t_ax2, cd_td3_m - cd_td3_s[:T2], cd_td3_m + cd_td3_s[:T2],
                       color=C_TD3, alpha=0.10)
    ax_cd.fill_between(t_ax2, cd_dsac_m - cd_dsac_s[:T2], cd_dsac_m + cd_dsac_s[:T2],
                       color=C_DSAC, alpha=0.10)

    line_te_td3,  = ax_te.plot([], [], color=C_TD3,  lw=1.8, label='TD3')
    line_te_dsac, = ax_te.plot([], [], color=C_DSAC, lw=1.8, label='DSAC-T')
    line_dj_td3,  = ax_dj.plot([], [], color=C_TD3,  lw=1.8, label='TD3')
    line_dj_dsac, = ax_dj.plot([], [], color=C_DSAC, lw=1.8, label='DSAC-T')
    line_cd_td3,  = ax_cd.plot([], [], color=C_TD3,  lw=1.8, label='TD3')
    line_cd_dsac, = ax_cd.plot([], [], color=C_DSAC, lw=1.8, label='DSAC-T')

    for ax, title, ylab in [
        (ax_te, 'Tracking Error',       'Error (m)'),
        (ax_dj, 'FIM det(J)',           'det(J)'),
        (ax_cd, 'Cumul. USV Motion',    'Distance (m)'),
    ]:
        ax.set_title(title, color=C_TEXT, fontsize=9, fontweight='bold', pad=4)
        ax.set_ylabel(ylab, color=C_SUB, fontsize=8)
        ax.set_xlabel('Timestep', color=C_SUB, fontsize=8)
        ax.tick_params(colors=C_SUB, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(C_GRID)
        ax.grid(color=C_GRID, linewidth=0.4, alpha=0.5)
        ax.set_xlim(0, T2)
        ax.legend(fontsize=7.5, facecolor='#0D1526', labelcolor=C_TEXT,
                  edgecolor=C_GRID, framealpha=0.85, loc='upper right')

    ax_te.set_ylim(0, max(te_td3_m.max(), te_dsac_m.max()) * 1.15)
    ax_dj.set_ylim(0, max(dj_td3_m.max(), dj_dsac_m.max()) * 1.15)
    ax_cd.set_ylim(0, max(cd_td3_m.max(), cd_dsac_m.max()) * 1.10)

    vline_te = ax_te.axvline(0, color=C_TEXT, lw=0.7, alpha=0.5, ls='--')
    vline_dj = ax_dj.axvline(0, color=C_TEXT, lw=0.7, alpha=0.5, ls='--')
    vline_cd = ax_cd.axvline(0, color=C_TEXT, lw=0.7, alpha=0.5, ls='--')

    time_txt = fig.text(0.5, 0.01, '', ha='center', color=C_SUB, fontsize=8)

    def update(fi):
        t = frame_ts[fi]
        lo = max(0, t - trail)

        trail_u_l.set_data(xu_td3[lo:t+1],  yu_td3[lo:t+1])
        trail_u_r.set_data(xu_dsac[lo:t+1], yu_dsac[lo:t+1])
        dot_u_l.set_offsets([[xu_td3[t],  yu_td3[t]]])
        dot_u_r.set_offsets([[xu_dsac[t], yu_dsac[t]]])
        for i in range(n_auv):
            tr_a_l[i].set_data(xa_td3[i, lo:t+1],  ya_td3[i, lo:t+1])
            tr_a_r[i].set_data(xa_dsac[i, lo:t+1], ya_dsac[i, lo:t+1])
            dots_l[i].set_offsets([[xa_td3[i, t],  ya_td3[i, t]]])
            dots_r[i].set_offsets([[xa_dsac[i, t], ya_dsac[i, t]]])

        te_now_td3  = np.mean([te_td3[i][min(t, len(te_td3[i])-1)]  for i in range(n_auv)])
        te_now_dsac = np.mean([te_dsac[i][min(t, len(te_dsac[i])-1)] for i in range(n_auv)])
        dist_td3  = float(np.sum(np.hypot(np.diff(xu_td3[:t+1]),  np.diff(yu_td3[:t+1]))))
        dist_dsac = float(np.sum(np.hypot(np.diff(xu_dsac[:t+1]), np.diff(yu_dsac[:t+1]))))

        tb_te.set_text(f'Track Err: {te_now_td3:.3f} m')
        tb_mv.set_text(f'USV Dist:  {dist_td3/1000:.2f} km')
        ts_te.set_text(f'Track Err: {te_now_dsac:.3f} m')
        ts_mv.set_text(f'USV Dist:  {dist_dsac/1000:.2f} km')
        te_imp = (te_now_td3 - te_now_dsac) / max(te_now_td3, 1e-6) * 100
        sign = '↓' if te_imp > 0 else '↑'
        ts_imp.set_text(f'DSAC-T {sign}{abs(te_imp):.1f}% Err')

        # bar chart
        t_clip = min(t, T2 - 1)
        te_vals = [te_td3_m[t_clip],  te_dsac_m[t_clip]]
        dj_vals = [dj_td3_m[t_clip],  dj_dsac_m[t_clip]]
        cd_vals = [cd_td3_m[t_clip] / 1000, cd_dsac_m[t_clip] / 1000]

        bars_td3[0].set_height(te_vals[0]);  bars_dsac[0].set_height(te_vals[1])
        bars_td3[1].set_height(cd_vals[0]);  bars_dsac[1].set_height(cd_vals[1])
        bars_td3[2].set_height(dj_vals[0]*1e7); bars_dsac[2].set_height(dj_vals[1]*1e7)

        max_bar = max(te_vals[0], cd_vals[0], dj_vals[0]*1e7, 1e-6)
        ax_br_top.set_ylim(0, max_bar * 1.4)

        for i, (vt, vs) in enumerate(zip(
                [te_vals[0], cd_vals[0], dj_vals[0]*1e7],
                [te_vals[1], cd_vals[1], dj_vals[1]*1e7])):
            imp = (vt - vs) / max(abs(vt), 1e-12) * 100
            h = max(vt, vs)
            sign2 = '↓' if imp > 0 else '↑'
            br_imp_txts[i].set_text(f'{sign2}{abs(imp):.0f}%')
            br_imp_txts[i].set_position((i, h * 1.05))
            br_imp_txts[i].set_color('#2ECC71' if imp > 0 else '#E74C3C')

        ts2 = np.arange(t_clip + 1)
        line_te_td3.set_data(ts2, te_td3_m[:t_clip+1])
        line_te_dsac.set_data(ts2, te_dsac_m[:t_clip+1])
        line_dj_td3.set_data(ts2, dj_td3_m[:t_clip+1])
        line_dj_dsac.set_data(ts2, dj_dsac_m[:t_clip+1])
        line_cd_td3.set_data(ts2, cd_td3_m[:t_clip+1])
        line_cd_dsac.set_data(ts2, cd_dsac_m[:t_clip+1])
        for v in (vline_te, vline_dj, vline_cd): v.set_xdata([t_clip, t_clip])

        time_txt.set_text(f'Timestep: {t} / {T}  (mean ± std over 50 runs)')

        return (trail_u_l, trail_u_r, dot_u_l, dot_u_r,
                *tr_a_l, *tr_a_r, *dots_l, *dots_r,
                tb_te, tb_mv, ts_te, ts_mv, ts_imp,
                *bars_td3, *bars_dsac, *br_imp_txts,
                line_te_td3, line_te_dsac, line_dj_td3, line_dj_dsac,
                line_cd_td3, line_cd_dsac,
                vline_te, vline_dj, vline_cd, time_txt)

    ani = animation.FuncAnimation(fig, update, frames=len(frame_ts),
                                   interval=1000//fps, blit=True)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    ani.save(output, writer=animation.PillowWriter(fps=fps, metadata={'loop': 0}), dpi=100)
    plt.close(fig)
    kb = os.path.getsize(output) // 1024
    print(f'  ✓  {output}  ({kb} KB)')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='Generate demo GIFs from experiment data')
    ap.add_argument('--data_dir', type=str,
                    default='delay_comparison_results',
                    help='path to delay_comparison_results directory')
    ap.add_argument('--out_dir',  type=str, default='docs')
    ap.add_argument('--n_auv',    type=int, default=3,
                    help='AUV count used for GIF 1 & 2 (default: 3)')
    ap.add_argument('--model',    type=str, default='td3',
                    choices=['td3', 'dsac'])
    ap.add_argument('--fps',      type=int, default=15)
    ap.add_argument('--skip',     nargs='*', default=[],
                    help='skip specific GIFs: e.g. --skip 1 3')
    args = ap.parse_args()

    # pick data file for GIF 1 & 2
    files = find_files(args.data_dir, args.n_auv, args.model)
    if not files:
        print(f'ERROR: no {args.model} result files found for N_AUV={args.n_auv} in {args.data_dir}')
        sys.exit(1)
    data_file = files[0]
    print(f'Using: {data_file}')

    if '1' not in args.skip:
        make_trajectory_gif(
            data_file,
            os.path.join(args.out_dir, f'trajectory_{args.n_auv}auv.gif'),
            fps=args.fps, duration=12.0, trail=120)

    if '2' not in args.skip:
        make_metrics_gif(
            data_file,
            os.path.join(args.out_dir, f'metrics_{args.n_auv}auv.gif'),
            fps=args.fps, duration=12.0)

    if '3' not in args.skip:
        make_team_summary_gif(
            args.data_dir,
            os.path.join(args.out_dir, 'team_size_summary.gif'),
            fps=8, duration=6.0, model=args.model)

    if '4' not in args.skip:
        make_td3_dsac_gif(
            args.data_dir,
            os.path.join(args.out_dir, f'td3_vs_dsac_{args.n_auv}auv.gif'),
            n_auv=args.n_auv, fps=args.fps, duration=12.0)

    print('\nDone.  GIFs saved to:', args.out_dir)


if __name__ == '__main__':
    main()
