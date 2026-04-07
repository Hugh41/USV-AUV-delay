import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import ijson
except ModuleNotFoundError:
    ijson = None


RESULT_DIR = Path(__file__).resolve().parent.parent / "delay_comparison_results"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "td3_figure_exports"
UPDATE_FREQ = 5

MODEL_DIRS = {
    "td3": {
        2: [
            "delay_comparison_20260219_110142_5_td3",
            "delay_comparison_20260219_110153_5_td3",
            "delay_comparison_20260219_110154_5_td3",
        ],
        3: [
            "delay_comparison_20260219_155913_5_td3",
            "delay_comparison_20260219_155915_5_td3",
            "delay_comparison_20260219_155921_5_td3",
        ],
        4: [
            "delay_comparison_20260219_160051_5_td3",
            "delay_comparison_20260219_160057_5_td3",
            "delay_comparison_20260219_160059_5_td3",
        ],
    },
    "dsac": {
        2: [
            "delay_comparison_20260219_044938_5_dsac",
            "delay_comparison_20260219_044955_5_dsac",
            "delay_comparison_20260219_044957_5_dsac",
        ],
        3: [
            "delay_comparison_20260219_105843_5_dsac",
            "delay_comparison_20260219_105850_5_dsac",
            "delay_comparison_20260219_155344_5_dsac",
        ],
        4: [
            "delay_comparison_20260220_010100_5_dsac",
            "delay_comparison_20260220_010112_5_dsac",
            "delay_comparison_20260220_010113_5_dsac",
        ],
    },
}

METHOD_META = {
    "traditional": {
        "label": "Baseline",
        "color": "#f3f4f6",
        "edge": "#111111",
        "line": "#6b7280",
    },
    "stackelberg": {
        "label": "Proposed Framework",
        "color": "#111827",
        "edge": "#111111",
        "line": "#374151",
    },
}

BACKBONE_META = {
    "td3": {"label": "TD3"},
    "dsac": {"label": "DSAC"},
}

AUV_META = {
    2: {"label": "2 AUVs", "short": "2AUVs", "marker": "o"},
    3: {"label": "3 AUVs", "short": "3AUVs", "marker": "s"},
    4: {"label": "4 AUVs", "short": "4AUVs", "marker": "^"},
}

TITLE_SIZE = 18.0
LABEL_SIZE = 18.0
TICK_SIZE = 18.0
LEGEND_SIZE = 18.0
ANNOTATION_SIZE = 22.0
DENSITY_BINS = 220
TOP_PANEL_XMAX = 10.5
TOP_PANEL_YMAX = 23.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate separated delay-compensation phase-map figures."
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--delay_key", type=str, default="delay_1.0")
    return parser.parse_args()


def resolve_output_path(output_dir, filename):
    path = output_dir / filename
    if not path.exists():
        return path
    try:
        with path.open("ab"):
            return path
    except OSError:
        return output_dir / f"{path.stem}_updated{path.suffix}"


def get_plotting_modules():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.compression"] = 9
    matplotlib.rcParams["path.simplify"] = True
    matplotlib.rcParams["path.simplify_threshold"] = 0.4

    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["axes.edgecolor"] = "#1a1a1a"
    matplotlib.rcParams["axes.linewidth"] = 1.15

    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    return plt, pe, Line2D, mark_inset


def stream_result_items(json_path, delay_key, method):
    if ijson is not None:
        prefix = f"results.{delay_key}.{method}.results.item"
        with json_path.open("rb") as handle:
            for result in ijson.items(handle, prefix):
                yield result
        return

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for result in payload["results"][delay_key][method]["results"]:
        yield result


def compress_usv_track(x_usv, y_usv, target_len):
    x_arr = np.asarray(x_usv, dtype=float)
    y_arr = np.asarray(y_usv, dtype=float)
    if target_len <= 0 or x_arr.size == target_len:
        return x_arr, y_arr
    if x_arr.size % target_len == 0:
        repeat_factor = x_arr.size // target_len
        return (
            x_arr.reshape(target_len, repeat_factor).mean(axis=1),
            y_arr.reshape(target_len, repeat_factor).mean(axis=1),
        )
    sample_idx = np.linspace(0, x_arr.size - 1, target_len).round().astype(int)
    return x_arr[sample_idx], y_arr[sample_idx]


def mean_radius(points):
    center = np.mean(points, axis=0)
    return float(np.mean(np.linalg.norm(points - center, axis=1)))


def compute_window_samples(result):
    x_auv = [np.asarray(track, dtype=float) for track in result["x_auv"]]
    y_auv = [np.asarray(track, dtype=float) for track in result["y_auv"]]
    track_len = min(len(track) for track in x_auv)

    tracking_error = [np.asarray(track, dtype=float) for track in result["tracking_error"]]
    err_len = min(len(track) for track in tracking_error)
    limit = min(track_len, err_len)

    x_auv = [track[:limit] for track in x_auv]
    y_auv = [track[:limit] for track in y_auv]

    x_usv, y_usv = compress_usv_track(result["x_usv"], result["y_usv"], limit)
    x_usv = x_usv[:limit]
    y_usv = y_usv[:limit]

    samples = []
    for t_idx in range(UPDATE_FREQ, limit - UPDATE_FREQ, UPDATE_FREQ):
        phase_ratio = t_idx / float(limit - 1)
        if phase_ratio < 0.06 or phase_ratio > 0.94:
            continue

        stale_xy = np.column_stack([
            [track[t_idx - UPDATE_FREQ] for track in x_auv],
            [track[t_idx - UPDATE_FREQ] for track in y_auv]
        ])
        current_xy = np.column_stack([
            [track[t_idx] for track in x_auv],
            [track[t_idx] for track in y_auv]
        ])
        future_xy = np.column_stack([
            [track[t_idx + UPDATE_FREQ] for track in x_auv],
            [track[t_idx + UPDATE_FREQ] for track in y_auv]
        ])

        lag_drift = float(np.mean(np.linalg.norm(current_xy - stale_xy, axis=1)))
        leader_motion = float(
            np.linalg.norm(
                np.array([x_usv[t_idx + UPDATE_FREQ], y_usv[t_idx + UPDATE_FREQ]])
                - np.array([x_usv[t_idx], y_usv[t_idx]])
            )
        )
        phase_end_error = float(np.mean([track[t_idx + UPDATE_FREQ] for track in tracking_error]))
        realized_spread = mean_radius(future_xy)

        samples.append(
            {
                "lag_drift": lag_drift,
                "leader_motion": leader_motion,
                "phase_end_error": phase_end_error,
                "realized_spread": realized_spread,
            }
        )
    return samples


def build_case_summary(json_paths, delay_key, method):
    lag_drift = []
    leader_motion = []
    phase_end_error = []
    realized_spread = []

    for json_path in json_paths:
        for result in stream_result_items(json_path, delay_key, method):
            for sample in compute_window_samples(result):
                lag_drift.append(sample["lag_drift"])
                leader_motion.append(sample["leader_motion"])
                phase_end_error.append(sample["phase_end_error"])
                realized_spread.append(sample["realized_spread"])

    lag_arr = np.asarray(lag_drift, dtype=float)
    motion_arr = np.asarray(leader_motion, dtype=float)
    err_arr = np.asarray(phase_end_error, dtype=float)
    spread_arr = np.asarray(realized_spread, dtype=float)

    return {
        "lag_drift": lag_drift,
        "leader_motion": leader_motion,
        "phase_end_error": phase_end_error,
        "realized_spread": realized_spread,
        "sample_count": int(lag_arr.size),
        "mean_lag_drift": float(np.mean(lag_arr)),
        "mean_leader_motion": float(np.mean(motion_arr)),
        "mean_phase_end_error": float(np.mean(err_arr)),
        "mean_realized_spread": float(np.mean(spread_arr)),
        "std_lag_drift": float(np.std(lag_arr)),
        "std_leader_motion": float(np.std(motion_arr)),
        "source_files": [str(path) for path in json_paths],
    }


def build_dataset(delay_key):
    dataset = {}
    for backbone, case_dirs in MODEL_DIRS.items():
        dataset[backbone] = {method: {} for method in ("traditional", "stackelberg")}
        for n_auv, dir_names in case_dirs.items():
            json_paths = []
            for dir_name in dir_names:
                result_dir = RESULT_DIR / dir_name
                json_files = sorted(result_dir.glob("delay_comparison_*.json"))
                if len(json_files) != 1:
                    raise FileNotFoundError(f"Expected one json file in {result_dir}, found {len(json_files)}.")
                json_paths.append(json_files[0])
            for method in ("traditional", "stackelberg"):
                dataset[backbone][method][n_auv] = build_case_summary(json_paths, delay_key, method)
    return dataset


def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_color("#1b1b1b")
        ax.spines[spine].set_linewidth(1.05)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, colors="#111111", width=0.95, length=4.8)


def gather_case_points(case_dict):
    x_all = []
    y_all = []
    for n_auv in (2, 3, 4):
        x_all.extend(case_dict[n_auv]["lag_drift"])
        y_all.extend(case_dict[n_auv]["leader_motion"])
    return np.asarray(x_all, dtype=float), np.asarray(y_all, dtype=float)


def compute_density_grid(x_vals, y_vals, xlim, ylim):
    hist, _, _ = np.histogram2d(
        x_vals,
        y_vals,
        bins=DENSITY_BINS,
        range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
    )
    hist = gaussian_filter(hist.T, sigma=9.5)
    return hist


def prepare_heatmap_grids(dataset, xlim, ylim):
    raw_grids = {}
    global_max = 0.0

    for backbone in ("td3", "dsac"):
        for method in ("traditional", "stackelberg"):
            x_vals, y_vals = gather_case_points(dataset[backbone][method])
            hist = compute_density_grid(x_vals, y_vals, xlim, ylim)
            raw_grids[(backbone, method)] = hist
            local_max = float(np.max(hist)) if hist.size > 0 else 0.0
            global_max = max(global_max, local_max)

    if global_max <= 0:
        global_max = 1.0

    normalized = {}
    for key, hist in raw_grids.items():
        normalized[key] = np.power(hist / global_max, 0.55)

    return normalized


def draw_auv_points(ax, case_dict, pe, size=110):
    for n_auv in (2, 3, 4):
        x = case_dict[n_auv]["mean_lag_drift"]
        y = case_dict[n_auv]["mean_leader_motion"]

        sc = ax.scatter(
            x,
            y,
            s=size,
            marker=AUV_META[n_auv]["marker"],
            facecolor="#ffffff",
            edgecolor="#111111",
            linewidth=1.35,
            zorder=5,
        )
        sc.set_path_effects([
            pe.Stroke(linewidth=2.4, foreground="#000000", alpha=0.12),
            pe.Normal(),
        ])


def build_panel_legend(ax, case_dict, Line2D):
    handles = []
    for n_auv in (2, 3, 4):
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=AUV_META[n_auv]["marker"],
                markersize=9.2,
                markerfacecolor="#ffffff",
                markeredgecolor="#111111",
                markeredgewidth=1.2,
                label=(
                    f"{AUV_META[n_auv]['label']} "
                    f"({case_dict[n_auv]['mean_lag_drift']:.2f}, "
                    f"{case_dict[n_auv]['mean_leader_motion']:.2f})"
                ),
            )
        )

    legend = ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        fancybox=True,
        fontsize=LEGEND_SIZE - 1.2,
        handletextpad=0.65,
        labelspacing=0.42,
        borderpad=0.42,
        borderaxespad=0.45,
    )
    legend.get_frame().set_facecolor("#d9d9d9")
    legend.get_frame().set_edgecolor("#9ca3af")
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_alpha(0.8)
    return legend


def _clamp_interval(lo, hi, lower_bound, upper_bound):
    span = hi - lo
    if lo < lower_bound:
        hi += lower_bound - lo
        lo = lower_bound
    if hi > upper_bound:
        lo -= hi - upper_bound
        hi = upper_bound
    lo = max(lo, lower_bound)
    hi = min(hi, upper_bound)
    if hi - lo < span:
        hi = min(lo + span, upper_bound)
        lo = max(hi - span, lower_bound)
    return lo, hi


def compute_zoom_bounds(case_dict, xlim, ylim):
    pts = np.array([
        [case_dict[n]["mean_lag_drift"], case_dict[n]["mean_leader_motion"]]
        for n in (2, 3, 4)
    ], dtype=float)

    x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
    y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    half_w = max(0.60, 0.90 * max(x_max - x_min, 0.25))
    half_h = max(0.60, 0.90 * max(y_max - y_min, 0.25))

    zx0, zx1 = cx - half_w, cx + half_w
    zy0, zy1 = cy - half_h, cy + half_h

    zx0, zx1 = _clamp_interval(zx0, zx1, xlim[0], xlim[1])
    zy0, zy1 = _clamp_interval(zy0, zy1, ylim[0], ylim[1])

    return (zx0, zx1), (zy0, zy1)


def add_zoom_inset(ax, heat, case_dict, xlim, ylim, pe, mark_inset):
    (zx0, zx1), (zy0, zy1) = compute_zoom_bounds(case_dict, xlim, ylim)

    axins = ax.inset_axes([0.055, 0.515, 0.35, 0.31])
    axins.set_facecolor("white")
    for spine in axins.spines.values():
        spine.set_color("#1b1b1b")
        spine.set_linewidth(0.9)

    axins.imshow(
        heat,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="bicubic",
        alpha=0.82,
        zorder=0,
    )

    draw_auv_points(axins, case_dict, pe, size=75)

    axins.set_xlim(zx0, zx1)
    axins.set_ylim(zy0, zy1)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.grid(False)

    rectpatch, connector1, connector2 = mark_inset(
        ax,
        axins,
        loc1=3,
        loc2=4,
        fc="none",
        ec="#3f3f46",
        lw=0.85,
    )
    rectpatch.set_alpha(0.80)
    connector1.set_alpha(0.65)
    connector2.set_alpha(0.65)

    return axins


def draw_single_heatmap_panel(
    ax,
    backbone,
    method,
    case_dict,
    heat,
    xlim,
    ylim,
    plt,
    pe,
    Line2D,
    mark_inset,
    panel_annotation,
):
    style_axis(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    ax.set_yticks([0.0, 5.0, 10.0, 15.0, 20.0])

    ax.set_xlabel(
        "Follower lag drift over 5-step window (m)",
        fontsize=LABEL_SIZE,
        labelpad=6,
        color="#111111"
    )
    ax.set_ylabel(
        "Leader motion over same window (m)",
        fontsize=LABEL_SIZE,
        labelpad=8,
        color="#111111"
    )
    ax.yaxis.set_label_coords(-0.07, 0.41)

    im = ax.imshow(
        heat,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="bicubic",
        alpha=0.995,
        zorder=0,
    )

    levels = np.linspace(0.14, 0.95, 8)
    contour = ax.contour(
        np.linspace(xlim[0], xlim[1], heat.shape[1]),
        np.linspace(ylim[0], ylim[1], heat.shape[0]),
        heat,
        levels=levels,
        colors="white",
        linewidths=0.48,
        alpha=0.16,
        zorder=1,
    )
    for col in contour.collections:
        col.set_path_effects([
            pe.Stroke(linewidth=0.8, foreground="#ffffff", alpha=0.08),
            pe.Normal()
        ])

    draw_auv_points(ax, case_dict, pe)
    build_panel_legend(ax, case_dict, Line2D)
    add_zoom_inset(ax, heat, case_dict, xlim, ylim, pe, mark_inset)

    ax.text(
        0.5,
        -0.27,
        panel_annotation,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=ANNOTATION_SIZE,
        color="#111111",
    )

    return im


def build_global_error_bins(dataset, n_bins=11):
    lag = []
    for backbone in ("td3", "dsac"):
        for method in ("traditional", "stackelberg"):
            for n_auv in (2, 3, 4):
                lag.extend(dataset[backbone][method][n_auv]["lag_drift"])
    lag_arr = np.asarray(lag, dtype=float)
    lo = float(np.quantile(lag_arr, 0.03))
    hi = float(np.quantile(lag_arr, 0.97))
    return np.linspace(lo, hi, n_bins + 1)


def build_error_curve(backbone_data, method, bins):
    lag = []
    err = []
    for n_auv in (2, 3, 4):
        lag.extend(backbone_data[method][n_auv]["lag_drift"])
        err.extend(backbone_data[method][n_auv]["phase_end_error"])
    lag_arr = np.asarray(lag, dtype=float)
    err_arr = np.asarray(err, dtype=float)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = []
    sems = []
    for start, end in zip(bins[:-1], bins[1:]):
        if end == bins[-1]:
            mask = (lag_arr >= start) & (lag_arr <= end)
        else:
            mask = (lag_arr >= start) & (lag_arr < end)
        vals = err_arr[mask]
        if vals.size < 20:
            means.append(np.nan)
            sems.append(np.nan)
        else:
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals) / np.sqrt(vals.size)))
    return centers, np.asarray(means, dtype=float), np.asarray(sems, dtype=float)


def draw_error_curve_panel(ax, backbone, backbone_data, bins, pe):
    style_axis(ax)
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle="-", linewidth=0.55, color="#d8d8d8", alpha=0.45)

    curve_meta = {
        "traditional": {"color": "#1f77b4", "fill": "#aec7e8"},
        "stackelberg": {"color": "#d62728", "fill": "#ff9896"},
    }

    for method in ("traditional", "stackelberg"):
        centers, means, sems = build_error_curve(backbone_data, method, bins)
        valid = np.isfinite(means)
        color = curve_meta[method]["color"]
        fill = curve_meta[method]["fill"]

        ax.fill_between(
            centers[valid],
            np.maximum(means[valid] - sems[valid], 0.0),
            means[valid] + sems[valid],
            color=fill,
            alpha=0.30,
            zorder=1,
        )
        line = ax.plot(
            centers[valid],
            means[valid],
            color=color,
            linewidth=3.0,
            marker="s",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor="#ffffff",
            markeredgewidth=0.6,
            zorder=2,
        )[0]
        line.set_path_effects([
            pe.Stroke(linewidth=2.9, foreground="#ffffff", alpha=0.22),
            pe.Normal()
        ])

    _, trad_means, _ = build_error_curve(backbone_data, "traditional", bins)
    _, prop_means, _ = build_error_curve(backbone_data, "stackelberg", bins)
    valid = np.isfinite(trad_means) & np.isfinite(prop_means)
    avg_drop = 100.0 * (1.0 - np.mean(prop_means[valid]) / max(np.mean(trad_means[valid]), 1e-8))

    ax.text(
        0.63,
        0.90,
        f"Comparable-drift error ↓ {avg_drop:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOTATION_SIZE,
        color="#1f2937",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#ffffff",
            edgecolor="#bfc5cc",
            linewidth=0.9,
            alpha=0.55,
        ),
    )
    ax.set_xlabel("Matched lag-drift level (m)", fontsize=LABEL_SIZE - 1.0, labelpad=4)
    ax.set_ylabel("Phase-end tracking error (m)", fontsize=LABEL_SIZE - 1.0, labelpad=4)
    ax.set_xlim(float(bins[0]), float(bins[-1]))


def build_curve_legend(fig, Line2D):
    handles = [
        Line2D(
            [0], [0],
            color="#1f77b4",
            linewidth=3.0,
            marker="s",
            markersize=8,
            markerfacecolor="#1f77b4",
            markeredgecolor="#ffffff",
            markeredgewidth=0.6,
            label=METHOD_META["traditional"]["label"],
        ),
        Line2D(
            [0], [0],
            color="#d62728",
            linewidth=3.0,
            marker="s",
            markersize=8,
            markerfacecolor="#d62728",
            markeredgecolor="#ffffff",
            markeredgewidth=0.6,
            label=METHOD_META["stackelberg"]["label"],
        ),
    ]
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=1.5,
        handletextpad=0.8,
    )
    legend.get_frame().set_facecolor("#ffffff")
    legend.get_frame().set_edgecolor("#9ca3af")
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_alpha(0.5)


def plot_heatmap_figure(dataset, output_path):
    plt, pe, Line2D, mark_inset = get_plotting_modules()
    xlim = (0.0, TOP_PANEL_XMAX)
    ylim = (0.0, TOP_PANEL_YMAX)

    heatmaps = prepare_heatmap_grids(dataset, xlim, ylim)

    fig = plt.figure(figsize=(18.0, 9.0), facecolor="white")
    grid = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.0, 0.060],
        wspace=0.20,
        hspace=0.39
    )

    ax_tl = fig.add_subplot(grid[0, 0])
    ax_tr = fig.add_subplot(grid[0, 1])
    ax_bl = fig.add_subplot(grid[1, 0])
    ax_br = fig.add_subplot(grid[1, 1])
    cax = fig.add_subplot(grid[:, 2])

    im = draw_single_heatmap_panel(
        ax_tl,
        "td3",
        "traditional",
        dataset["td3"]["traditional"],
        heatmaps[("td3", "traditional")],
        xlim,
        ylim,
        plt,
        pe,
        Line2D,
        mark_inset,
        panel_annotation="(a) Baseline with TD3",
    )

    draw_single_heatmap_panel(
        ax_tr,
        "td3",
        "stackelberg",
        dataset["td3"]["stackelberg"],
        heatmaps[("td3", "stackelberg")],
        xlim,
        ylim,
        plt,
        pe,
        Line2D,
        mark_inset,
        panel_annotation="(b) Proposed Framework with TD3",
    )

    draw_single_heatmap_panel(
        ax_bl,
        "dsac",
        "traditional",
        dataset["dsac"]["traditional"],
        heatmaps[("dsac", "traditional")],
        xlim,
        ylim,
        plt,
        pe,
        Line2D,
        mark_inset,
        panel_annotation="(c) Baseline with DSAC-T",
    )

    draw_single_heatmap_panel(
        ax_br,
        "dsac",
        "stackelberg",
        dataset["dsac"]["stackelberg"],
        heatmaps[("dsac", "stackelberg")],
        xlim,
        ylim,
        plt,
        pe,
        Line2D,
        mark_inset,
        panel_annotation="(d) Proposed Framework with DSAC-T",
    )

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Normalized Density", fontsize=LABEL_SIZE, rotation=90, labelpad=10)
    cb.ax.tick_params(labelsize=TICK_SIZE)
    cb.outline.set_linewidth(1.0)
    cb.outline.set_edgecolor("#1b1b1b")

    fig.subplots_adjust(left=0.10, right=0.935, top=0.93, bottom=0.12)

    # 只单独把 colorbar 往左移动一点点，不影响其他子图间距
    pos = cax.get_position()
    shift_left = 0.023  # 可改成 0.005 / 0.010 / 0.012 试一下
    cax.set_position([pos.x0 - shift_left, pos.y0, pos.width, pos.height])

    fig.savefig(output_path, dpi=420, bbox_inches="tight")
    plt.close(fig)


def plot_curve_figure(dataset, output_path):
    plt, pe, Line2D, mark_inset = get_plotting_modules()
    error_bins = build_global_error_bins(dataset)

    fig = plt.figure(figsize=(18.0, 4.8), facecolor="#ffffff")
    grid = fig.add_gridspec(1, 2, wspace=0.15)

    ax_td3_curve = fig.add_subplot(grid[0, 0])
    ax_dsac_curve = fig.add_subplot(grid[0, 1])

    draw_error_curve_panel(ax_td3_curve, "td3", dataset["td3"], error_bins, pe)
    draw_error_curve_panel(ax_dsac_curve, "dsac", dataset["dsac"], error_bins, pe)

    # subcaptions under each curve subplot
    ax_td3_curve.text(
        0.5,
        -0.24,
        "(a) Phase-end tracking error under matched lag drift (TD3)",
        transform=ax_td3_curve.transAxes,
        ha="center",
        va="top",
        fontsize=ANNOTATION_SIZE,
        color="#111111",
    )

    ax_dsac_curve.text(
        0.5,
        -0.24,
        "(b) Phase-end tracking error under matched lag drift (DSAC-T)",
        transform=ax_dsac_curve.transAxes,
        ha="center",
        va="top",
        fontsize=ANNOTATION_SIZE,
        color="#111111",
    )

    build_curve_legend(fig, Line2D)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.22)
    fig.savefig(output_path, dpi=420, bbox_inches="tight")
    plt.close(fig)


def build_manifest(dataset, delay_key):
    manifest = {
        "delay_key": delay_key,
        "figure_type": "delay_compensation_phase_map",
        "phase_window": UPDATE_FREQ,
        "axes": {
            "x": "Follower lag-induced drift over 5-step window (m)",
            "y": "Leader compensation motion over same window (m)",
        },
        "backbones": {},
    }
    for backbone in ("td3", "dsac"):
        manifest["backbones"][backbone] = {}
        for method in ("traditional", "stackelberg"):
            manifest["backbones"][backbone][method] = {}
            for n_auv in (2, 3, 4):
                case = dataset[backbone][method][n_auv]
                manifest["backbones"][backbone][method][str(n_auv)] = {
                    "sample_count": case["sample_count"],
                    "mean_lag_drift": case["mean_lag_drift"],
                    "mean_leader_motion": case["mean_leader_motion"],
                    "mean_phase_end_error": case["mean_phase_end_error"],
                    "mean_realized_spread": case["mean_realized_spread"],
                    "source_files": case["source_files"],
                }
    return manifest


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args.delay_key)

    heatmap_output_path = resolve_output_path(
        args.output_dir,
        "delay_compensation_phase_map_heatmap_2x2_with_colorbar_and_auv_points.pdf"
    )
    curve_output_path = resolve_output_path(
        args.output_dir,
        "delay_compensation_phase_map_curve_only.pdf"
    )

    plot_heatmap_figure(dataset, heatmap_output_path)
    plot_curve_figure(dataset, curve_output_path)

    manifest = build_manifest(dataset, args.delay_key)
    manifest_path = args.output_dir / "delay_compensation_phase_map_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Generated separated figures:")
    print(f"  - Heatmap PDF: {heatmap_output_path}")
    print(f"  - Curve PDF:   {curve_output_path}")
    print(f"  - Manifest:    {manifest_path}")


if __name__ == "__main__":
    main()