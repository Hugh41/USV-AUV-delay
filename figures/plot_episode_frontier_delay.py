import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from matplotlib.path import Path as MplPath

try:
    import ijson
except ModuleNotFoundError:
    ijson = None


RESULT_DIR = Path(__file__).resolve().parent / "delay_comparison_results"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "td3_figure_exports"

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
    "traditional": {"label": "Baseline", "color": "#1f77b4", "cloud_color": "#1f77b4"},
    "stackelberg": {"label": "Proposed Framework", "color": "#d62728", "cloud_color": "#d62728"},
}

BACKBONE_META = {
    "td3": {"label": "Mobility–accuracy distribution", "panel_label": "(a)"},
    "dsac": {"label": "Mobility–accuracy distribution", "panel_label": "(b)"},
}

AUV_META = {
    2: {"label": "2 AUVs", "marker": "o"},
    3: {"label": "3 AUVs", "marker": "s"},
    4: {"label": "4 AUVs", "marker": "^"},
}

LABEL_SIZE = 18
TICK_SIZE = 18
LEGEND_SIZE = 18
SUBCAPTION_SIZE = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate episode-level mobility-accuracy-throughput frontier figures under delayed communication."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the generated figures.",
    )
    parser.add_argument(
        "--delay_key",
        type=str,
        default="delay_1.0",
        help="Delay result key to analyze.",
    )
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
    matplotlib.rcParams["font.family"] = "Times New Roman"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    return plt, mcolors, pe, Line2D


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


def compute_team_mean_tracking_error(result):
    if "avg_tracking_error" in result and result["avg_tracking_error"]:
        values = np.asarray(result["avg_tracking_error"], dtype=float)
        return float(np.mean(values))

    tracking_error = np.asarray(result["tracking_error"], dtype=float)
    if tracking_error.ndim == 2:
        return float(np.mean(tracking_error))
    return float(np.mean(tracking_error.ravel()))


def build_case_summary(json_paths, delay_key, method):
    x_vals = []
    y_vals = []
    throughput_vals = []
    packet_loss_vals = []

    for json_path in json_paths:
        for result in stream_result_items(json_path, delay_key, method):
            x_vals.append(float(result["avg_usv_move"]))
            y_vals.append(compute_team_mean_tracking_error(result))
            throughput_vals.append(float(result["sum_rate"]))
            packet_loss_vals.append(float(result.get("avg_packet_loss_rate", 0.0)) * 100.0)

    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    throughput_arr = np.asarray(throughput_vals, dtype=float)
    covariance = np.cov(np.column_stack([x_arr, y_arr]).T) if x_arr.size > 1 else np.zeros((2, 2), dtype=float)
    eigvals = np.linalg.eigvalsh(covariance) if x_arr.size > 1 else np.zeros(2, dtype=float)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    ellipse_area_1sigma = float(math.pi * math.sqrt(eigvals[0]) * math.sqrt(eigvals[1])) if x_arr.size > 1 else 0.0

    return {
        "x": x_vals,
        "y": y_vals,
        "throughput": throughput_vals,
        "packet_loss_pp": packet_loss_vals,
        "run_count": int(x_arr.size),
        "mean_x": float(np.mean(x_arr)),
        "mean_y": float(np.mean(y_arr)),
        "mean_throughput": float(np.mean(throughput_arr)),
        "std_x": float(np.std(x_arr)),
        "std_y": float(np.std(y_arr)),
        "std_throughput": float(np.std(throughput_arr)),
        "ellipse_area_1sigma": ellipse_area_1sigma,
        "source_files": [str(path.with_suffix(".pkl")) for path in json_paths],
    }


def build_dataset(delay_key):
    dataset = {}
    for backbone, case_dirs in MODEL_DIRS.items():
        dataset[backbone] = {}
        for method in ("traditional", "stackelberg"):
            dataset[backbone][method] = {}

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
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.8, color="#dce1ea", alpha=0.95, zorder=0)

    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#232323")
        ax.spines[spine].set_linewidth(1.05)

    ax.tick_params(axis="both", labelsize=TICK_SIZE, colors="#1d1d1d", width=0.9, length=5)


def style_bar_axis(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.8, color="#dce1ea", alpha=0.95, zorder=0)

    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#232323")
        ax.spines[spine].set_linewidth(1.05)

    ax.tick_params(axis="both", labelsize=14, colors="#1d1d1d", width=0.9, length=5)


def collect_axis_limits(dataset):
    limits = {}
    for backbone in ("td3", "dsac"):
        all_x = []
        all_y = []
        for method in ("traditional", "stackelberg"):
            for n_auv in (2, 3, 4):
                all_x.extend(dataset[backbone][method][n_auv]["x"])
                all_y.extend(dataset[backbone][method][n_auv]["y"])
        x_arr = np.asarray(all_x, dtype=float)
        y_arr = np.asarray(all_y, dtype=float)
        x_margin = max(0.08, (float(np.max(x_arr)) - float(np.min(x_arr))) * 0.06)
        y_margin = max(0.008, (float(np.max(y_arr)) - float(np.min(y_arr))) * 0.08)
        y_lower = max(0.02, float(np.min(y_arr)) - y_margin)
        limits[backbone] = {
            "xlim": (float(np.min(x_arr)) - x_margin, float(np.max(x_arr)) + x_margin),
            "ylim": (y_lower, float(np.max(y_arr)) + y_margin),
        }
    return limits


def compute_size_scaler(dataset):
    all_rates = []
    for backbone in dataset.values():
        for method in backbone.values():
            for case in method.values():
                all_rates.extend(case["throughput"])
    rate_arr = np.asarray(all_rates, dtype=float)
    rate_arr = np.clip(rate_arr, a_min=0.0, a_max=None)
    sqrt_min = float(np.min(np.sqrt(rate_arr)))
    sqrt_max = float(np.max(np.sqrt(rate_arr)))

    def scale(values, min_size, max_size):
        values_arr = np.asarray(values, dtype=float)
        sqrt_vals = np.sqrt(np.clip(values_arr, a_min=0.0, a_max=None))
        if sqrt_max <= sqrt_min + 1e-12:
            return np.full_like(values_arr, (min_size + max_size) / 2.0, dtype=float)
        weights = (sqrt_vals - sqrt_min) / (sqrt_max - sqrt_min)
        return min_size + weights * (max_size - min_size)

    return scale


def build_density_field(x_vals, y_vals, xlim, ylim, bins=120, sigma=2.1):
    hist, x_edges, y_edges = np.histogram2d(x_vals, y_vals, bins=bins, range=[xlim, ylim])
    density = gaussian_filter(hist.T.astype(float), sigma=sigma, mode="nearest")
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    return xx, yy, density


def add_density_layers(ax, x_vals, y_vals, color, xlim, ylim, mcolors):
    xx, yy, density = build_density_field(x_vals, y_vals, xlim, ylim)
    positive = density[density > 0]
    if positive.size < 4:
        return

    max_density = float(np.max(positive))
    min_density = max(float(np.quantile(positive, 0.55)), max_density * 0.18)
    levels = np.linspace(min_density, max_density, 4)
    rgb = mcolors.to_rgb(color)
    fill_colors = [
        (*rgb, 0.06),
        (*rgb, 0.10),
        (*rgb, 0.15),
    ]
    ax.contourf(xx, yy, density, levels=levels, colors=fill_colors, antialiased=True, zorder=1)
    ax.contour(xx, yy, density, levels=levels[1:], colors=[(*rgb, 0.34)], linewidths=1.0, zorder=2)


def add_better_region_glow(ax, xlim, ylim):
    x_grid = np.linspace(0.0, 1.0, 220)
    y_grid = np.linspace(0.0, 1.0, 220)
    xx, yy = np.meshgrid(x_grid, y_grid)
    field = np.exp(-3.6 * (xx**1.15 + yy**1.15))
    rgba = np.zeros((field.shape[0], field.shape[1], 4), dtype=float)
    rgba[..., 0] = 0.85
    rgba[..., 1] = 0.95
    rgba[..., 2] = 0.89
    rgba[..., 3] = 0.16 * field
    ax.imshow(
        rgba,
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        interpolation="bicubic",
        aspect="auto",
        zorder=0.2,
    )


def chaikin_smoothing(points, iterations=2):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts

    closed = np.vstack([pts, pts[0]])
    for _ in range(iterations):
        new_pts = []
        for i in range(len(closed) - 1):
            p0 = closed[i]
            p1 = closed[i + 1]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.extend([q, r])
        new_pts = np.asarray(new_pts, dtype=float)
        closed = np.vstack([new_pts, new_pts[0]])
    return closed[:-1]


def build_closed_spline(points, smooth_factor=0.0, n_interp=360):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None

    pts_closed = np.vstack([pts, pts[0]])
    try:
        tck, _ = splprep(
            [pts_closed[:, 0], pts_closed[:, 1]],
            s=smooth_factor,
            per=True,
        )
        u_new = np.linspace(0.0, 1.0, n_interp, endpoint=False)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack([x_new, y_new])
    except Exception:
        return pts


def polygon_contains_all_points(polygon_pts, points, tol=1e-9):
    if polygon_pts is None or len(polygon_pts) < 3:
        return False

    poly_path = MplPath(np.vstack([polygon_pts, polygon_pts[0]]))
    pts = np.asarray(points, dtype=float)
    return np.all(poly_path.contains_points(pts, radius=-tol))


def compute_smooth_outer_polygon(
    x_vals,
    y_vals,
    base_expand_ratio=0.04,
    expand_step=0.02,
    max_expand_ratio=0.40,
    chaikin_iter=2,
    spline_smooth=0.0,
    n_interp=360,
):
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    if x_arr.size < 3:
        return None

    points = np.column_stack([x_arr, y_arr])
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return None

    try:
        hull = ConvexHull(points)
    except Exception:
        return None

    hull_pts = points[hull.vertices]
    if len(hull_pts) < 3:
        return None

    center = np.mean(hull_pts, axis=0, keepdims=True)
    best_polygon = None
    expand_ratio = base_expand_ratio

    while expand_ratio <= max_expand_ratio + 1e-12:
        expanded = center + (1.0 + expand_ratio) * (hull_pts - center)

        smooth_poly = chaikin_smoothing(expanded, iterations=chaikin_iter)
        smooth_poly = build_closed_spline(
            smooth_poly,
            smooth_factor=spline_smooth,
            n_interp=n_interp,
        )

        if smooth_poly is None or len(smooth_poly) < 3:
            expand_ratio += expand_step
            continue

        if polygon_contains_all_points(smooth_poly, points):
            best_polygon = smooth_poly
            break

        expand_ratio += expand_step

    if best_polygon is None:
        expanded = center + (1.0 + max_expand_ratio) * (hull_pts - center)
        best_polygon = expanded

    return np.vstack([best_polygon, best_polygon[0]])


def add_irregular_region(
    ax,
    x_vals,
    y_vals,
    color,
    mcolors,
    pe,
    fill_alpha=0.09,
    edge_alpha=0.60,
    base_expand_ratio=0.04,
):
    from matplotlib.patches import Polygon

    polygon_pts = compute_smooth_outer_polygon(
        x_vals,
        y_vals,
        base_expand_ratio=base_expand_ratio,
        expand_step=0.02,
        max_expand_ratio=0.40,
        chaikin_iter=2,
        spline_smooth=0.0,
        n_interp=360,
    )
    if polygon_pts is None:
        return

    rgb = mcolors.to_rgb(color)

    filled = Polygon(
        polygon_pts,
        closed=True,
        facecolor=(*rgb, fill_alpha),
        edgecolor="none",
        zorder=2.6,
        joinstyle="round",
    )

    edge = Polygon(
        polygon_pts,
        closed=True,
        facecolor="none",
        edgecolor=(*rgb, edge_alpha),
        linewidth=1.9,
        zorder=4.6,
        joinstyle="round",
    )
    edge.set_path_effects([
        pe.Stroke(linewidth=3.2, foreground="#ffffff", alpha=0.72),
        pe.Normal()
    ])

    ax.add_patch(filled)
    ax.add_patch(edge)


def mute_color(color, mcolors, mix=0.22):
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    neutral = np.array([0.58, 0.58, 0.58], dtype=float)
    muted = (1.0 - mix) * rgb + mix * neutral
    return tuple(np.clip(muted, 0.0, 1.0))


def draw_matte_marker(ax, x_val, y_val, size, marker, color, mcolors):
    matte_color = mute_color(color, mcolors, mix=0.10)
    ax.scatter(
        [x_val],
        [y_val],
        s=size,
        marker=marker,
        c=[matte_color],
        edgecolors="#ffffff",
        linewidths=0.75,
        alpha=0.95,
        zorder=7,
    )


def draw_pair_arrow(ax, point_a, point_b, pe, color="#6E7E95", alpha=0.78):
    from matplotlib.patches import FancyArrowPatch

    x1, y1 = point_a
    x2, y2 = point_b

    if x1 >= x2:
        start_xy = (x1, y1)
        end_xy = (x2, y2)
    else:
        start_xy = (x2, y2)
        end_xy = (x1, y1)

    arrow = FancyArrowPatch(
        start_xy,
        end_xy,
        arrowstyle="-|>",
        mutation_scale=11.5,
        color=color,
        linewidth=1.35,
        alpha=alpha,
        zorder=10,
        shrinkA=8,
        shrinkB=8,
        connectionstyle="arc3,rad=0.0",
    )
    arrow.set_path_effects([
        pe.Stroke(linewidth=2.5, foreground="#ffffff", alpha=0.65),
        pe.Normal()
    ])
    ax.add_patch(arrow)


def annotate_bar_values(ax, xs, vals, ylim_top, fmt="{:.2f}"):
    offset = 0.03 * ylim_top
    for x, v in zip(xs, vals):
        ax.text(
            x,
            v + offset,
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=12,
            color="#333333",
        )


def plot_grouped_metric_bar(
    ax,
    categories,
    baseline_means,
    proposed_means,
    baseline_errs,
    proposed_errs,
    title,
    ylabel,
    baseline_color="#1f77b4",
    proposed_color="#d62728",
    value_fmt="{:.2f}",
):
    style_bar_axis(ax)

    x = np.arange(len(categories), dtype=float)
    width = 0.30

    ax.bar(
        x - width / 2,
        baseline_means,
        width=width,
        color=baseline_color,
        alpha=0.24,
        edgecolor="none",
        zorder=1,
    )
    ax.bar(
        x + width / 2,
        proposed_means,
        width=width,
        color=proposed_color,
        alpha=0.24,
        edgecolor="none",
        zorder=1,
    )

    ax.errorbar(
        x - width / 2,
        baseline_means,
        yerr=baseline_errs,
        fmt="o-",
        color=baseline_color,
        linewidth=1.6,
        markersize=4.0,
        capsize=3,
        zorder=3,
        label="Baseline",
    )
    ax.errorbar(
        x + width / 2,
        proposed_means,
        yerr=proposed_errs,
        fmt="o-",
        color=proposed_color,
        linewidth=1.6,
        markersize=4.0,
        capsize=3,
        zorder=3,
        label="Proposed Framework",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=18)
    ax.set_title(title, fontsize=18, pad=8)
    ax.set_xlabel("Number of AUVs", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)

    y_max = max(
        [m + e for m, e in zip(baseline_means, baseline_errs)] +
        [m + e for m, e in zip(proposed_means, proposed_errs)]
    )
    ax.set_ylim(0, y_max * 1.10)

    annotate_bar_values(ax, x - width / 2, baseline_means, ax.get_ylim()[1], fmt=value_fmt)
    annotate_bar_values(ax, x + width / 2, proposed_means, ax.get_ylim()[1], fmt=value_fmt)


def plot_combined_figure(dataset, output_path, bar_backbone="td3"):
    plt, mcolors, pe, Line2D = get_plotting_modules()
    size_scale = compute_size_scaler(dataset)
    limits = collect_axis_limits(dataset)

    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.5))
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.00, top=0.82, wspace=0.27)

    # 左边两张：散点图
    for ax, backbone in zip(axes[:2], ("td3", "dsac")):
        style_axis(ax)
        xlim = limits[backbone]["xlim"]
        ylim = limits[backbone]["ylim"]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.set_xlabel("Average USV motion (km)", fontsize=LABEL_SIZE)
        ax.set_ylabel("Mean tracking error (m)", fontsize=LABEL_SIZE)
        add_better_region_glow(ax, xlim, ylim)

        for method in ("traditional", "stackelberg"):
            all_x = []
            all_y = []
            for n_auv in (2, 3, 4):
                all_x.extend(dataset[backbone][method][n_auv]["x"])
                all_y.extend(dataset[backbone][method][n_auv]["y"])
            add_density_layers(ax, all_x, all_y, METHOD_META[method]["color"], xlim, ylim, mcolors)
            add_irregular_region(
                ax,
                all_x,
                all_y,
                METHOD_META[method]["color"],
                mcolors,
                pe,
                fill_alpha=0.09,
                edge_alpha=0.60,
                base_expand_ratio=0.04,
            )

        for method in ("traditional", "stackelberg"):
            for n_auv in (2, 3, 4):
                case = dataset[backbone][method][n_auv]
                ax.scatter(
                    case["x"],
                    case["y"],
                    s=size_scale(case["throughput"], 14.0, 58.0),
                    marker=AUV_META[n_auv]["marker"],
                    c=METHOD_META[method]["cloud_color"],
                    alpha=0.14,
                    edgecolors="none",
                    zorder=3,
                )

        for n_auv in (2, 3, 4):
            pt_trad = (
                dataset[backbone]["traditional"][n_auv]["mean_x"],
                dataset[backbone]["traditional"][n_auv]["mean_y"],
            )
            pt_stack = (
                dataset[backbone]["stackelberg"][n_auv]["mean_x"],
                dataset[backbone]["stackelberg"][n_auv]["mean_y"],
            )
            draw_pair_arrow(ax, pt_trad, pt_stack, pe, color="#6D7D93", alpha=0.80)

        for method in ("traditional", "stackelberg"):
            mean_x = [dataset[backbone][method][n]["mean_x"] for n in (2, 3, 4)]
            mean_y = [dataset[backbone][method][n]["mean_y"] for n in (2, 3, 4)]
            line = ax.plot(mean_x, mean_y, color=METHOD_META[method]["color"], linewidth=2.4, alpha=0.94, zorder=5)[0]
            line.set_path_effects([pe.Stroke(linewidth=3.5, foreground="#ffffff", alpha=0.64), pe.Normal()])

            for n_auv in (2, 3, 4):
                case = dataset[backbone][method][n_auv]
                bubble_size = size_scale([case["mean_throughput"]], 118.0, 250.0)[0]
                draw_matte_marker(
                    ax,
                    case["mean_x"],
                    case["mean_y"],
                    bubble_size,
                    AUV_META[n_auv]["marker"],
                    METHOD_META[method]["color"],
                    mcolors,
                )

        ax.text(
            0.5,
            -0.20,
            f"{BACKBONE_META[backbone]['panel_label']} {BACKBONE_META[backbone]['label']}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=SUBCAPTION_SIZE,
        )

    # 右边两张：柱状图
    categories = ["2 AUVs", "3 AUVs", "4 AUVs"]
    auv_cases = (2, 3, 4)

    motion_baseline = [dataset[bar_backbone]["traditional"][n]["mean_x"] for n in auv_cases]
    motion_proposed = [dataset[bar_backbone]["stackelberg"][n]["mean_x"] for n in auv_cases]
    motion_baseline_err = [dataset[bar_backbone]["traditional"][n]["std_x"] for n in auv_cases]
    motion_proposed_err = [dataset[bar_backbone]["stackelberg"][n]["std_x"] for n in auv_cases]

    error_baseline = [dataset[bar_backbone]["traditional"][n]["mean_y"] for n in auv_cases]
    error_proposed = [dataset[bar_backbone]["stackelberg"][n]["mean_y"] for n in auv_cases]
    error_baseline_err = [dataset[bar_backbone]["traditional"][n]["std_y"] for n in auv_cases]
    error_proposed_err = [dataset[bar_backbone]["stackelberg"][n]["std_y"] for n in auv_cases]

    plot_grouped_metric_bar(
        axes[2],
        categories,
        motion_baseline,
        motion_proposed,
        motion_baseline_err,
        motion_proposed_err,
        title="",
        ylabel="Mean USV motion (km)",
        value_fmt="{:.2f}",
    )

    plot_grouped_metric_bar(
        axes[3],
        categories,
        error_baseline,
        error_proposed,
        error_baseline_err,
        error_proposed_err,
        title="",
        ylabel="Mean tracking error (m)",
        value_fmt="{:.3f}",
    )

    # 右边两张图的 subcaption
    axes[2].text(
        0.5,
        -0.20,
        "(c) Mean USV motion",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=SUBCAPTION_SIZE,
    )
    axes[3].text(
        0.5,
        -0.20,
        "(d) Mean tracking error",
        transform=axes[3].transAxes,
        ha="center",
        va="top",
        fontsize=SUBCAPTION_SIZE,
    )

    # 左侧 legend：放在左边两张图中央正上方
    left_legend_handles = [
        Line2D(
            [0], [0],
            color=METHOD_META["traditional"]["color"],
            marker="o",
            linewidth=0,
            markersize=9.5,
            label="Baseline"
        ),
        Line2D(
            [0], [0],
            color=METHOD_META["stackelberg"]["color"],
            marker="o",
            linewidth=0,
            markersize=9.5,
            label="Proposed Framework"
        ),
        Line2D(
            [0], [0],
            color="#151515",
            marker=AUV_META[2]["marker"],
            markerfacecolor="#f3f3f3",
            markeredgewidth=1.2,
            linewidth=0,
            markersize=9.2,
            label="2 AUVs",
        ),
        Line2D(
            [0], [0],
            color="#151515",
            marker=AUV_META[3]["marker"],
            markerfacecolor="#f3f3f3",
            markeredgewidth=1.2,
            linewidth=0,
            markersize=9.2,
            label="3 AUVs",
        ),
        Line2D(
            [0], [0],
            color="#151515",
            marker=AUV_META[4]["marker"],
            markerfacecolor="#f3f3f3",
            markeredgewidth=1.2,
            linewidth=0,
            markersize=9.2,
            label="4 AUVs",
        ),
    ]

    left_legend = fig.legend(
        handles=left_legend_handles,
        loc="upper center",
        ncol=5,
        frameon=True,
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.285, 0.985),
        fancybox=True,
        framealpha=0.96,
        borderpad=0.50,
        columnspacing=0.50,
        handletextpad=0.10,
    )
    left_legend.get_frame().set_facecolor("#ffffff")
    left_legend.get_frame().set_edgecolor("#d5dbe5")
    left_legend.get_frame().set_linewidth(1.0)

    # 右侧 legend：放在右边两张图中央正上方
    right_legend_handles = [
        Line2D(
            [0], [0],
            color=METHOD_META["traditional"]["color"],
            marker="o",
            linewidth=1.6,
            markersize=5.0,
            label="Baseline"
        ),
        Line2D(
            [0], [0],
            color=METHOD_META["stackelberg"]["color"],
            marker="o",
            linewidth=1.6,
            markersize=5.0,
            label="Proposed Framework"
        ),
    ]

    right_legend = fig.legend(
        handles=right_legend_handles,
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.765, 0.985),
        fancybox=True,
        framealpha=0.96,
        borderpad=0.50,
        columnspacing=1.40,
        handletextpad=0.50,
    )
    right_legend.get_frame().set_facecolor("#ffffff")
    right_legend.get_frame().set_edgecolor("#d5dbe5")
    right_legend.get_frame().set_linewidth(1.0)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_manifest(dataset, delay_key):
    manifest = {
        "delay_key": delay_key,
        "metrics": {
            "x_axis": "avg_usv_move",
            "y_axis": "mean(avg_tracking_error over AUVs)",
            "bubble_area": "sum_rate",
        },
        "models": {},
    }

    for backbone in ("td3", "dsac"):
        manifest["models"][backbone] = {}
        for method in ("traditional", "stackelberg"):
            manifest["models"][backbone][method] = {}
            for n_auv in (2, 3, 4):
                case = dataset[backbone][method][n_auv]
                manifest["models"][backbone][method][str(n_auv)] = {
                    "run_count": case["run_count"],
                    "mean_avg_usv_move": case["mean_x"],
                    "std_avg_usv_move": case["std_x"],
                    "mean_tracking_error": case["mean_y"],
                    "std_tracking_error": case["std_y"],
                    "mean_sum_rate": case["mean_throughput"],
                    "std_sum_rate": case["std_throughput"],
                    "ellipse_area_1sigma": case["ellipse_area_1sigma"],
                    "source_files": case["source_files"],
                }
    return manifest


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args.delay_key)

    combined_path = resolve_output_path(args.output_dir, "episode_frontier_with_bars.pdf")
    plot_combined_figure(dataset, combined_path, bar_backbone="td3")

    manifest = build_manifest(dataset, args.delay_key)
    manifest_path = args.output_dir / "episode_frontier_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Generated episode-level frontier figures:")
    print(f"  - {combined_path}")
    print(f"  - {manifest_path}")


if __name__ == "__main__":
    main()