import argparse
import json
from pathlib import Path

import ijson
import numpy as np


RESULT_DIR = Path(__file__).resolve().parent.parent / "delay_comparison_results"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "td3_figure_exports"

TD3_SELECTION = {
    2: {"load_ep": 575},
    3: {"load_ep": 200},
    4: {"load_ep": 400},
}

TD3_RESULT_DIRS = {
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
}

METHOD_META = {
    "traditional": {"label": "Baseline", "color": "#1f77b4"},
    "stackelberg": {"label": "Proposed Framework", "color": "#d62728"},
}

AUV_COLORS = ["#B04DDB", "#F27D72", "#4F83CC", "#F2B84B", "#72B7B2", "#B279A2"]
USV_COLOR = "#7BEF7D"
SN_PRIORITY_COLORS = ["#f08c2b", "#f26d7d", "#53d3d1", "#7a3cff", "#ffcf4d", "#2e86ab"]
OUTPUT_BASENAMES = (
    "td3_trajectory_2_3_4auv",
    "td3_tracking_error_2_3_4auv",
    "td3_fim_detj_2_3_4auv",
)
TRACKING_TRIM_STEPS = 0
TRACKING_MEAN_WINDOW = 1
TRACKING_STD_WINDOW = 1

SUPTITLE_SIZE = 21
SUBTITLE_SIZE = 17
LABEL_SIZE = 15.5
TICK_SIZE = 14
LEGEND_SIZE = 13.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TD3 figures for 2/3/4 AUV settings from saved comparison results."
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
        help="Delay result key to visualize.",
    )
    return parser.parse_args()


def get_plotting_modules():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "Times New Roman"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    return plt, Line2D


def find_matching_jsons():
    grouped = {n_auv: [] for n_auv in TD3_RESULT_DIRS}
    for n_auv, dir_names in TD3_RESULT_DIRS.items():
        for dir_name in dir_names:
            result_dir = RESULT_DIR / dir_name
            json_files = sorted(result_dir.glob("delay_comparison_*.json"))
            if len(json_files) != 1:
                raise FileNotFoundError(f"Expected one json file in {result_dir}, found {len(json_files)}.")
            grouped[n_auv].append(json_files[0])
    return grouped


def compute_percentile_ranks(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float)
    if values.size == 1:
        return np.asarray([1.0], dtype=float)

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float) / (values.size - 1)
    return ranks


def smooth_series(series, window=6):
    if series.size == 0 or window <= 1:
        return series
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(series, kernel, mode="same")


def align_series_stack(series_list):
    if not series_list:
        return None
    min_len = min(series.shape[0] for series in series_list)
    if min_len == 0:
        return None
    return np.vstack([series[:min_len] for series in series_list])


def find_representative_index(values):
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    mean_value = float(np.mean(arr))
    idx = int(np.argmin(np.abs(arr - mean_value)))
    return idx, float(arr[idx])


def compute_trajectory_bbox_metrics(result):
    x_values = [np.asarray(track, dtype=float) for track in result["x_auv"]]
    y_values = [np.asarray(track, dtype=float) for track in result["y_auv"]]
    x_values.append(np.asarray(result["x_usv"], dtype=float))
    y_values.append(np.asarray(result["y_usv"], dtype=float))

    x_all = np.concatenate(x_values)
    y_all = np.concatenate(y_values)

    bbox_w = float(np.max(x_all) - np.min(x_all))
    bbox_h = float(np.max(y_all) - np.min(y_all))
    max_dim = max(bbox_w, bbox_h, 1e-8)
    min_dim = min(bbox_w, bbox_h)
    bbox_area = max(bbox_w * bbox_h, 0.0)

    aspect_balance = min_dim / max_dim
    coverage_score = min(np.sqrt(bbox_area) / 200.0, 1.0)
    return aspect_balance, coverage_score


def compute_trajectory_fill_score(result, bins=7):
    x_tracks = [np.asarray(track, dtype=float) for track in result["x_auv"]]
    y_tracks = [np.asarray(track, dtype=float) for track in result["y_auv"]]
    x_tracks.append(np.asarray(result["x_usv"], dtype=float))
    y_tracks.append(np.asarray(result["y_usv"], dtype=float))

    sampled_points = []
    for x_track, y_track in zip(x_tracks, y_tracks):
        draw_x, draw_y = downsample_track(x_track, y_track, step=6)
        sampled_points.append(np.column_stack([draw_x, draw_y]))

    if not sampled_points:
        return 0.0

    points = np.vstack(sampled_points)
    points = np.clip(points, 0.0, 199.999)
    cell_x = np.floor(points[:, 0] / (200.0 / bins)).astype(int)
    cell_y = np.floor(points[:, 1] / (200.0 / bins)).astype(int)
    occupied = {(int(ix), int(iy)) for ix, iy in zip(cell_x, cell_y)}
    occupancy_ratio = len(occupied) / float(bins * bins)
    return float(min(occupancy_ratio / 0.42, 1.0))


def compute_centeredness_metrics(result):
    x_values = [np.asarray(track, dtype=float) for track in result["x_auv"]]
    y_values = [np.asarray(track, dtype=float) for track in result["y_auv"]]
    x_values.append(np.asarray(result["x_usv"], dtype=float))
    y_values.append(np.asarray(result["y_usv"], dtype=float))

    x_all = np.concatenate(x_values)
    y_all = np.concatenate(y_values)

    center_xy = np.array([100.0, 100.0])
    centroid = np.array([np.mean(x_all), np.mean(y_all)])
    dist_to_center = np.linalg.norm(centroid - center_xy)
    center_score = max(0.0, 1.0 - dist_to_center / (np.sqrt(2.0) * 100.0))

    edge_clearance = np.minimum.reduce([x_all, 200.0 - x_all, y_all, 200.0 - y_all])
    edge_clearance = np.clip(edge_clearance, 0.0, 30.0) / 30.0
    boundary_margin_score = float(np.mean(edge_clearance))

    return float(center_score), boundary_margin_score


def compute_path_smoothness_score(result):
    tracks_x = [np.asarray(track, dtype=float) for track in result["x_auv"]]
    tracks_y = [np.asarray(track, dtype=float) for track in result["y_auv"]]
    tracks_x.append(np.asarray(result["x_usv"], dtype=float))
    tracks_y.append(np.asarray(result["y_usv"], dtype=float))

    smoothness_scores = []
    for x_track, y_track in zip(tracks_x, tracks_y):
        draw_x, draw_y = downsample_track(x_track, y_track, step=8)
        points = np.column_stack([draw_x, draw_y])
        if points.shape[0] < 3:
            smoothness_scores.append(0.5)
            continue

        vectors = np.diff(points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > 1e-8
        vectors = vectors[valid]
        lengths = lengths[valid]
        if vectors.shape[0] < 2:
            smoothness_scores.append(0.5)
            continue

        unit = vectors / lengths[:, None]
        cosine = np.sum(unit[:-1] * unit[1:], axis=1)
        cosine = np.clip(cosine, -1.0, 1.0)
        turn_angles = np.arccos(cosine)
        mean_turn = float(np.mean(turn_angles))
        smoothness_scores.append(max(0.0, 1.0 - mean_turn / np.pi))

    return float(np.mean(smoothness_scores)) if smoothness_scores else 0.0


def init_group_summary(json_paths):
    return {
        "source_files": [str(path.with_suffix(".pkl")) for path in json_paths],
        "traditional": {
            "tracking_series": [],
            "detj_series": [],
            "avg_detJ_values": [],
        },
        "stackelberg": {
            "tracking_series": [],
            "detj_series": [],
            "avg_detJ_values": [],
            "trajectory_candidates": [],
        },
    }


def stream_result_items(json_path, delay_key, method):
    prefix = f"results.{delay_key}.{method}.results.item"
    with json_path.open("rb") as handle:
        for result in ijson.items(handle, prefix):
            yield result


def load_group_summary(json_paths, delay_key):
    experiment_info = None
    summary = init_group_summary(json_paths)

    for json_path in json_paths:
        if experiment_info is None:
            with json_path.open("rb") as handle:
                experiment_info = next(ijson.items(handle, "experiment_info"))

        for method in ("traditional", "stackelberg"):
            for result_idx, result in enumerate(stream_result_items(json_path, delay_key, method)):
                tracking_series = np.mean(np.asarray(result["tracking_error"], dtype=float), axis=0)
                detj_series = np.asarray(result["detJ_values"], dtype=float)
                avg_detj = float(result.get("avg_detJ", 0.0))

                summary[method]["tracking_series"].append(tracking_series.astype(np.float32))
                summary[method]["detj_series"].append(detj_series.astype(np.float32))
                summary[method]["avg_detJ_values"].append(avg_detj)

                if method == "stackelberg":
                    aspect_balance, coverage_score = compute_trajectory_bbox_metrics(result)
                    center_score, boundary_margin_score = compute_centeredness_metrics(result)
                    smoothness_score = compute_path_smoothness_score(result)
                    summary["stackelberg"]["trajectory_candidates"].append(
                        {
                            "global_index": len(summary["stackelberg"]["trajectory_candidates"]),
                            "json_path": str(json_path),
                            "result_idx": result_idx,
                            "avg_detJ": avg_detj,
                            "crash": float(result.get("crash", 0.0)),
                            "aspect_balance": float(aspect_balance),
                            "coverage_score": float(coverage_score),
                            "fill_score": float(compute_trajectory_fill_score(result)),
                            "center_score": float(center_score),
                            "boundary_margin_score": float(boundary_margin_score),
                            "smoothness_score": float(smoothness_score),
                        }
                    )

    return experiment_info, summary


def extract_trajectory_payload(json_path, delay_key, result_idx):
    for current_idx, result in enumerate(stream_result_items(json_path, delay_key, "stackelberg")):
        if current_idx == result_idx:
            return {
                "x_auv": result["x_auv"],
                "y_auv": result["y_auv"],
                "x_usv": result["x_usv"],
                "y_usv": result["y_usv"],
                "SoPcenter": result["SoPcenter"],
                "lda": result["lda"],
            }
    raise IndexError(f"Stackelberg result {result_idx} not found in {json_path}")


def select_aesthetic_stackelberg_candidate(candidates, n_auv):
    if not candidates:
        return None

    detj_ranks = compute_percentile_ranks([candidate["avg_detJ"] for candidate in candidates])
    crash_ranks = compute_percentile_ranks([candidate["crash"] for candidate in candidates])

    best_candidate = None
    best_score = -np.inf

    for idx, candidate in enumerate(candidates):
        target_coverage = 0.82
        weights = {
            "aspect": 0.18,
            "coverage": 0.24,
            "fill": 0.23,
            "center": 0.14,
            "boundary": 0.10,
            "smooth": 0.06,
            "detj": 0.05,
            "crash": 0.05,
        }

        if n_auv == 4:
            target_coverage = 0.92
            weights = {
                "aspect": 0.12,
                "coverage": 0.28,
                "fill": 0.30,
                "center": 0.12,
                "boundary": 0.07,
                "smooth": 0.04,
                "detj": 0.035,
                "crash": 0.035,
            }

        coverage_target_score = max(
            0.0, 1.0 - abs(candidate["coverage_score"] - target_coverage) / target_coverage
        )
        score = (
            weights["aspect"] * candidate["aspect_balance"]
            + weights["coverage"] * coverage_target_score
            + weights["fill"] * candidate["fill_score"]
            + weights["center"] * candidate["center_score"]
            + weights["boundary"] * candidate["boundary_margin_score"]
            + weights["smooth"] * candidate["smoothness_score"]
            + weights["detj"] * float(detj_ranks[idx])
            + weights["crash"] * (1.0 - float(crash_ranks[idx]))
        )
        if score > best_score:
            best_score = score
            best_candidate = dict(candidate)
            best_candidate["score"] = float(score)

    return best_candidate


def style_trajectory_axis(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle=(0, (4, 4)), linewidth=0.85, color="#cfd4db", alpha=0.95)
    for spine in ax.spines.values():
        spine.set_color("#2f2f2f")
        spine.set_linewidth(1.0)
    ax.tick_params(colors="#202020", labelsize=TICK_SIZE)
    ax.set_xlim(10, 195)
    ax.set_ylim(5, 190)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Y (m)", fontsize=LABEL_SIZE)


def build_sn_priority_colors(priorities):
    unique_priorities = sorted({int(value) for value in priorities})
    color_map = {}
    for idx, priority in enumerate(unique_priorities):
        color_map[priority] = SN_PRIORITY_COLORS[idx % len(SN_PRIORITY_COLORS)]
    return color_map


def downsample_track(x_track, y_track, step=6):
    if len(x_track) <= step + 1:
        return x_track, y_track
    keep = np.arange(0, len(x_track), step, dtype=int)
    if keep[-1] != len(x_track) - 1:
        keep = np.append(keep, len(x_track) - 1)
    return x_track[keep], y_track[keep]


def build_delay_label(delay_key):
    if delay_key == "delay_1.0":
        return r"$Delay = T_{trans} + T_{fixed} + T_{sample}$ (with packet loss)"
    return r"$Delay = T_{trans} + T_{fixed} + T_{sample}$"


def plot_trajectory_panel(grouped_data, output_path, delay_key):
    plt, Line2D = get_plotting_modules()
    fig, axes = plt.subplots(1, 3, figsize=(18.3, 6.9), sharex=True, sharey=False)
    selection_info = {}
    priority_color_map = None

    for ax, n_auv in zip(axes, sorted(grouped_data)):
        candidate = select_aesthetic_stackelberg_candidate(
            grouped_data[n_auv]["stackelberg"]["trajectory_candidates"],
            n_auv,
        )
        if candidate is None:
            ax.set_axis_off()
            continue

        trajectory_payload = extract_trajectory_payload(
            Path(candidate["json_path"]),
            delay_key,
            candidate["result_idx"],
        )
        selection_info[n_auv] = {
            "index": int(candidate["global_index"]),
            "score": float(candidate["score"]),
        }

        x_auv = trajectory_payload["x_auv"]
        y_auv = trajectory_payload["y_auv"]
        x_usv = np.asarray(trajectory_payload["x_usv"], dtype=float)
        y_usv = np.asarray(trajectory_payload["y_usv"], dtype=float)
        sensor_nodes = np.asarray(trajectory_payload["SoPcenter"], dtype=float)
        priorities = np.asarray(trajectory_payload["lda"], dtype=float)

        if sensor_nodes.size > 0:
            if priority_color_map is None:
                priority_color_map = build_sn_priority_colors(priorities)
            sn_colors = [priority_color_map[int(value)] for value in priorities]
            ax.scatter(
                sensor_nodes[:, 0],
                sensor_nodes[:, 1],
                s=120,
                c=sn_colors,
                marker="P",
                alpha=0.95,
                edgecolors="#1f1f1f",
                linewidths=0.55,
                zorder=3,
            )

        for idx in range(n_auv):
            color = AUV_COLORS[idx % len(AUV_COLORS)]
            x_track = np.asarray(x_auv[idx], dtype=float)
            y_track = np.asarray(y_auv[idx], dtype=float)
            draw_x, draw_y = downsample_track(x_track, y_track, step=6)

            ax.plot(
                draw_x,
                draw_y,
                color=color,
                linewidth=2.25,
                alpha=0.96,
                linestyle=(0, (4, 2)),
                dash_capstyle="butt",
                zorder=4,
            )
            ax.scatter(
                x_track[0],
                y_track[0],
                marker="d",
                s=96,
                facecolors=color,
                edgecolors="#121212",
                linewidths=1.0,
                zorder=5,
            )
            ax.scatter(
                x_track[-1],
                y_track[-1],
                marker="s",
                s=96,
                facecolors=color,
                edgecolors="#121212",
                linewidths=1.0,
                zorder=5,
            )

        draw_x_usv, draw_y_usv = downsample_track(x_usv, y_usv, step=6)
        ax.plot(
            draw_x_usv,
            draw_y_usv,
            color=USV_COLOR,
            linestyle=(0, (5, 2)),
            linewidth=2.4,
            alpha=0.96,
            dash_capstyle="butt",
            zorder=4,
        )
        ax.scatter(
            x_usv[0],
            y_usv[0],
            marker="d",
            s=96,
            facecolors=USV_COLOR,
            edgecolors="#121212",
            linewidths=1.0,
            zorder=5,
        )
        ax.scatter(
            x_usv[-1],
            y_usv[-1],
            marker="s",
            s=96,
            facecolors=USV_COLOR,
            edgecolors="#121212",
            linewidths=1.0,
            zorder=5,
        )

        style_trajectory_axis(ax)
        ax.tick_params(labelleft=True)
        ax.set_title(f"{n_auv} AUV - Stackelberg", fontsize=SUBTITLE_SIZE, color="#151515", pad=8)

    legend_handles = []
    if priority_color_map:
        for priority, color in priority_color_map.items():
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="P",
                    linestyle="None",
                    markersize=9,
                    markerfacecolor=color,
                    markeredgecolor="#1f1f1f",
                    markeredgewidth=0.7,
                    label=f"SN priority {priority}",
                )
            )
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=AUV_COLORS[idx],
                linewidth=2.25,
                linestyle=(0, (4, 2)),
                label=f"AUV {idx + 1}",
            )
            for idx in range(4)
        ]
    )
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=USV_COLOR,
                linewidth=2.4,
                linestyle=(0, (5, 2)),
                label="USV",
            ),
            Line2D(
                [0],
                [0],
                marker="d",
                linestyle="None",
                markersize=8,
                markerfacecolor="#ffffff",
                markeredgecolor="#121212",
                markeredgewidth=1.1,
                label="Start",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                markersize=8,
                markerfacecolor="#ffffff",
                markeredgecolor="#121212",
                markeredgewidth=1.1,
                label="End",
            ),
        ]
    )

    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 1.035),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="#ffffff",
        edgecolor="#d0d0d0",
        fontsize=LEGEND_SIZE,
        columnspacing=1.0,
        handlelength=2.2,
    )
    fig.suptitle(
        f"Trajectory Comparison - Stackelberg - {build_delay_label(delay_key)}",
        y=1.09,
        fontsize=SUPTITLE_SIZE,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return selection_info


def plot_tracking_error_panel(grouped_data, output_path):
    plt, Line2D = get_plotting_modules()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.5), sharex=True)

    for ax, n_auv in zip(axes, sorted(grouped_data)):
        ax.set_facecolor("#ffffff")
        ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.75, color="#d9dee7", alpha=0.85)
        for spine in ax.spines.values():
            spine.set_color("#3a3a3a")
            spine.set_linewidth(0.9)
        panel_values = []
        right_limit = None
        for method in ("traditional", "stackelberg"):
            stack = align_series_stack(grouped_data[n_auv][method]["tracking_series"])
            if stack is None:
                continue
            if stack.shape[1] > TRACKING_TRIM_STEPS + 5:
                stack = stack[:, TRACKING_TRIM_STEPS:]

            mean = smooth_series(stack.mean(axis=0), window=TRACKING_MEAN_WINDOW)
            std = smooth_series(stack.std(axis=0), window=TRACKING_STD_WINDOW)
            x_axis = np.arange(TRACKING_TRIM_STEPS, TRACKING_TRIM_STEPS + mean.size)
            color = METHOD_META[method]["color"]
            lower = np.maximum(mean - std, 0.0)
            upper = mean + std
            panel_values.append(lower)
            panel_values.append(mean)
            panel_values.append(upper)
            right_limit = int(x_axis[-1])

            ax.plot(
                x_axis,
                mean,
                color=color,
                linewidth=2.1,
                label=METHOD_META[method]["label"],
                solid_capstyle="round",
                solid_joinstyle="round",
                antialiased=True,
            )
            ax.fill_between(
                x_axis,
                lower,
                upper,
                color=color,
                alpha=0.14,
            )

        ax.set_title(f"{n_auv} AUV", fontsize=SUBTITLE_SIZE, color="#1b1b1b", pad=7)
        ax.set_xlabel("Time Step", fontsize=LABEL_SIZE)
        if right_limit is not None:
            ax.set_xlim(left=0, right=right_limit)
        ax.margins(x=0)
        if panel_values:
            pooled = np.concatenate(panel_values)
            upper_q = float(np.quantile(pooled, 0.99))
            max_value = float(np.max(pooled))
            top = max(upper_q, max_value * 0.96)
            ax.set_ylim(bottom=0.0, top=top * 1.06)
        ax.tick_params(labelsize=TICK_SIZE, colors="#262626", width=0.8, length=4)

    axes[0].set_ylabel("Average Tracking Error (m)", fontsize=LABEL_SIZE)
    legend_handles = [
        Line2D([0], [0], color=METHOD_META[method]["color"], linewidth=2.1, label=METHOD_META[method]["label"])
        for method in ("traditional", "stackelberg")
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.995),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="#ffffff",
        edgecolor="#d0d0d0",
        fontsize=LEGEND_SIZE,
        handlelength=2.5,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_fim_panel(grouped_data, output_path):
    plt, Line2D = get_plotting_modules()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharex=True)

    for ax, n_auv in zip(axes, sorted(grouped_data)):
        right_limit = None
        for method in ("traditional", "stackelberg"):
            stack = align_series_stack(grouped_data[n_auv][method]["detj_series"])
            if stack is None:
                continue

            mean = smooth_series(stack.mean(axis=0), window=8)
            std = smooth_series(stack.std(axis=0), window=8)
            x_axis = np.arange(mean.size)
            color = METHOD_META[method]["color"]
            right_limit = int(x_axis[-1])

            ax.plot(x_axis, mean, color=color, linewidth=2.0, label=METHOD_META[method]["label"])
            ax.fill_between(
                x_axis,
                np.maximum(mean - std, 0.0),
                mean + std,
                color=color,
                alpha=0.18,
            )

        ax.set_title(f"{n_auv} AUV", fontsize=SUBTITLE_SIZE)
        ax.set_xlabel("Time Step", fontsize=LABEL_SIZE)
        ax.grid(True, linestyle="--", alpha=0.3)
        if right_limit is not None:
            ax.set_xlim(left=0, right=right_limit)
        ax.margins(x=0)
        ax.set_ylim(bottom=0.0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.tick_params(labelsize=TICK_SIZE)

    axes[0].set_ylabel("FIM Value det(J)", fontsize=LABEL_SIZE)
    legend_handles = [
        Line2D([0], [0], color=METHOD_META[method]["color"], linewidth=2.0, label=METHOD_META[method]["label"])
        for method in ("traditional", "stackelberg")
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.995),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="#ffffff",
        edgecolor="#d0d0d0",
        fontsize=LEGEND_SIZE,
        handlelength=2.5,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_manifest(grouped_data, trajectory_selection):
    manifest = {}
    for n_auv in sorted(grouped_data):
        manifest[str(n_auv)] = {
            "load_ep": TD3_SELECTION[n_auv]["load_ep"],
            "source_files": grouped_data[n_auv]["source_files"],
            "traditional_runs": len(grouped_data[n_auv]["traditional"]["avg_detJ_values"]),
            "stackelberg_runs": len(grouped_data[n_auv]["stackelberg"]["avg_detJ_values"]),
            "trajectory_stackelberg_selected_index": trajectory_selection.get(n_auv, {}).get("index"),
            "trajectory_stackelberg_aesthetic_score": trajectory_selection.get(n_auv, {}).get("score"),
        }

        for method in ("traditional", "stackelberg"):
            rep_idx, rep_value = find_representative_index(grouped_data[n_auv][method]["avg_detJ_values"])
            manifest[str(n_auv)][f"{method}_representative_index"] = rep_idx
            manifest[str(n_auv)][f"{method}_representative_avg_detJ"] = rep_value

    return manifest


def cleanup_existing_outputs(output_dir):
    for basename in OUTPUT_BASENAMES:
        for extension in (".png", ".pdf"):
            path = output_dir / f"{basename}{extension}"
            if path.exists():
                try:
                    path.unlink()
                except PermissionError:
                    pass


def resolve_output_path(output_dir, filename):
    path = output_dir / filename
    if not path.exists():
        return path

    try:
        with path.open("ab"):
            return path
    except OSError:
        stem = path.stem
        suffix = path.suffix
        return output_dir / f"{stem}_updated{suffix}"


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_existing_outputs(args.output_dir)

    grouped_files = find_matching_jsons()
    grouped_data = {}

    for n_auv, json_paths in grouped_files.items():
        _, summary = load_group_summary(json_paths, args.delay_key)
        grouped_data[n_auv] = summary

    trajectory_output = resolve_output_path(args.output_dir, "td3_trajectory_2_3_4auv.pdf")
    tracking_output = resolve_output_path(args.output_dir, "td3_tracking_error_2_3_4auv.pdf")
    fim_output = resolve_output_path(args.output_dir, "td3_fim_detj_2_3_4auv.pdf")

    trajectory_selection = plot_trajectory_panel(
        grouped_data, trajectory_output, args.delay_key
    )
    plot_tracking_error_panel(grouped_data, tracking_output)
    plot_fim_panel(grouped_data, fim_output)

    manifest = build_manifest(grouped_data, trajectory_selection)
    manifest_path = args.output_dir / "td3_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Generated figures:")
    print(f"  - {trajectory_output}")
    print(f"  - {tracking_output}")
    print(f"  - {fim_output}")
    print(f"  - {manifest_path}")


if __name__ == "__main__":
    main()
