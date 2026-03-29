import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


RESULT_DIR = Path(__file__).resolve().parent / "delay_comparison_results"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "td3_figure_exports"

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
    "traditional": {
        "title": "Baseline",
        "curve_label": "Baseline",
        "color": "#1f77b4",
        "panel_prefix": "Baseline",
    },
    "stackelberg": {
        "title": "Proposed Framework",
        "curve_label": "Proposed Framework",
        "color": "#d62728",
        "panel_prefix": "Proposed Framework",
    },
}

HEATMAP_BINS = 20
HEATMAP_SMOOTH_SIGMA = 1.1
SUBTITLE_SIZE = 18
LABEL_SIZE = 18
TICK_SIZE = 18
LEGEND_SIZE = 18
ANNOTATION_SIZE = 18
BAR_VALUE_SIZE = 18


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TD3 USV occupancy heatmaps and H/R summary curves."
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
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    return plt, cm, mcolors


def find_matching_jsons():
    grouped = {n_auv: [] for n_auv in TD3_RESULT_DIRS}
    for n_auv, dir_names in TD3_RESULT_DIRS.items():
        for dir_name in dir_names:
            result_dir = RESULT_DIR / dir_name
            json_files = sorted(result_dir.glob("delay_comparison_*.json"))
            if len(json_files) != 1:
                raise FileNotFoundError(
                    f"Expected one json file in {result_dir}, found {len(json_files)}."
                )
            grouped[n_auv].append(json_files[0])
    return grouped


def load_json_file(json_path):
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stream_result_items(json_path, delay_key, method):
    data = load_json_file(json_path)
    return data["results"][delay_key][method]["results"]


def load_experiment_info(json_path):
    data = load_json_file(json_path)
    return data["experiment_info"]


def compute_heatmap_and_metrics(x_usv, y_usv, border_x, border_y, bins):
    counts, _, _ = np.histogram2d(
        np.asarray(x_usv, dtype=float),
        np.asarray(y_usv, dtype=float),
        bins=bins,
        range=[[0.0, border_x], [0.0, border_y]],
    )

    flat = counts.ravel().astype(float)
    total = np.sum(flat)
    if total <= 0.0:
        return counts, 0.0, 0.0

    probabilities = flat[flat > 0.0] / total
    entropy = float(-(probabilities * np.log(probabilities)).sum())

    centers_x = (np.arange(bins, dtype=float) + 0.5) * (border_x / bins)
    centers_y = (np.arange(bins, dtype=float) + 0.5) * (border_y / bins)
    xx, yy = np.meshgrid(centers_x, centers_y, indexing="xy")
    weights = counts.T.astype(float)
    total_weight = np.sum(weights)
    centroid_x = float(np.sum(xx * weights) / total_weight)
    centroid_y = float(np.sum(yy * weights) / total_weight)
    radial_distance = np.sqrt((xx - centroid_x) ** 2 + (yy - centroid_y) ** 2)
    radial_spread = float(np.sum(radial_distance * weights) / total_weight)
    return counts, entropy, radial_spread


def aggregate_case_method(json_paths, delay_key, method):
    experiment_info = load_experiment_info(json_paths[0])
    border_x = float(experiment_info["border_x"])
    border_y = float(experiment_info["border_y"])

    aggregate_counts = np.zeros((HEATMAP_BINS, HEATMAP_BINS), dtype=np.int64)
    entropies = []
    radial_spreads = []
    total_runs = 0

    for json_path in json_paths:
        results = stream_result_items(json_path, delay_key, method)
        for result in results:
            counts, entropy, radial_spread = compute_heatmap_and_metrics(
                result["x_usv"],
                result["y_usv"],
                border_x,
                border_y,
                HEATMAP_BINS,
            )
            aggregate_counts += counts.astype(np.int64)
            entropies.append(entropy)
            radial_spreads.append(radial_spread)
            total_runs += 1

    entropies = np.asarray(entropies, dtype=float)
    radial_spreads = np.asarray(radial_spreads, dtype=float)

    return {
        "counts": aggregate_counts,
        "mean_entropy": float(np.mean(entropies)) if entropies.size else 0.0,
        "std_entropy": float(np.std(entropies, ddof=0)) if entropies.size else 0.0,
        "var_entropy": float(np.var(entropies, ddof=0)) if entropies.size else 0.0,
        "mean_radial_spread": float(np.mean(radial_spreads)) if radial_spreads.size else 0.0,
        "std_radial_spread": float(np.std(radial_spreads, ddof=0)) if radial_spreads.size else 0.0,
        "var_radial_spread": float(np.var(radial_spreads, ddof=0)) if radial_spreads.size else 0.0,
        "border_x": border_x,
        "border_y": border_y,
        "total_runs": total_runs,
        "source_files": [str(path.with_suffix(".pkl")) for path in json_paths],
    }


def build_display_heatmap(counts):
    smoothed = gaussian_filter(
        counts.T.astype(float),
        sigma=HEATMAP_SMOOTH_SIGMA,
        mode="constant",
        cval=0.0,
    )
    return np.maximum(smoothed, 1e-3)


def build_occupancy_dataset(delay_key):
    grouped_files = find_matching_jsons()
    dataset = {}
    for n_auv, json_paths in grouped_files.items():
        dataset[n_auv] = {}
        for method in ("traditional", "stackelberg"):
            dataset[n_auv][method] = aggregate_case_method(json_paths, delay_key, method)
    return dataset


def style_heatmap_axis(ax, border_x, border_y):
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle=(0, (2, 2)), linewidth=0.45, color="#d7dce4", alpha=0.55)
    ax.set_xlim(0.0, border_x)
    ax.set_ylim(0.0, border_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X-axis (m)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Y-axis (m)", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)


def style_curve_axis(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.75, color="#d9dee7", alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color("#404040")
        spine.set_linewidth(0.9)
    ax.tick_params(labelsize=TICK_SIZE, colors="#252525")


def plot_occupancy_heatmap_figure(dataset, output_path):
    plt, cm, mcolors = get_plotting_modules()

    fig = plt.figure(figsize=(18.0, 11.0))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=7,
        width_ratios=[1, 1, 1, 1, 1, 1, 0.18],
        height_ratios=[1.0, 1.0],
        hspace=0.30,
        wspace=0.05,
        left=0.055,
        right=0.945,
        top=0.97,
        bottom=0.08,
    )

    all_counts = []
    for n_auv in sorted(dataset):
        for method in ("traditional", "stackelberg"):
            all_counts.append(build_display_heatmap(dataset[n_auv][method]["counts"]))
    vmax = max(float(np.max(counts)) for counts in all_counts)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#ffffff")
    norm = mcolors.LogNorm(vmin=1, vmax=max(vmax, 1))

    heatmap_artist = None
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    panel_idx = 0

    for row_idx, method in enumerate(("traditional", "stackelberg")):
        for col_idx, n_auv in enumerate(sorted(dataset)):
            ax = fig.add_subplot(gs[row_idx, col_idx * 2 : (col_idx + 1) * 2])
            case_data = dataset[n_auv][method]
            display_counts = build_display_heatmap(case_data["counts"])

            heatmap_artist = ax.imshow(
                display_counts,
                origin="lower",
                extent=[0.0, case_data["border_x"], 0.0, case_data["border_y"]],
                cmap=cmap,
                norm=norm,
                interpolation="bicubic",
                aspect="equal",
            )

            style_heatmap_axis(ax, case_data["border_x"], case_data["border_y"])

            subcaption = (
                f"{panel_labels[panel_idx]} {METHOD_META[method]['panel_prefix']} ({n_auv} AUVs)"
            )
            panel_idx += 1
            ax.text(
                0.5,
                -0.18,
                subcaption,
                transform=ax.transAxes,
                fontsize=22,
                ha="center",
                va="top",
            )

    cax = fig.add_subplot(gs[:, 6])
    cbar = fig.colorbar(heatmap_artist, cax=cax)
    cbar.set_label("USV occupancy intensity", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def add_bar_value_labels(ax, x_positions, values, yerrs, fmt, fontsize):
    current_ylim = ax.get_ylim()[1]
    for x, y, err in zip(x_positions, values, yerrs):
        y_offset = max(current_ylim * 0.012, err * 0.25)
        ax.text(
            x,
            y + err + y_offset,
            format(y, fmt),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#252525",
        )


def plot_hr_summary_figure(dataset, output_path):
    plt, cm, mcolors = get_plotting_modules()

    fig, (h_ax, r_ax) = plt.subplots(1, 2, figsize=(18.0, 4.0))
    fig.subplots_adjust(
        left=0.075,
        right=0.98,
        top=0.71,
        bottom=0.05,
        wspace=0.10,
    )

    auv_counts = np.array(sorted(dataset), dtype=int)
    x_tick_labels = [f"{n} AUVs" for n in auv_counts]

    for ax in (h_ax, r_ax):
        style_curve_axis(ax)
        ax.set_xticks(auv_counts)
        ax.set_xticklabels(x_tick_labels)
        ax.set_xlim(auv_counts[0] - 0.4, auv_counts[-1] + 0.4)
        ax.margins(x=0.0)

    bar_width = 0.25
    offsets = {
        "traditional": -bar_width / 2.0,
        "stackelberg": bar_width / 2.0,
    }

    h_max_upper = 0.0
    r_max_upper = 0.0
    plot_cache = []

    for method in ("traditional", "stackelberg"):
        color = METHOD_META[method]["color"]
        h_values = np.array([dataset[n_auv][method]["mean_entropy"] for n_auv in auv_counts], dtype=float)
        h_stds = np.array([dataset[n_auv][method]["std_entropy"] for n_auv in auv_counts], dtype=float)

        r_values = np.array([dataset[n_auv][method]["mean_radial_spread"] for n_auv in auv_counts], dtype=float)
        r_stds = np.array([dataset[n_auv][method]["std_radial_spread"] for n_auv in auv_counts], dtype=float)

        x_positions = auv_counts + offsets[method]

        h_ax.bar(
            x_positions,
            h_values,
            width=bar_width,
            color=color,
            alpha=0.24,
            edgecolor=color,
            linewidth=1.1,
            yerr=h_stds,
            ecolor=color,
            capsize=4,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
            zorder=2,
        )
        h_ax.plot(
            x_positions,
            h_values,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=6.5,
            label=METHOD_META[method]["curve_label"],
            zorder=3,
        )

        r_ax.bar(
            x_positions,
            r_values,
            width=bar_width,
            color=color,
            alpha=0.24,
            edgecolor=color,
            linewidth=1.1,
            yerr=r_stds,
            ecolor=color,
            capsize=4,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
            zorder=2,
        )
        r_ax.plot(
            x_positions,
            r_values,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=6.5,
            label=METHOD_META[method]["curve_label"],
            zorder=3,
        )

        h_max_upper = max(h_max_upper, float(np.max(h_values + h_stds)))
        r_max_upper = max(r_max_upper, float(np.max(r_values + r_stds)))

        plot_cache.append((x_positions, h_values, h_stds, r_values, r_stds))

    h_ax.set_ylim(0, h_max_upper * 1.18 if h_max_upper > 0 else 1.0)
    r_ax.set_ylim(0, r_max_upper * 1.18 if r_max_upper > 0 else 1.0)

    for x_positions, h_values, h_stds, r_values, r_stds in plot_cache:
        add_bar_value_labels(h_ax, x_positions, h_values, h_stds, ".2f", BAR_VALUE_SIZE)
        add_bar_value_labels(r_ax, x_positions, r_values, r_stds, ".1f", BAR_VALUE_SIZE)

    h_ax.set_xlabel("Number of AUVs", fontsize=LABEL_SIZE)
    h_ax.set_ylabel(r"Mean USV occupancy entropy", fontsize=LABEL_SIZE)

    r_ax.set_xlabel("Number of AUVs", fontsize=LABEL_SIZE)
    r_ax.set_ylabel(r"Mean USV radial spread (m)", fontsize=LABEL_SIZE)

    handles, labels = h_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="#ffffff",
        edgecolor="#d0d0d0",
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, 0.88),
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_manifest(dataset, delay_key):
    manifest = {
        "delay_key": delay_key,
        "heatmap_bins": HEATMAP_BINS,
        "display_heatmap_smoothing_sigma": HEATMAP_SMOOTH_SIGMA,
        "cases": {},
    }
    for n_auv in sorted(dataset):
        manifest["cases"][str(n_auv)] = {}
        for method in ("traditional", "stackelberg"):
            case_data = dataset[n_auv][method]
            manifest["cases"][str(n_auv)][method] = {
                "mean_entropy": case_data["mean_entropy"],
                "std_entropy": case_data["std_entropy"],
                "var_entropy": case_data["var_entropy"],
                "mean_radial_spread": case_data["mean_radial_spread"],
                "std_radial_spread": case_data["std_radial_spread"],
                "var_radial_spread": case_data["var_radial_spread"],
                "total_runs": case_data["total_runs"],
                "source_files": case_data["source_files"],
                "max_occupancy_count": int(np.max(case_data["counts"])),
            }
    return manifest


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_occupancy_dataset(args.delay_key)

    heatmap_figure_path = resolve_output_path(
        args.output_dir,
        "td3_usv_occupancy_heatmap_2_3_4auv.pdf",
    )
    hr_summary_figure_path = resolve_output_path(
        args.output_dir,
        "td3_usv_occupancy_hr_summary_2_3_4auv.pdf",
    )

    plot_occupancy_heatmap_figure(dataset, heatmap_figure_path)
    plot_hr_summary_figure(dataset, hr_summary_figure_path)

    manifest = build_manifest(dataset, args.delay_key)
    manifest_path = args.output_dir / "td3_usv_occupancy_heatmap_hr_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Generated occupancy figures:")
    print(f"  - {heatmap_figure_path}")
    print(f"  - {hr_summary_figure_path}")
    print(f"  - {manifest_path}")


if __name__ == "__main__":
    main()