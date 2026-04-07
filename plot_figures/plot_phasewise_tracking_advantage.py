import argparse
import json
from pathlib import Path

import numpy as np

try:
    import ijson
except ModuleNotFoundError:
    ijson = None


RESULT_DIR = Path(__file__).resolve().parent.parent / "delay_comparison_results"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "td3_figure_exports"
PHASE_COUNT = 5

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
    "traditional": {"label": "Traditional FIM", "color": "#1f77b4"},
    "stackelberg": {"label": "Proposed Stackelberg", "color": "#d62728"},
}

BACKBONE_META = {
    "td3": {"label": "TD3", "color": "#3b6eea"},
    "dsac": {"label": "DSAC-T", "color": "#d97a12"},
}

AUV_META = {
    2: {"label": "2 AUVs", "short": "2A", "marker": "o", "color": "#2878B5"},
    3: {"label": "3 AUVs", "short": "3A", "marker": "s", "color": "#7FB77E"},
    4: {"label": "4 AUVs", "short": "4A", "marker": "^", "color": "#E6B422"},
}

TITLE_SIZE = 18
LABEL_SIZE = 18
TICK_SIZE = 18
LEGEND_SIZE = 16
ANNOTATION_SIZE = 16
SUBCAPTION_SIZE = 20
COORD_TEXT_SIZE = 12
BAR_VALUE_SIZE = 11


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate phase-aware tracking-improvement figures from delayed comparison results."
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
        help="Delay key to analyze.",
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
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    return plt, pe, Line2D


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


def summarize_case(json_paths, delay_key):
    summary = {
        method: {
            "phase_means_per_run": [],
            "overall_errors": [],
            "packet_loss_rates": [],
            "run_count": 0,
            "source_files": [str(path.with_suffix(".pkl")) for path in json_paths],
        }
        for method in ("traditional", "stackelberg")
    }

    for json_path in json_paths:
        for method in ("traditional", "stackelberg"):
            for result in stream_result_items(json_path, delay_key, method):
                tracking = np.asarray(result["tracking_error"], dtype=float)
                mean_error_series = tracking.mean(axis=0)
                phase_means = [
                    float(np.mean(mean_error_series[np.arange(mean_error_series.size) % PHASE_COUNT == phase]))
                    for phase in range(PHASE_COUNT)
                ]
                summary[method]["phase_means_per_run"].append(phase_means)
                summary[method]["overall_errors"].append(float(np.mean(mean_error_series)))
                summary[method]["packet_loss_rates"].append(float(result["avg_packet_loss_rate"]) * 100.0)
                summary[method]["run_count"] += 1

    return summary


def build_stats(delay_key):
    stats = {}
    for backbone, case_dirs in MODEL_DIRS.items():
        stats[backbone] = {}
        for n_auv, dir_names in case_dirs.items():
            json_paths = []
            for dir_name in dir_names:
                result_dir = RESULT_DIR / dir_name
                json_files = sorted(result_dir.glob("delay_comparison_*.json"))
                if len(json_files) != 1:
                    raise FileNotFoundError(f"Expected one json file in {result_dir}, found {len(json_files)}.")
                json_paths.append(json_files[0])

            summary = summarize_case(json_paths, delay_key)
            trad_phase = np.mean(np.asarray(summary["traditional"]["phase_means_per_run"], dtype=float), axis=0)
            prop_phase = np.mean(np.asarray(summary["stackelberg"]["phase_means_per_run"], dtype=float), axis=0)
            phase_reduction = (trad_phase - prop_phase) / trad_phase * 100.0

            trad_overall = float(np.mean(summary["traditional"]["overall_errors"]))
            prop_overall = float(np.mean(summary["stackelberg"]["overall_errors"]))
            overall_reduction = (trad_overall - prop_overall) / trad_overall * 100.0

            trad_loss = float(np.mean(summary["traditional"]["packet_loss_rates"]))
            prop_loss = float(np.mean(summary["stackelberg"]["packet_loss_rates"]))
            additional_loss = prop_loss - trad_loss

            stats[backbone][n_auv] = {
                "phase_reduction_pct": phase_reduction.tolist(),
                "overall_reduction_pct": overall_reduction,
                "additional_packet_loss_pp": additional_loss,
                "traditional_mean_error": trad_overall,
                "proposed_mean_error": prop_overall,
                "traditional_mean_packet_loss_pp": trad_loss,
                "proposed_mean_packet_loss_pp": prop_loss,
                "run_count": summary["traditional"]["run_count"],
                "source_files": summary["traditional"]["source_files"],
            }
    return stats


def style_axis(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.7, color="#d9dee7", alpha=0.9, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#4a4a4a")
        spine.set_linewidth(0.9)
    ax.tick_params(labelsize=TICK_SIZE, colors="#252525")


def apply_bar_depth_effect(bar, pe):
    bar.set_path_effects(
        [
            pe.withSimplePatchShadow(offset=(1.6, -1.6), shadow_rgbFace=(0.15, 0.18, 0.24), alpha=0.18),
            pe.Normal(),
        ]
    )


def add_subcaption_below(ax, text):
    ax.text(
        0.5,
        -0.24,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=SUBCAPTION_SIZE,
    )


def format_coord(x, y):
    return f"({x:.2f}, {y:.2f})"


def annotate_bar_values(ax, bars, values):
    ymax = ax.get_ylim()[1]
    offset = ymax * 0.015
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=BAR_VALUE_SIZE,
            color="#222222",
            zorder=6,
        )


def plot_reviewer_figure(stats, output_path):
    plt, pe, Line2D = get_plotting_modules()
    fig = plt.figure(figsize=(18.0, 6.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.12, 1.18], wspace=0.20)

    axes = {
        "td3": fig.add_subplot(gs[0, 0]),
        "dsac": fig.add_subplot(gs[0, 1]),
        "tradeoff": fig.add_subplot(gs[0, 2]),
    }

    phase_x = np.arange(PHASE_COUNT, dtype=float)
    bar_width = 0.22
    offsets = {2: -bar_width, 3: 0.0, 4: bar_width}

    for backbone, subcaption in (
        ("td3", "(a) TD3 phase-wise reduction"),
        ("dsac", "(b) DSAC-T phase-wise reduction"),
    ):
        ax = axes[backbone]
        style_axis(ax)
        ax.axvspan(-0.5, 0.5, color="#f2f5fa", alpha=0.85, zorder=0)
        max_y = 0.0

        bars_per_group = []

        for n_auv in (2, 3, 4):
            x_pos = phase_x + offsets[n_auv]
            y_val = np.asarray(stats[backbone][n_auv]["phase_reduction_pct"], dtype=float)
            max_y = max(max_y, float(np.max(y_val)))
            bars = ax.bar(
                x_pos,
                y_val,
                width=bar_width,
                color=AUV_META[n_auv]["color"],
                alpha=0.90,
                edgecolor="#ffffff",
                linewidth=0.9,
                label=AUV_META[n_auv]["label"],
                zorder=3,
            )
            bars_per_group.append((bars, y_val))

            for bar in bars:
                apply_bar_depth_effect(bar, pe)

            line = ax.plot(
                x_pos,
                y_val,
                color=AUV_META[n_auv]["color"],
                linewidth=2.1,
                marker="o",
                markersize=5.2,
                zorder=4,
                alpha=0.95,
            )[0]
            line.set_path_effects([pe.Stroke(linewidth=3.6, foreground="#ffffff", alpha=0.65), pe.Normal()])

        ax.set_xticks(phase_x)
        ax.set_xlabel("Communication phase", fontsize=LABEL_SIZE)
        ax.set_ylabel("Tracking-error reduction (%)", fontsize=LABEL_SIZE)
        ax.set_ylim(0.0, max_y * 1.24)

        for bars, y_val in bars_per_group:
            annotate_bar_values(ax, bars, y_val)

        ax.text(
            0.00,
            0.96,
            "Update phase",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_SIZE,
            color="#5a6678",
        )

        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=3,
            frameon=True,
            fancybox=True,
            framealpha=0.96,
            facecolor="#ffffff",
            edgecolor="#d0d0d0",
            fontsize=LEGEND_SIZE,
            columnspacing=0.55,
            handletextpad=0.28,
            labelspacing=0.25,
            borderpad=0.28,
            handlelength=1.15,
            borderaxespad=0.18,
        )

        add_subcaption_below(ax, subcaption)

    trade_ax = axes["tradeoff"]
    style_axis(trade_ax)
    trade_ax.axvline(0.0, color="#95a2b8", linestyle="--", linewidth=1.2, zorder=1)
    trade_ax.axhline(0.0, color="#95a2b8", linestyle="--", linewidth=1.2, zorder=1)

    point_offsets = {
        ("td3", 2): (-12, -0),
        ("td3", 3): (0, -16),
        ("td3", 4): (-2, -13),
        ("dsac", 2): (0, -12),
        ("dsac", 3): (-1, 12),
        ("dsac", 4): (-10, -0),
    }

    for backbone in ("td3", "dsac"):
        color = BACKBONE_META[backbone]["color"]
        x_vals = [stats[backbone][n]["additional_packet_loss_pp"] for n in (2, 3, 4)]
        y_vals = [stats[backbone][n]["overall_reduction_pct"] for n in (2, 3, 4)]

        trade_line = trade_ax.plot(
            x_vals, y_vals, color=color, linewidth=2.2, alpha=0.94, zorder=3
        )[0]
        trade_line.set_path_effects(
            [pe.Stroke(linewidth=4.1, foreground="#ffffff", alpha=0.7), pe.Normal()]
        )

        for n_auv, x_val, y_val in zip((2, 3, 4), x_vals, y_vals):
            trade_ax.vlines(x_val, 0.0, y_val, color=color, linewidth=1.7, alpha=0.36, zorder=2)
            marker = trade_ax.scatter(
                x_val,
                y_val,
                s=350,
                marker=AUV_META[n_auv]["marker"],
                color=color,
                edgecolors="#1f1f1f",
                linewidths=1.25,
                alpha=0.95,
                zorder=4,
            )
            marker.set_path_effects(
                [
                    pe.withSimplePatchShadow(
                        offset=(1.6, -1.6),
                        shadow_rgbFace=(0.15, 0.18, 0.24),
                        alpha=0.20,
                    ),
                    pe.Normal(),
                ]
            )

            ox, oy = point_offsets[(backbone, n_auv)]
            trade_ax.annotate(
                format_coord(x_val, y_val),
                xy=(x_val, y_val),
                xytext=(ox, oy),
                textcoords="offset points",
                fontsize=COORD_TEXT_SIZE,
                color=color,
                ha="left" if ox >= 0 else "right",
                va="bottom" if oy >= 0 else "top",
                zorder=6,
            )

    model_handles = [
        Line2D(
            [0], [0],
            color=BACKBONE_META[key]["color"],
            linewidth=2.2,
            marker="o",
            markersize=7.5,
            label=BACKBONE_META[key]["label"],
        )
        for key in ("td3", "dsac")
    ]
    auv_handles = [
        Line2D(
            [0], [0],
            color="#111111",
            linewidth=0,
            marker=AUV_META[n]["marker"],
            markerfacecolor="#ffffff",
            markeredgewidth=1.15,
            markersize=8.5,
            label=AUV_META[n]["label"],
        )
        for n in (2, 3, 4)
    ]

    trade_ax.legend(
        handles=model_handles + auv_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=5,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        facecolor="#ffffff",
        edgecolor="#d0d0d0",
        fontsize=LEGEND_SIZE,
        columnspacing=0.20,
        handletextpad=0.15,
        labelspacing=0.10,
        borderpad=0.20,
        handlelength=1.10,
        borderaxespad=0.18,
    )

    trade_ax.set_xlabel("Additional packet loss (%)", fontsize=LABEL_SIZE)
    trade_ax.set_ylabel(
        "Overall tracking-error reduction (%)",
        fontsize=LABEL_SIZE,
        labelpad=10,
    )
    trade_ax.yaxis.set_label_coords(-0.10, 0.46)
    trade_ax.set_xlim(0,13.3)
    trade_ax.set_ylim(0,22)

    add_subcaption_below(trade_ax, "(c) Gain retained despite harsher packet loss")

    fig.subplots_adjust(top=0.80, bottom=0.30)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_manifest(stats, delay_key):
    manifest = {"delay_key": delay_key, "phase_count": PHASE_COUNT, "models": {}}
    for backbone in ("td3", "dsac"):
        manifest["models"][backbone] = {}
        for n_auv in (2, 3, 4):
            manifest["models"][backbone][str(n_auv)] = stats[backbone][n_auv]
    return manifest


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = build_stats(args.delay_key)

    reviewer_path = resolve_output_path(args.output_dir, "phasewise_tracking_reduction_reviewer.pdf")
    plot_reviewer_figure(stats, reviewer_path)

    manifest = build_manifest(stats, args.delay_key)
    manifest_path = args.output_dir / "phasewise_tracking_reduction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Generated phase-wise tracking-improvement figures:")
    print(f"  - {reviewer_path}")
    print(f"  - {manifest_path}")


if __name__ == "__main__":
    main()