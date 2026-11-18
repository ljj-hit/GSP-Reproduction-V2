"""Generate V2 figures from existing CSV/JSON artifacts only.

Training: exactly four curves (loss, validation success, validation weighted
path-cost ratio, validation transition evaluations).

Evaluation: one comparison chart for each fixed evaluation metric.  No search
or training is run here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from metrics import PREFERRED_ALGORITHM_ORDER, algorithm_sort_key, read_evaluation_csv
from utils import ensure_dir, setup_logger


# Validated light-mode palette; fixed per algorithm.
SERIES_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
)
ALGORITHM_COLORS = {
    "Dijkstra": SERIES_COLORS[0],
    "A* + Manhattan": SERIES_COLORS[1],
    "WA* + Manhattan": SERIES_COLORS[2],
    "Untrained R-GNN + WA*": SERIES_COLORS[3],
    "Trained GSP-RGNN + WA*": SERIES_COLORS[4],
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 9.5,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
        }
    )


def _style_axes(ax: plt.Axes, *, xgrid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.7)
    if xgrid:
        ax.grid(axis="y", visible=False)
    else:
        ax.grid(axis="x", visible=False)


def _save(fig: plt.Figure, path: Path, logger: Any) -> Path:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any) -> Optional[int]:
    result = _float(value)
    return None if result is None else int(result)


# ---------------------------------------------------------------------------
# Training curves (exactly four)
# ---------------------------------------------------------------------------
def plot_training_loss(training_log: Path, figures_dir: Path, logger: Any) -> Optional[Path]:
    if not training_log.is_file():
        logger.warning("Missing %s", training_log)
        return None
    rows = _read_csv(training_log)
    points = [
        (_int(row.get("episode")), _float(row.get("loss")))
        for row in rows
    ]
    points = [(episode, loss) for episode, loss in points if episode is not None and loss is not None]
    if not points:
        logger.warning("No numeric loss entries in %s", training_log)
        return None
    x = np.asarray([episode for episode, _ in points], dtype=float)
    y = np.asarray([loss for _, loss in points], dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    ax.plot(x, y, color=SERIES_COLORS[0], linewidth=1.6, label="Training loss")
    _style_axes(ax)
    ax.set_title("Training Loss")
    ax.set_xlabel("Episode")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    return _save(fig, figures_dir / "training_loss.png", logger)


def _validation_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for row in _read_csv(path):
        episode = _int(row.get("episode"))
        if episode is None:
            continue
        rows.append(
            {
                "episode": episode,
                "success_rate": _float(row.get("success_rate")),
                "pcr": _float(row.get("mean_pcr_successful")),
                "transition_evaluations": _float(row.get("mean_transition_evaluations")),
            }
        )
    return rows


def _plot_validation_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    figures_dir: Path,
    logger: Any,
) -> Optional[Path]:
    usable = [row for row in rows if row.get(metric) is not None]
    if not usable:
        logger.warning("No %s values in validation log", metric)
        return None
    x = np.asarray([row["episode"] for row in usable], dtype=float)
    y = np.asarray([row[metric] for row in usable], dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    ax.plot(
        x,
        y,
        color=SERIES_COLORS[0],
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        label=title,
    )
    _style_axes(ax)
    ax.set_title(title + " during training")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    if metric == "success_rate":
        ax.set_ylim(0, 105)
    ax.legend(loc="best")
    ax.annotate(
        f"{y[-1]:.3f}" if metric == "pcr" else f"{y[-1]:.1f}",
        xy=(x[-1], y[-1]),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    return _save(fig, figures_dir / filename, logger)


# ---------------------------------------------------------------------------
# Test metric charts
# ---------------------------------------------------------------------------
def _algorithm_metric(
    records: Sequence[Mapping[str, Any]],
    algorithm: str,
    metric: str,
    *,
    successful_only: bool = False,
) -> Tuple[Optional[float], Optional[float], int]:
    selected = [
        record
        for record in records
        if str(record["algorithm"]) == algorithm
        and (not successful_only or bool(record.get("success")))
    ]
    values = [
        float(record[metric])
        for record in selected
        if record.get(metric) is not None
    ]
    if not values:
        return None, None, 0
    if len(values) == 1:
        return values[0], 0.0, 1
    return float(np.mean(values)), float(np.std(values, ddof=1)), len(values)


def _bar_chart(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    title: str,
    xlabel: str,
    filename: str,
    figures_dir: Path,
    logger: Any,
    successful_only: bool = False,
    scale: float = 1.0,
    value_format: str = ".2f",
    xlim: Optional[Tuple[float, float]] = None,
    note: Optional[str] = None,
) -> Optional[Path]:
    present = {str(record["algorithm"]) for record in records}
    algorithms = [name for name in PREFERRED_ALGORITHM_ORDER if name in present]
    algorithms.extend(
        sorted(
            (name for name in present if name not in PREFERRED_ALGORITHM_ORDER),
            key=algorithm_sort_key,
        )
    )
    means: List[float] = []
    errors: List[float] = []
    labels: List[str] = []
    kept: List[str] = []
    counts: List[int] = []
    for algorithm in algorithms:
        mean, std, count = _algorithm_metric(
            records,
            algorithm,
            metric,
            successful_only=successful_only,
        )
        if mean is None or std is None:
            continue
        mean *= scale
        std *= scale
        means.append(mean)
        errors.append(std)
        labels.append(f"{mean:{value_format}}")
        kept.append(algorithm)
        counts.append(count)
    if not kept:
        logger.warning("No observations for %s", title)
        return None

    positions = np.arange(len(kept))
    colors = [ALGORITHM_COLORS.get(name, SERIES_COLORS[i % len(SERIES_COLORS)]) for i, name in enumerate(kept)]
    fig, ax = plt.subplots(
        figsize=(8.4, max(3.8, 0.72 * len(kept) + 2.0)),
        constrained_layout=True,
    )
    bars = ax.barh(
        positions,
        means,
        height=0.46,
        color=colors,
        edgecolor="none",
        xerr=errors,
        error_kw={
            "ecolor": INK_SECONDARY,
            "elinewidth": 1.0,
            "capsize": 3,
            "capthick": 1.0,
        },
        zorder=2,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(kept, color=INK)
    ax.invert_yaxis()
    _style_axes(ax, xgrid=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        upper = max(mean + error for mean, error in zip(means, errors))
        ax.set_xlim(0, max(upper * 1.20, upper + 0.02))
    x_low, x_high = ax.get_xlim()
    offset = (x_high - x_low) * 0.012
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            color=INK,
            fontsize=8.5,
            fontweight="semibold",
            clip_on=False,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0},
        )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=INK_SECONDARY,
                linewidth=1.0,
                marker="|",
                markersize=7,
                label="Mean ± std across test instances",
            )
        ],
        loc="lower right",
    )
    if note:
        fig.text(0.01, -0.035, note, color=INK_MUTED, fontsize=8.2, ha="left", va="top")
    return _save(fig, figures_dir / filename, logger)


def visualize(
    *,
    log_dir: str | Path = "logs/16x16_rgnn",
    result_dir: str | Path = "results/16x16_rgnn",
) -> List[Path]:
    _configure_style()
    log_dir = Path(log_dir)
    result_dir = Path(result_dir)
    figures_dir = ensure_dir(result_dir / "figures")
    logger = setup_logger("gsp.v2.visualize", figures_dir / "visualization.log")
    created: List[Path] = []

    loss = plot_training_loss(log_dir / "training_log.csv", figures_dir, logger)
    if loss:
        created.append(loss)
    validation_rows = _validation_rows(log_dir / "validation_log.csv")
    for kwargs in (
        dict(
            metric="success_rate",
            title="Validation Success Rate",
            ylabel="Success rate (%)",
            filename="validation_success_rate.png",
        ),
        dict(
            metric="pcr",
            title="Validation Weighted Path Cost Ratio",
            ylabel="Weighted path cost ratio",
            filename="validation_weighted_path_cost_ratio.png",
        ),
        dict(
            metric="transition_evaluations",
            title="Validation Transition Evaluations",
            ylabel="Mean transition evaluations",
            filename="validation_transition_evaluations.png",
        ),
    ):
        path = _plot_validation_curve(
            validation_rows, figures_dir=figures_dir, logger=logger, **kwargs
        )
        if path:
            created.append(path)

    evaluation_csv = result_dir / "evaluation.csv"
    if not evaluation_csv.is_file():
        logger.warning("Missing evaluation CSV: %s", evaluation_csv)
        return created
    records = read_evaluation_csv(evaluation_csv)
    summary_path = result_dir / "summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    budget = summary.get("budget_transition_evaluations", 512)

    for kwargs in (
        dict(
            metric="success",
            title="Test Success Rate",
            xlabel="Success rate (%)",
            filename="test_success_rate.png",
            scale=100.0,
            value_format=".1f",
            xlim=(0, 120),
            note="Values are percentages. Error bars show standard deviation across the evaluation maps.",
        ),
        dict(
            metric="pcr",
            successful_only=True,
            title="Test Weighted Path Cost Ratio",
            xlabel="Path cost / Dijkstra optimal cost",
            filename="test_weighted_path_cost_ratio.png",
            value_format=".4f",
            note="Computed on solved maps only; failed maps are not assigned a path cost.",
        ),
        dict(
            metric="unique_state_expansions",
            title="Test Unique State Expansions",
            xlabel="Mean unique expanded states per map",
            filename="test_unique_state_expansions.png",
            value_format=".2f",
            note=f"All attempts included. A*/WA*/R-GNN use a {budget}-transition-evaluation budget; Dijkstra is unbudgeted.",
        ),
        dict(
            metric="transition_evaluations",
            title="Test Transition Evaluations",
            xlabel="Mean legal transition evaluations per map",
            filename="test_transition_evaluations.png",
            value_format=".2f",
            note=f"Unified workload metric: A*/WA*/R-GNN use a strict {budget}-evaluation budget; Dijkstra is unbudgeted.",
        ),
        dict(
            metric="search_time_ms",
            title="Test End-to-End Search Time",
            xlabel="Wall-clock time per map (ms)",
            filename="test_search_time.png",
            value_format=".2f",
            note="Includes graph construction, GNN inference, frontier work and path reconstruction; excludes loading/warmup.",
        ),
    ):
        path = _bar_chart(
            records,
            figures_dir=figures_dir,
            logger=logger,
            **kwargs,
        )
        if path:
            created.append(path)
    logger.info("Generated %d V2 figures in %s", len(created), figures_dir)
    return created


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate V2 figures from saved CSV/JSON only")
    parser.add_argument("--log-dir", type=str, default="logs/16x16_rgnn")
    parser.add_argument("--result-dir", type=str, default="results/16x16_rgnn")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    figures = visualize(log_dir=args.log_dir, result_dir=args.result_dir)
    if not figures:
        print("No figures generated; check the supplied log/result directories.")
        return 1
    print(f"Generated {len(figures)} figures in {Path(args.result_dir) / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())