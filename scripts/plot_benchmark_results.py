

"""Plot benchmark results from a saved benchmark run.

This script expects the output of `scripts/benchmark_models.py` and can read either
the run directory or the `benchmark_results.json` / `benchmark_results.csv` file
directly.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot VQA benchmark results")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Benchmark run directory or benchmark_results.json/csv file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where plots are written. Defaults to <run>/plots.",
    )
    parser.add_argument(
        "--include_artifacts",
        action="store_true",
        help="Also plot per-class F1 and confusion matrices from model artifacts when available.",
    )
    return parser.parse_args()


def resolve_report_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    json_path = input_path / "benchmark_results.json"
    csv_path = input_path / "benchmark_results.csv"

    if json_path.exists():
        return json_path
    if csv_path.exists():
        return csv_path

    raise FileNotFoundError(
        f"Could not find benchmark_results.json or benchmark_results.csv in {input_path}"
    )


def parse_optional_mapping(value: Any) -> dict[str, float] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, dict):
        return {str(key): float(score) for key, score in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "nan"}:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return {str(key): float(score) for key, score in parsed.items()}
    return None


def load_results(report_path: Path) -> pd.DataFrame:
    if report_path.suffix.lower() == ".json":
        with open(report_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        frame = pd.DataFrame(payload)
    elif report_path.suffix.lower() == ".csv":
        frame = pd.read_csv(report_path)
    else:
        raise ValueError(f"Unsupported report file: {report_path}")

    for column in ["accuracy", "macro_f1", "weighted_f1", "num_samples"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "per_class_f1" in frame.columns:
        frame["per_class_f1"] = frame["per_class_f1"].apply(parse_optional_mapping)

    return frame


def plot_metric_bars(frame: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("accuracy", "Accuracy (%)"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
    ]

    if "status" in frame.columns:
        ok_frame = frame.loc[frame["status"].fillna("ok") == "ok"].copy()
    else:
        ok_frame = frame.copy()

    if ok_frame.empty:
        return

    ok_frame = ok_frame.sort_values(by="accuracy", ascending=False, na_position="last")
    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    color_palette = sns.color_palette("deep", n_colors=len(ok_frame))

    for axis, (metric, label) in zip(axes, metrics, strict=True):
        values = ok_frame[metric]
        bars = axis.bar(ok_frame["name"], values, color=color_palette)
        axis.set_title(label)
        axis.set_xlabel("Model")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylim(bottom=0)

        for bar, value in zip(bars, values, strict=True):
            if pd.isna(value):
                continue
            axis.annotate(
                f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=10,
            )

    fig.suptitle("Benchmark Comparison", fontsize=18)
    fig.savefig(output_dir / "benchmark_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_status_summary(frame: pd.DataFrame, output_dir: Path) -> None:
    if "status" not in frame.columns:
        return

    status_counts = frame["status"].fillna("unknown").value_counts().sort_index()
    if status_counts.empty:
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = axis.bar(
        status_counts.index.astype(str),
        status_counts.values,
        color=sns.color_palette("muted", n_colors=len(status_counts)),
    )
    axis.set_title("Benchmark Status Summary")
    axis.set_xlabel("Status")
    axis.set_ylabel("Number of Models")

    for bar, value in zip(bars, status_counts.values, strict=True):
        axis.annotate(
            f"{int(value)}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    fig.savefig(output_dir / "benchmark_status.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_heatmap(frame: pd.DataFrame, output_dir: Path) -> None:
    if "per_class_f1" not in frame.columns:
        return

    records: list[pd.Series] = []
    for _, row in frame.iterrows():
        mapping = row.get("per_class_f1")
        if not isinstance(mapping, dict) or not mapping:
            continue
        records.append(pd.Series(mapping, name=row["name"]))

    if not records:
        return

    matrix = pd.DataFrame(records)
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    sns.set_theme(style="white", context="talk")
    fig_height = max(4.5, 0.6 * len(matrix.index) + 2)
    fig_width = max(10, 0.5 * len(matrix.columns) + 4)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "F1"},
        ax=axis,
    )
    axis.set_title("Per-Class F1 by Model")
    axis.set_xlabel("Class")
    axis.set_ylabel("Model")
    fig.savefig(output_dir / "per_class_f1_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(path: Path, model_name: str, output_dir: Path) -> None:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    labels = payload.get("labels") or []
    matrix = payload.get("matrix") or []
    if not labels or not matrix:
        return

    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    sns.set_theme(style="white", context="talk")
    fig_size = max(8, 0.5 * len(labels) + 4)
    fig, axis = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)
    sns.heatmap(
        frame,
        annot=False,
        cmap="mako",
        square=True,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Count"},
        ax=axis,
    )
    axis.set_title(f"Confusion Matrix: {model_name}")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.tick_params(axis="x", rotation=45)
    axis.tick_params(axis="y", rotation=0)
    safe_name = model_name.replace("/", "-").replace(" ", "_")
    fig.savefig(output_dir / f"confusion_matrix_{safe_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_optional_artifacts(run_dir: Path, frame: pd.DataFrame, output_dir: Path) -> None:
    if "confusion_matrix_path" not in frame.columns:
        return

    for _, row in frame.iterrows():
        confusion_value = row.get("confusion_matrix_path")
        if not isinstance(confusion_value, str) or not confusion_value:
            continue
        confusion_path = Path(confusion_value)
        if not confusion_path.is_absolute():
            confusion_path = run_dir / confusion_path
        if confusion_path.exists():
            plot_confusion_matrix(confusion_path, str(row.get("name", confusion_path.parent.name)), output_dir)


def write_summary_table(frame: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        column
        for column in ["name", "status", "accuracy", "macro_f1", "weighted_f1", "num_samples", "notes"]
        if column in frame.columns
    ]
    summary = frame.loc[:, columns].copy()
    summary.to_csv(output_dir / "benchmark_plot_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    report_path = resolve_report_path(input_path)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif input_path.is_dir():
        output_dir = input_path / "plots"
    else:
        output_dir = report_path.parent / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_results(report_path)
    write_summary_table(frame, output_dir)
    plot_status_summary(frame, output_dir)
    plot_metric_bars(frame, output_dir)

    if args.include_artifacts:
        run_dir = report_path.parent if report_path.is_file() else input_path
        plot_per_class_heatmap(frame, output_dir)
        plot_optional_artifacts(run_dir, frame, output_dir)

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()

# uv run python plot_benchmark_results.py --input outputs/benchmarks/run_001 --include_artifacts