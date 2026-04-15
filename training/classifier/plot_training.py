#!/usr/bin/env python3
"""Plot classifier training curves from saved metrics.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        nargs="+",
        help="One or more metrics.json files. Multiple files are concatenated in argument order.",
    )
    parser.add_argument("--output", required=True, help="Output image path, usually .png.")
    parser.add_argument(
        "--title",
        default="Classifier CNN Training",
        help="Figure title.",
    )
    return parser.parse_args()


def load_history(paths: list[str]) -> tuple[list[int], dict[str, list[float]]]:
    epochs: list[int] = []
    series: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_roc_auc": [],
        "val_average_precision": [],
        "val_precision": [],
        "val_recall": [],
        "val_top20_precision": [],
    }

    offset = 0
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        for row in metrics["history"]:
            epochs.append(offset + int(row["epoch"]))
            val = row["val"]
            series["train_loss"].append(float(row["train_loss"]))
            series["val_loss"].append(float(val["loss"]))
            series["val_accuracy"].append(float(val["accuracy"]))
            series["val_roc_auc"].append(float(val["roc_auc"]))
            series["val_average_precision"].append(float(val["average_precision"]))
            series["val_precision"].append(float(val["precision"]))
            series["val_recall"].append(float(val["recall"]))
            series["val_top20_precision"].append(float(val["precision_at_positive_count"]))
        offset = epochs[-1] if epochs else offset
    return epochs, series


def main() -> int:
    args = parse_args()
    epochs, series = load_history(args.metrics)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    fig.suptitle(args.title, fontsize=16, fontweight="bold")

    axes[0, 0].plot(epochs, series["train_loss"], label="train loss", linewidth=2)
    axes[0, 0].plot(epochs, series["val_loss"], label="validation loss", linewidth=2)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, series["val_accuracy"], label="validation accuracy", linewidth=2)
    axes[0, 1].axhline(0.80, color="#777777", linestyle="--", linewidth=1.5, label="all-negative baseline")
    axes[0, 1].axhline(0.90, color="#2f855a", linestyle=":", linewidth=1.5, label="90% target")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0.45, 0.95)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, series["val_roc_auc"], label="ROC-AUC", linewidth=2)
    axes[1, 0].plot(epochs, series["val_average_precision"], label="average precision", linewidth=2)
    axes[1, 0].axhline(0.20, color="#777777", linestyle="--", linewidth=1.5, label="AP baseline")
    axes[1, 0].set_title("Ranking Metrics")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim(0.15, 1.0)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, series["val_precision"], label="precision", linewidth=2)
    axes[1, 1].plot(epochs, series["val_recall"], label="recall", linewidth=2)
    axes[1, 1].plot(epochs, series["val_top20_precision"], label="top-20% precision", linewidth=2)
    axes[1, 1].axhline(0.20, color="#777777", linestyle="--", linewidth=1.5, label="base rate")
    axes[1, 1].set_title("Positive-Class Metrics")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xticks(epochs)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, bbox_inches="tight")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
