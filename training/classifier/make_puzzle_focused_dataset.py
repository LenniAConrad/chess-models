#!/usr/bin/env python3
"""Create a puzzle-focused classifier dataset from a full classifier export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-base", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument(
        "--negatives-per-positive",
        type=float,
        default=1.0,
        help="How many real negative rows to sample per positive row.",
    )
    parser.add_argument(
        "--max-gib",
        type=float,
        default=48.0,
        help="Maximum focused dataset tensor size in GiB, leaving headroom under a 50 GiB RAM budget.",
    )
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--chunk-rows", type=int, default=16384)
    return parser.parse_args()


def paths(base: str) -> tuple[Path, Path, Path]:
    stem = Path(base)
    return (
        stem.with_name(stem.name + ".classifier.inputs.npy"),
        stem.with_name(stem.name + ".classifier.labels.npy"),
        stem.with_name(stem.name + ".classifier.meta.json"),
    )


def main() -> int:
    args = parse_args()
    if args.negatives_per_positive < 0.0:
        raise ValueError("--negatives-per-positive must be non-negative")

    source_inputs, source_labels, source_meta = paths(args.source_base)
    out_inputs, out_labels, out_meta = paths(args.output_base)
    out_inputs.parent.mkdir(parents=True, exist_ok=True)

    x = np.load(source_inputs, mmap_mode="r")
    y = np.load(source_labels, mmap_mode="r")
    pos_idx = np.flatnonzero(y == 1.0)
    neg_idx = np.flatnonzero(y == 0.0)
    if len(pos_idx) == 0:
        raise ValueError("Source dataset has no positive rows")

    rng = np.random.default_rng(args.seed)
    row_bytes = int(x.shape[1] * np.dtype(np.float32).itemsize + np.dtype(np.float32).itemsize)
    max_rows_by_size = int((args.max_gib * 1024**3) // row_bytes)
    if len(pos_idx) > max_rows_by_size:
        raise ValueError(
            f"All positives alone need {len(pos_idx) * row_bytes / 1024**3:.2f} GiB, "
            f"which exceeds --max-gib={args.max_gib}"
        )
    max_neg_by_ratio = int(round(len(pos_idx) * args.negatives_per_positive))
    max_neg_by_size = max_rows_by_size - len(pos_idx)
    max_neg = min(len(neg_idx), max_neg_by_ratio, max_neg_by_size)
    sampled_neg = rng.choice(neg_idx, size=max_neg, replace=False) if max_neg > 0 else np.empty(0, dtype=neg_idx.dtype)
    selected = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(selected)

    out_x = open_memmap(out_inputs, mode="w+", dtype=np.float32, shape=(len(selected), x.shape[1]))
    out_y = open_memmap(out_labels, mode="w+", dtype=np.float32, shape=(len(selected),))
    for start in range(0, len(selected), args.chunk_rows):
        chunk_idx = selected[start : start + args.chunk_rows]
        out_x[start : start + len(chunk_idx)] = x[chunk_idx]
        out_y[start : start + len(chunk_idx)] = y[chunk_idx]
    out_x.flush()
    out_y.flush()

    with source_meta.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    focused_meta = {
        "source": str(source_meta),
        "source_rows": int(len(y)),
        "source_positives": int(len(pos_idx)),
        "source_negatives": int(len(neg_idx)),
        "rows_written": int(len(selected)),
        "positives": int(len(pos_idx)),
        "negatives": int(len(sampled_neg)),
        "negatives_per_positive": float(args.negatives_per_positive),
        "max_gib": float(args.max_gib),
        "estimated_tensor_gib": float(len(selected) * row_bytes / 1024**3),
        "seed": int(args.seed),
        "inputs": {
            "shape": [int(len(selected)), int(x.shape[1])],
            "encoder": meta.get("inputs", {}).get("encoder", "classifier-21planes"),
        },
        "labels": meta.get("labels", {}),
    }
    with out_meta.open("w", encoding="utf-8") as fh:
        json.dump(focused_meta, fh, indent=2, sort_keys=True)

    print(
        f"Wrote {out_inputs} and {out_labels} "
        f"({len(selected)} rows: {len(pos_idx)} positive, {len(sampled_neg)} negative)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
