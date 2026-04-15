#!/usr/bin/env python3
"""Small baseline trainer for ChessRTK classifier NPY exports."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        required=True,
        help="Dataset stem without .classifier.inputs.npy / .classifier.labels.npy suffix.",
    )
    parser.add_argument(
        "--out-dir",
        default="training/classifier/runs",
        help="Directory for run artifacts.",
    )
    parser.add_argument("--model", choices=("linear", "mlp", "cnn"), default="linear")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--init-model",
        default="",
        help="Optional model.pt checkpoint to initialize from before training.",
    )
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row limit after class-ratio sampling. 0 means use the largest available subset for the requested ratio.",
    )
    parser.add_argument(
        "--positive-rate",
        type=float,
        default=0.5,
        help="Target positive rate for train/validation/test splits. Ignored with --use-all-data.",
    )
    parser.add_argument(
        "--use-all-data",
        action="store_true",
        help="Use every labeled row and preserve the dataset's class ratio.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Use BCE pos_weight based on the training split class ratio.",
    )
    parser.add_argument(
        "--oversample-positives-to-rate",
        type=float,
        default=0.0,
        help="Duplicate positive rows in the training split until this positive rate is reached. 0 disables oversampling.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without validation improvement. 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation ROC-AUC gain counted as improvement.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Torch CPU thread count. 0 keeps the library default.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index used when --device cuda or auto selects CUDA.",
    )
    return parser.parse_args()


def dataset_paths(base: str) -> tuple[Path, Path, Path]:
    stem = Path(base)
    return (
        stem.with_name(stem.name + ".classifier.inputs.npy"),
        stem.with_name(stem.name + ".classifier.labels.npy"),
        stem.with_name(stem.name + ".classifier.meta.json"),
    )


def stratified_split(
    labels: np.ndarray,
    seed: int,
    limit: int,
    positive_rate: float,
    use_all_data: bool = False,
    train_fraction: float = 0.80,
    val_fraction: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("--val-fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("--train-fraction + --val-fraction must be less than 1")
    if not use_all_data and not (0.0 < positive_rate < 1.0):
        raise ValueError("--positive-rate must be between 0 and 1")
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels == 1.0)
    neg = np.flatnonzero(labels == 0.0)
    rng.shuffle(pos)
    rng.shuffle(neg)

    if use_all_data:
        if limit > 0:
            scale = min(limit / max(1, len(pos) + len(neg)), 1.0)
            pos_count = max(1, int(len(pos) * scale))
            neg_count = max(1, int(len(neg) * scale))
        else:
            pos_count = len(pos)
            neg_count = len(neg)
    elif limit > 0:
        pos_count = max(1, int(round(limit * positive_rate)))
        neg_count = max(1, limit - pos_count)
        scale = min(len(pos) / pos_count, len(neg) / neg_count, 1.0)
        pos_count = max(1, int(pos_count * scale))
        neg_count = max(1, int(neg_count * scale))
    else:
        pos_count = min(len(pos), int(len(neg) * positive_rate / (1.0 - positive_rate)))
        neg_count = min(len(neg), int(pos_count * (1.0 - positive_rate) / positive_rate))
    pos = pos[:pos_count]
    neg = neg[:neg_count]

    def split_class(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(indices)
        n_train = int(n * train_fraction)
        n_val = int(n * val_fraction)
        train = indices[:n_train]
        val = indices[n_train : n_train + n_val]
        test = indices[n_train + n_val :]
        return train, val, test

    p_train, p_val, p_test = split_class(pos)
    n_train, n_val, n_test = split_class(neg)

    train = np.concatenate([p_train, n_train])
    val = np.concatenate([p_val, n_val])
    test = np.concatenate([p_test, n_test])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def oversample_positives(
    labels: np.ndarray,
    train_idx: np.ndarray,
    target_rate: float,
    seed: int,
) -> np.ndarray:
    if target_rate <= 0.0:
        return train_idx
    if not (0.0 < target_rate < 1.0):
        raise ValueError("--oversample-positives-to-rate must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train_labels = np.asarray(labels[train_idx])
    pos = train_idx[train_labels == 1.0]
    neg = train_idx[train_labels == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        return train_idx
    current_rate = len(pos) / len(train_idx)
    if current_rate >= target_rate:
        return train_idx
    desired_pos = int(np.ceil(target_rate * len(neg) / (1.0 - target_rate)))
    extra = max(0, desired_pos - len(pos))
    sampled = rng.choice(pos, size=extra, replace=True)
    oversampled = np.concatenate([train_idx, sampled])
    rng.shuffle(oversampled)
    return oversampled


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class ClassifierCnn(torch.nn.Module):
    """Training-time version of ChessRTK's classifier residual CNN."""

    def __init__(
        self,
        input_channels: int = 21,
        trunk_channels: int = 64,
        residual_blocks: int = 6,
        head_channels: int = 32,
    ) -> None:
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, trunk_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(trunk_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.trunk = torch.nn.Sequential(
            *[ResidualBlock(trunk_channels) for _ in range(residual_blocks)]
        )
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(trunk_channels, head_channels, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(head_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.output = torch.nn.Linear(head_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(-1, 21, 8, 8)
        x = self.stem(x)
        x = self.trunk(x)
        x = self.head(x)
        x = x.mean(dim=(2, 3))
        return self.output(x)


def make_model(kind: str, input_dim: int) -> torch.nn.Module:
    if kind == "linear":
        return torch.nn.Linear(input_dim, 1)
    if kind == "cnn":
        if input_dim != 21 * 64:
            raise ValueError(f"CNN expects 21*64 inputs, got {input_dim}")
        return ClassifierCnn()
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.10),
        torch.nn.Linear(128, 1),
    )


def iterate_batches(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device | None = None,
    pin_memory: bool = False,
):
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        xb = torch.from_numpy(np.asarray(x[batch_idx], dtype=np.float32))
        yb = torch.from_numpy(np.asarray(y[batch_idx], dtype=np.float32)).view(-1, 1)
        if pin_memory:
            xb = xb.pin_memory()
            yb = yb.pin_memory()
        if device is not None:
            non_blocking = bool(pin_memory and device.type == "cuda")
            xb = xb.to(device, non_blocking=non_blocking)
            yb = yb.to(device, non_blocking=non_blocking)
        yield xb, yb


def select_device(requested: str, cuda_device: int) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        return torch.device(f"cuda:{cuda_device}")
    if cuda_available:
        return torch.device(f"cuda:{cuda_device}")
    return torch.device("cpu")


def device_info(device: torch.device) -> dict[str, object]:
    info: dict[str, object] = {
        "selected": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "cuda_device_index": int(idx),
            "cuda_device_name": torch.cuda.get_device_name(idx),
            "cuda_capability": list(torch.cuda.get_device_capability(idx)),
            "cuda_total_memory_bytes": int(props.total_memory),
        })
    return info


def evaluate(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float | int | list[list[int]]]:
    model.eval()
    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    loss_fn = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for xb, yb in iterate_batches(
            x,
            y,
            indices,
            batch_size,
            device=device,
            pin_memory=(device.type == "cuda"),
        ):
            logits = model(xb)
            loss = loss_fn(logits, yb)
            losses.append(float(loss.item()) * len(yb))
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            all_probs.append(probs)
            all_labels.append(yb.detach().cpu().numpy().reshape(-1))

    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    preds = (probs >= threshold).astype(np.float32)
    cm = confusion_matrix(labels, preds, labels=[0.0, 1.0])
    order = np.argsort(-probs)
    positives = max(1, int(labels.sum()))
    top_k = positives
    top_k_hits = int(labels[order[:top_k]].sum())
    top_k_precision = top_k_hits / max(1, top_k)
    return {
        "loss": float(sum(losses) / max(1, len(indices))),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
        "positive_rate": float(np.mean(labels)),
        "baseline_accuracy_all_negative": float(1.0 - np.mean(labels)),
        "baseline_average_precision": float(np.mean(labels)),
        "precision_at_positive_count": float(top_k_precision),
        "confusion_matrix_0_1": cm.astype(int).tolist(),
    }


def collect_probs(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in iterate_batches(
            x,
            y,
            indices,
            batch_size,
            device=device,
            pin_memory=(device.type == "cuda"),
        ):
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            all_probs.append(probs)
            all_labels.append(yb.detach().cpu().numpy().reshape(-1))
    return np.concatenate(all_probs), np.concatenate(all_labels)


def best_threshold_for_accuracy(probs: np.ndarray, labels: np.ndarray) -> dict[str, float | int | list[list[int]]]:
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], probs)))
    best: dict[str, float | int | list[list[int]]] | None = None
    for threshold in thresholds:
        preds = (probs >= threshold).astype(np.float32)
        acc = accuracy_score(labels, preds)
        recall = recall_score(labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        # Accuracy is the requested objective, but prefer useful classifiers
        # over trivial all-negative ties.
        score = (acc, f1, recall)
        if best is None or score > (best["accuracy"], best["f1"], best["recall"]):
            cm = confusion_matrix(labels, preds, labels=[0.0, 1.0])
            best = {
                "threshold": float(threshold),
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion_matrix_0_1": cm.astype(int).tolist(),
            }
    assert best is not None
    return best


def main() -> int:
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    inputs_path, labels_path, meta_path = dataset_paths(args.base)
    x = np.load(inputs_path, mmap_mode="r")
    y = np.load(labels_path, mmap_mode="r")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D inputs, got {x.shape}")
    if y.shape != (x.shape[0],):
        raise ValueError(f"Labels shape {y.shape} does not match inputs shape {x.shape}")

    train_idx, val_idx, test_idx = stratified_split(
        y,
        args.seed,
        args.limit,
        args.positive_rate,
        use_all_data=args.use_all_data,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    train_epoch_idx = oversample_positives(
        y,
        train_idx,
        args.oversample_positives_to_rate,
        args.seed + 2,
    )
    device = select_device(args.device, args.cuda_device)
    print(json.dumps({"device": device_info(device)}, sort_keys=True))
    model = make_model(args.model, int(x.shape[1])).to(device)
    if args.init_model:
        model.load_state_dict(torch.load(args.init_model, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    train_epoch_labels = np.asarray(y[train_epoch_idx])
    train_pos = float(np.sum(train_epoch_labels == 1.0))
    train_neg = float(len(train_epoch_idx) - train_pos)
    pos_weight = train_neg / max(1.0, train_pos)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device) if args.weighted_loss else None
    )
    rng = np.random.default_rng(args.seed + 1)

    run_name = time.strftime("%Y%m%d-%H%M%S") + f"-{args.model}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    best_val = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_epoch_idx)
        train_loss = 0.0
        for xb, yb in iterate_batches(
            x,
            y,
            train_epoch_idx,
            args.batch_size,
            device=device,
            pin_memory=(device.type == "cuda"),
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(yb)

        val_metrics = evaluate(model, x, y, val_idx, args.batch_size, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss / max(1, len(train_epoch_idx))),
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        val_score = float(val_metrics["roc_auc"])
        if val_score > best_val + args.early_stopping_min_delta:
            best_val = val_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                stopped_early = True
                print(json.dumps({
                    "early_stopping": True,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_roc_auc": best_val,
                    "epochs_without_improvement": epochs_without_improvement,
                }, sort_keys=True))
                break

    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device, weights_only=True))
    test_metrics = evaluate(model, x, y, test_idx, args.batch_size, device)
    val_probs, val_labels = collect_probs(model, x, y, val_idx, args.batch_size, device)
    best_accuracy_threshold = best_threshold_for_accuracy(val_probs, val_labels)
    test_metrics_at_best_accuracy_threshold = evaluate(
        model,
        x,
        y,
        test_idx,
        args.batch_size,
        device,
        threshold=float(best_accuracy_threshold["threshold"]),
    )
    elapsed = time.time() - start_time

    with meta_path.open("r", encoding="utf-8") as fh:
        source_meta = json.load(fh)

    metrics = {
        "config": vars(args),
        "device": device_info(device),
        "dataset": {
            "inputs_path": str(inputs_path),
            "labels_path": str(labels_path),
            "meta_path": str(meta_path),
            "shape": list(x.shape),
            "positive_rate": float(np.mean(y)),
            "source_rows_written": source_meta.get("rows_written"),
            "source_positives": source_meta.get("positives"),
            "source_negatives": source_meta.get("negatives"),
            "label_definition": source_meta.get("labels", {}).get("positive_definition"),
        },
        "split": {
            "train_rows": int(len(train_idx)),
            "val_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
            "sampled_positive_rate": float(np.mean(np.asarray(y[np.concatenate([train_idx, val_idx, test_idx])]))),
            "train_positive_rows": int(np.sum(np.asarray(y[train_idx]) == 1.0)),
            "train_negative_rows": int(np.sum(np.asarray(y[train_idx]) == 0.0)),
            "train_epoch_rows": int(len(train_epoch_idx)),
            "train_epoch_positive_rate": float(np.mean(np.asarray(y[train_epoch_idx]))),
            "train_pos_weight": float(pos_weight),
        },
        "best_epoch": int(best_epoch),
        "best_val_roc_auc": float(best_val),
        "stopped_early": bool(stopped_early),
        "history": history,
        "test": test_metrics,
        "validation_best_accuracy_threshold": best_accuracy_threshold,
        "test_at_validation_best_accuracy_threshold": test_metrics_at_best_accuracy_threshold,
        "elapsed_seconds": float(elapsed),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)

    print("wrote", out_dir)
    print(json.dumps({
        "best_epoch": best_epoch,
        "test": test_metrics,
        "validation_best_accuracy_threshold": best_accuracy_threshold,
        "test_at_validation_best_accuracy_threshold": test_metrics_at_best_accuracy_threshold,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
