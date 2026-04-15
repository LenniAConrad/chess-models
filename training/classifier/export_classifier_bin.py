#!/usr/bin/env python3
"""Export a trained classifier CNN checkpoint to the ChessRTK .bin format."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from train_classifier import ClassifierCnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Input PyTorch model.pt checkpoint.")
    parser.add_argument("--output", required=True, help="Output classifier .bin path.")
    return parser.parse_args()


def as_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype("<f4", copy=False)


def fold_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    weight = as_np(conv.weight)
    gamma = as_np(bn.weight)
    beta = as_np(bn.bias)
    mean = as_np(bn.running_mean)
    var = as_np(bn.running_var)
    scale = gamma / np.sqrt(var + bn.eps)
    folded_weight = weight * scale[:, None, None, None]
    folded_bias = beta - mean * scale
    return folded_weight.astype("<f4", copy=False), folded_bias.astype("<f4", copy=False)


def write_i32(fh, value: int) -> None:
    fh.write(struct.pack("<i", value))


def write_f32_array(fh, values: np.ndarray) -> None:
    flat = np.ascontiguousarray(values.reshape(-1), dtype="<f4")
    write_i32(fh, int(flat.size))
    fh.write(flat.tobytes(order="C"))


def write_conv(fh, weight: np.ndarray, bias: np.ndarray) -> None:
    out_channels, in_channels, kernel_h, kernel_w = weight.shape
    if kernel_h != kernel_w:
        raise ValueError(f"Non-square conv kernel: {weight.shape}")
    write_i32(fh, int(out_channels))
    write_i32(fh, int(in_channels))
    write_i32(fh, int(kernel_h))
    write_f32_array(fh, weight)
    write_f32_array(fh, bias)


def write_dense(fh, layer: torch.nn.Linear) -> None:
    weight = as_np(layer.weight)
    bias = as_np(layer.bias)
    out_dim, in_dim = weight.shape
    write_i32(fh, int(out_dim))
    write_i32(fh, int(in_dim))
    write_f32_array(fh, weight)
    write_f32_array(fh, bias)


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    model = ClassifierCnn()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        fh.write(b"CLSF")
        write_i32(fh, 1)
        write_i32(fh, 21)
        write_i32(fh, 64)
        write_i32(fh, 6)
        write_i32(fh, 32)
        write_i32(fh, 1)

        write_conv(fh, *fold_conv_bn(model.stem[0], model.stem[1]))
        for block in model.trunk:
            write_conv(fh, *fold_conv_bn(block.conv1, block.bn1))
            write_conv(fh, *fold_conv_bn(block.conv2, block.bn2))
        write_conv(fh, *fold_conv_bn(model.head[0], model.head[1]))
        write_dense(fh, model.output)

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
