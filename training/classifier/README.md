# Classifier Training

Local experiments for the ChessRTK 21-plane binary classifier export.

The copied USB-stack dataset is stored locally under:

```text
training/classifier/data/usb-stacks-20260415-130734/
```

It contains:

- `*.classifier.inputs.npy`: float32 inputs shaped `(N, 1344)`
- `*.classifier.labels.npy`: float32 binary labels shaped `(N,)`
- `*.classifier.meta.json`: export metadata and label definition

Run a small baseline:

```bash
python3 training/classifier/train_classifier.py \
  --base training/classifier/data/usb-stacks-20260415-130734/usb-stacks-20260415-130734 \
  --model mlp \
  --epochs 5
```

Run the classifier residual CNN with a realistic 1:4 puzzle/non-puzzle split:

```bash
python3 training/classifier/train_classifier.py \
  --base training/classifier/data/usb-stacks-20260415-130734/usb-stacks-20260415-130734 \
  --model cnn \
  --positive-rate 0.2 \
  --weighted-loss \
  --limit 10000 \
  --epochs 3
```

Outputs are written under `training/classifier/runs/`.

Run the full USB pipeline:

```bash
training/classifier/run_full_usb_pipeline.sh
```

The full pipeline scans all USB stack rows, writes every real puzzle row, and
caps real non-puzzle rows at `MAX_NEGATIVES=3000000` by default. This avoids a
giant mostly-non-puzzle dataset while still using as many actual puzzles as the
USB contains.

CUDA is used automatically when PyTorch can see it:

```bash
DEVICE=auto training/classifier/run_full_usb_pipeline.sh
```

Force CUDA and fail fast if it is unavailable:

```bash
DEVICE=cuda CUDA_DEVICE=0 training/classifier/run_full_usb_pipeline.sh
```

Generate the training graph:

```bash
python3 training/classifier/plot_training.py \
  training/classifier/runs/20260415-134613-cnn/metrics.json \
  training/classifier/runs/20260415-140257-cnn/metrics.json \
  --output training/classifier/runs/classifier-cnn-training.png \
  --title "Classifier CNN 1:4 Training"
```

Continue from an existing checkpoint:

```bash
python3 training/classifier/train_classifier.py \
  --base training/classifier/data/usb-stacks-20260415-130734/usb-stacks-20260415-130734 \
  --model cnn \
  --device auto \
  --positive-rate 0.2 \
  --init-model training/classifier/runs/20260415-134613-cnn/model.pt \
  --epochs 6 \
  --lr 0.0003
```

Export a trained CNN checkpoint to the ChessRTK classifier `.bin` format:

```bash
python3 training/classifier/export_classifier_bin.py \
  --checkpoint training/classifier/runs/20260415-141952-cnn/model.pt \
  --output models/puzzle-classifier_21planes-6blocksx64-head32-logit1.bin
```
