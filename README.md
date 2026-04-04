# Fixed-Point Vision Transformer (Int-ViT)

Integer-only inference for Vision Transformers. This project fine-tunes a pretrained DeiT-Tiny on CIFAR-10, then converts all operations to fixed-point arithmetic (int64) using lookup tables for non-linear functions (GELU, Softmax, LayerNorm). The goal is to evaluate how classification accuracy degrades across different fixed-point precision levels (Q = 8, 12, 16, 20, 24 fractional bits).

## Project Structure

```
├── config.py              # Hyperparameters, LUT settings, paths
├── data.py                # CIFAR-10 data loading and transforms
├── model.py               # Pretrained DeiT-Tiny model loading (via timm)
├── train_fp.py            # FP32 fine-tuning on CIFAR-10
├── fixed_point_ops.py     # Fixed-point arithmetic primitives and LUT-based layers
├── fixed_point_vit.py     # Full fixed-point ViT wrapper for integer-only inference
├── evaluate.py            # FP32 vs fixed-point accuracy comparison across Q values
├── checkpoints/           # Saved model weights
└── data/                  # CIFAR-10 dataset (auto-downloaded)
```

## Environment Setup

**Requirements:** Python 3.8+, CUDA-capable GPU (recommended for training)

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   This installs `torch`, `torchvision`, and `timm`.

> CIFAR-10 is downloaded automatically on the first run to `./data/`.

## Usage

### 1. Fine-tune the FP32 baseline

Train DeiT-Tiny on CIFAR-10 for 10 epochs (parameters configurable in `config.py`):

```bash
python train_fp.py
```

The best checkpoint is saved to `checkpoints/deit_tiny_cifar10_best.pth`.

### 2. Evaluate fixed-point inference

Compare the FP32 baseline against fixed-point models at Q = 8, 12, 16, 20, 24:

```bash
python evaluate.py
```

This will:
- Load the fine-tuned FP32 model and report its accuracy
- Convert the model to fixed-point at each Q value
- Run inference on the CIFAR-10 validation set using integer-only arithmetic
- Print a summary table of accuracy and accuracy drop for each Q

### Configuration

Key settings in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `deit_tiny_patch16_224` | Backbone model (5.7M params) |
| `BATCH_SIZE` | 128 | Training and evaluation batch size |
| `LR` | 1e-4 | Learning rate for fine-tuning |
| `EPOCHS` | 10 | Number of fine-tuning epochs |
| `Q_VALUES` | [8, 12, 16, 20, 24] | Fractional bits to sweep during evaluation |
| `*_LUT_SIZE` | 65536 | Number of entries in each lookup table |

## How It Works

**Fixed-point representation:** A real value `v` is stored as the integer `round(v * 2^Q)`. All arithmetic (matrix multiplies, additions, residual connections) is performed on `int64` tensors using bit-shifts for rescaling.

**Lookup tables** replace non-linear functions that cannot be computed with integer arithmetic:
- **GELU** — precomputed over [-8, 8]
- **Softmax** — uses an `exp()` LUT (over [-16, 0]) and a `1/x` reciprocal LUT for normalization
- **LayerNorm** — uses a `1/sqrt(x)` LUT for variance normalization
