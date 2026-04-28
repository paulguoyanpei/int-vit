# Int-ViT: Fixed-Point Vision Transformer

Int-ViT fine-tunes a pretrained DeiT-Tiny Vision Transformer on CIFAR-10, then runs an integer-only fixed-point inference path for comparison with the FP32 baseline.

The fixed-point model represents values as signed `int64` tensors with `Q` fractional bits. Linear layers, residual paths, attention, MLPs, patch embedding, and classifier projection run in fixed-point arithmetic. Nonlinear operations use dense lookup tables for GELU, Softmax support functions, and LayerNorm variance normalization.

## What This Repo Contains

```
.
├── config.py              # Dataset, model, training, fixed-point, and LUT settings
├── data.py                # CIFAR-10 dataloaders and DeiT-compatible transforms
├── model.py               # timm DeiT-Tiny model factory with a CIFAR-10 head
├── train_fp.py            # FP32 fine-tuning script
├── fixed_point_ops.py     # Quantization helpers, arithmetic primitives, and LUT layers
├── fixed_point_vit.py     # Integer-only DeiT-Tiny inference wrapper
├── evaluate.py            # FP32 vs fixed-point evaluation
├── requirements.txt       # Python dependencies
├── checkpoints/           # Created during training; stores model weights
└── data/                  # Created on first run; stores CIFAR-10
```

## Requirements

- Python 3.8+
- A CUDA-capable GPU is recommended for FP32 fine-tuning
- CPU memory for dense lookup tables during fixed-point evaluation

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs:

- `torch`
- `torchvision`
- `timm`

CIFAR-10 is downloaded automatically into `./data/` the first time a training or evaluation script loads the dataset.

## Quick Start

### 1. Fine-Tune the FP32 Model

```bash
python train_fp.py
```

This trains `deit_tiny_patch16_224` on CIFAR-10 using the settings in `config.py`. The best validation checkpoint is saved as:

```text
checkpoints/deit_tiny_cifar10_best.pth
```

### 2. Evaluate Fixed-Point Inference

```bash
python evaluate.py
```

The evaluation script:

1. Loads the fine-tuned FP32 checkpoint.
2. Reports FP32 validation accuracy.
3. Converts the model to the fixed-point wrapper.
4. Evaluates fixed-point inference on the CIFAR-10 validation set using `config.DEFAULT_Q`.
5. Prints an accuracy summary.

Fixed-point inference runs on CPU with `int64` tensors. It can be much slower than normal FP32 GPU inference.

## Configuration

Edit [config.py](/hpc/home/e1374338/int-vit/config.py) to change training, model, and fixed-point settings.

| Setting | Default | Description |
| --- | ---: | --- |
| `DATASET` | `cifar10` | Dataset name used by this project |
| `NUM_CLASSES` | `10` | CIFAR-10 classifier output classes |
| `IMG_SIZE` | `224` | Input size expected by DeiT |
| `MODEL_NAME` | `deit_tiny_patch16_224` | timm model name |
| `BATCH_SIZE` | `128` | Training and evaluation batch size |
| `LR` | `1e-4` | Fine-tuning learning rate |
| `EPOCHS` | `10` | Fine-tuning epochs |
| `WEIGHT_DECAY` | `0.05` | AdamW weight decay |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `DEFAULT_Q` | `12` | Fixed-point fractional bits used by `evaluate.py` |
| `CHECKPOINT_DIR` | `checkpoints` | Directory for saved weights |

LUT ranges are also configured in [config.py](/hpc/home/e1374338/int-vit/config.py):

| Setting | Default Range | Used For |
| --- | ---: | --- |
| `GELU_LUT_MIN`, `GELU_LUT_MAX` | `[-8.0, 8.0]` | GELU lookup |
| `EXP_LUT_MIN`, `EXP_LUT_MAX` | `[-16.0, 0.0]` | Softmax `exp(x - max(x))` lookup |
| `RECIP_LUT_MIN`, `RECIP_LUT_MAX` | `[0.01, 128.0]` | Softmax reciprocal normalization |
| `RSQRT_LUT_MIN`, `RSQRT_LUT_MAX` | `[0.001, 128.0]` | LayerNorm reciprocal square root |

## Fixed-Point Method

A real value `v` is quantized as:

```text
v_int = round(v * 2^Q)
```

The integer value is dequantized as:

```text
v = v_int / 2^Q
```

Matrix multiplication and elementwise products produce values at a higher scale and are shifted back down with `>> Q`. Biases, residual additions, class tokens, position embeddings, and LayerNorm parameters are stored at the same fixed-point scale.

Dense lookup tables map fixed-point integer inputs to fixed-point integer outputs:

- GELU uses a direct LUT over the configured GELU input range.
- Softmax subtracts the row maximum, applies an `exp` LUT over non-positive inputs, then normalizes with a reciprocal LUT.
- LayerNorm computes integer mean and variance, then applies a reciprocal-square-root LUT for normalization.

Out-of-range LUT inputs are clamped to the configured table range.

## Notes and Limitations

- `evaluate.py` expects `checkpoints/deit_tiny_cifar10_best.pth` to exist. Run `train_fp.py` first.
- The current evaluation path uses `config.DEFAULT_Q`; it does not sweep multiple Q values by default.
- Higher `Q` values improve fractional precision but grow dense LUTs quickly. Wide LUT ranges at high `Q` can consume significant memory.
- The fixed-point path dequantizes only the final logits so accuracy can be computed with normal PyTorch tensor operations.
- This implementation is intended for experimentation and analysis, not optimized deployment.
