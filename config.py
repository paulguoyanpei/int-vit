"""Configuration for fixed-point ViT project."""

# Dataset
DATASET = "cifar10"
NUM_CLASSES = 10
IMG_SIZE = 224  # DeiT expects 224x224

# Model
MODEL_NAME = "deit_tiny_patch16_224"  # 5.7M params

# Training (FP32 fine-tuning on CIFAR-10)
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 10
WEIGHT_DECAY = 0.05
NUM_WORKERS = 4

# Fixed-point
DEFAULT_Q = 12

# LUT settings
# Dense LUT sizes are derived from each range and DEFAULT_Q:
# size = round(max_val * 2^Q) - round(min_val * 2^Q) + 1
# Input range for GELU LUT (in real-valued domain)
GELU_LUT_MIN = -8.0
GELU_LUT_MAX = 8.0

# Input range for exp LUT (softmax shifts inputs so they are <= 0)
EXP_LUT_MIN = -16.0
EXP_LUT_MAX = 0.0

# Reciprocal LUT range (for softmax normalization)
RECIP_LUT_MIN = 0.01
RECIP_LUT_MAX = 128.0

# Reciprocal sqrt LUT range (for LayerNorm)
RSQRT_LUT_MIN = 0.001
RSQRT_LUT_MAX = 128.0

# Paths
CHECKPOINT_DIR = "checkpoints"
