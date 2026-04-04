"""Load pretrained DeiT-Tiny and adapt for CIFAR-10."""

import timm
import config


def get_model(pretrained=True):
    """Load DeiT-Tiny with CIFAR-10 head (10 classes)."""
    model = timm.create_model(
        config.MODEL_NAME,
        pretrained=pretrained,
        num_classes=config.NUM_CLASSES,
    )
    return model
