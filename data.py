"""CIFAR-10 data loading with transforms suitable for DeiT."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config


def get_transforms():
    """Get train/val transforms. Resize CIFAR-10 (32x32) to 224x224 for DeiT."""
    train_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.RandomCrop(config.IMG_SIZE, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def get_dataloaders(batch_size=None, num_workers=None):
    """Return train and val DataLoaders for CIFAR-10."""
    bs = config.BATCH_SIZE if batch_size is None else batch_size
    nw = config.NUM_WORKERS if num_workers is None else num_workers

    train_transform, val_transform = get_transforms()

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True
    )
    return train_loader, val_loader
