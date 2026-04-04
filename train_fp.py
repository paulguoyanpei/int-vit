"""Fine-tune pretrained DeiT-Tiny on CIFAR-10 (FP32 baseline)."""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from model import get_model
from data import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_dataloaders()

    # Model
    model = get_model(pretrained=True).to(device)
    print(f"Model: {config.MODEL_NAME}, params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_acc = 0.0

    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            path = os.path.join(config.CHECKPOINT_DIR, "deit_tiny_cifar10_best.pth")
            torch.save(model.state_dict(), path)
            print(f"  -> Saved best model (acc={best_acc:.4f})")

    print(f"\nBest validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
