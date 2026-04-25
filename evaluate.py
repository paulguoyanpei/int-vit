"""Evaluate FP32 model vs fixed-point model at various Q values."""

import os
import time
import torch
import config
from model import get_model
from data import get_dataloaders
from fixed_point_vit import FixedPointViT


@torch.no_grad()
def evaluate_fp32(model, loader, device):
    """Evaluate floating-point model."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total


def evaluate_fixed_point(fp_model, loader, Q, max_batches=None):
    """Evaluate fixed-point model on CPU (int64 ops)."""
    print(f"\n--- Evaluating fixed-point with Q={Q} ---")
    fxp_vit = FixedPointViT(fp_model, Q)

    correct, total = 0, 0
    t0 = time.time()
    for i, (images, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        logits = fxp_vit(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}, running acc: {correct/total:.4f}")
    elapsed = time.time() - t0

    acc = correct / total
    print(f"  Q={Q}: accuracy={acc:.4f}, samples={total}, time={elapsed:.1f}s")
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    _, val_loader = get_dataloaders()

    # Load fine-tuned FP32 model
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "deit_tiny_cifar10_best.pth")
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.to(device)

    # FP32 baseline
    print("=== FP32 Baseline ===")
    fp32_acc = evaluate_fp32(model, val_loader, device)
    print(f"FP32 accuracy: {fp32_acc:.4f}")

    # Move model to CPU for fixed-point conversion (int64 on CPU)
    model.cpu()
    model.eval()

    # Sweep Q values
    results = {"fp32": fp32_acc}
    acc = evaluate_fixed_point(model, val_loader, config.DEFAULT_Q, max_batches=None)
    # for Q in config.Q_VALUES:
    #     results[f"Q={Q}"] = acc

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Method':<15} {'Accuracy':>10}")
    print("-" * 25)
    for name, acc in results.items():
        drop = (fp32_acc - acc) * 100 if name != "fp32" else 0.0
        print(f"{name:<15} {acc:>9.4f}  (drop: {drop:+.2f}%)")


if __name__ == "__main__":
    main()
