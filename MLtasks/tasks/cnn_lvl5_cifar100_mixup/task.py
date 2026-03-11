"""
CNN with Mixup Data Augmentation on CIFAR-100

Mathematical Formulation:
- Mixup: x_mix = lambda * x_i + (1 - lambda) * x_j
          y_mix = lambda * y_i + (1 - lambda) * y_j
  where lambda ~ Beta(alpha, alpha)
- Cross-Entropy Loss with soft labels for mixup training
- Cosine Annealing LR schedule: lr_t = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*t/T))

This task trains a small ResNet-style CNN on CIFAR-100 (100 classes) using Mixup
augmentation to improve generalization. The model is self-verifiable.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the task."""
    return {
        'task_name': 'cnn_cifar100_mixup',
        'description': 'CNN on CIFAR-100 with Mixup data augmentation',
        'series': 'Convolutional Neural Networks',
        'level': 5,
        'input_shape': [3, 32, 32],
        'num_classes': 100,
        'model_type': 'resnet_small',
        'loss_type': 'cross_entropy_mixup',
        'optimization': 'adam_cosine_annealing',
        'new_feature': 'mixup_augmentation'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device():
    """Get the appropriate device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _generate_synthetic_cifar100(n_train=5000, n_val=1000, num_classes=100):
    """
    Generate a synthetic dataset mimicking CIFAR-100 structure.
    Uses structured random patterns so a CNN can learn non-trivial features.
    Each class gets a unique base pattern (low-freq spatial + color bias).
    """
    def _make_samples(n, num_classes):
        imgs = []
        labels = []
        per_class = max(n // num_classes, 1)
        for c in range(num_classes):
            rng = np.random.RandomState(c)
            # Create a class-specific base pattern
            base = rng.randn(3, 32, 32).astype(np.float32) * 0.3
            # Smooth it to create low-frequency structure
            for ch in range(3):
                from scipy.ndimage import gaussian_filter  # noqa: lazy import
                base[ch] = gaussian_filter(base[ch], sigma=4)
            for _ in range(per_class):
                noise = np.random.randn(3, 32, 32).astype(np.float32) * 0.15
                img = base + noise
                imgs.append(img)
                labels.append(c)
        imgs = np.stack(imgs[:n])
        labels = np.array(labels[:n], dtype=np.int64)
        # Shuffle
        perm = np.random.permutation(n)
        return imgs[perm], labels[perm]

    X_train, y_train = _make_samples(n_train, num_classes)
    X_val, y_val = _make_samples(n_val, num_classes)
    return (
        torch.from_numpy(X_train), torch.from_numpy(y_train),
        torch.from_numpy(X_val), torch.from_numpy(y_val),
    )


def make_dataloaders(batch_size=128, n_train=5000, n_val=1000, num_classes=100):
    """
    Create CIFAR-100-like dataloaders.

    Tries to load real CIFAR-100 via torchvision; falls back to synthetic
    data if torchvision is unavailable or download fails.

    Returns:
        train_loader, val_loader
    """
    try:
        import torchvision
        import torchvision.transforms as T

        transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
        ])
        transform_val = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
        ])

        data_dir = os.path.join(OUTPUT_DIR, 'data')

        train_set = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True,
            transform=transform_train)
        val_set = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True,
            transform=transform_val)

        # Use subsets for faster training in CI / homework verification
        if n_train < len(train_set):
            train_set = torch.utils.data.Subset(
                train_set, list(range(n_train)))
        if n_val < len(val_set):
            val_set = torch.utils.data.Subset(
                val_set, list(range(n_val)))

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=batch_size,
                                shuffle=False, num_workers=0)
        print("[INFO] Loaded real CIFAR-100 dataset.")
        return train_loader, val_loader

    except Exception as e:
        print(f"[WARN] Could not load CIFAR-100 ({e}); using synthetic data.")
        X_tr, y_tr, X_va, y_va = _generate_synthetic_cifar100(
            n_train, n_val, num_classes)
        train_loader = DataLoader(
            TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
        return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Basic residual block."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class SmallResNet(nn.Module):
    """Compact ResNet for CIFAR-100."""
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    @staticmethod
    def _make_layer(in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def build_model(device=None, num_classes=100):
    """Build and return the SmallResNet model."""
    if device is None:
        device = get_device()
    model = SmallResNet(num_classes=num_classes).to(device)
    return model


# ---------------------------------------------------------------------------
# Mixup helpers
# ---------------------------------------------------------------------------

def mixup_data(x, y, alpha=1.0):
    """
    Apply Mixup augmentation.

    x_mix = lam * x_i + (1 - lam) * x_j
    y returns both targets + lam for soft cross-entropy.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss: lam * L(pred, y_a) + (1-lam) * L(pred, y_b)."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# Train / Evaluate / Predict / Save
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, epochs=10, lr=0.001, alpha=1.0,
          device=None):
    """
    Train with Mixup augmentation and cosine annealing LR.

    Returns dict with loss_history, val_loss_history, val_acc_history.
    """
    if device is None:
        device = get_device()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history, val_loss_history, val_acc_history = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Mixup
            mixed_x, y_a, y_b, lam = mixup_data(X_batch, y_batch, alpha)
            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        # Validation
        val_metrics = evaluate(model, val_loader, device=device)
        val_loss_history.append(val_metrics['val_loss'])
        val_acc_history.append(val_metrics['top1_accuracy'])

        print(f"Epoch [{epoch+1}/{epochs}]  "
              f"train_loss={avg_loss:.4f}  "
              f"val_loss={val_metrics['val_loss']:.4f}  "
              f"top1={val_metrics['top1_accuracy']:.2f}%  "
              f"top5={val_metrics['top5_accuracy']:.2f}%  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

    return {
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'val_acc_history': val_acc_history,
    }


def evaluate(model, data_loader, device=None):
    """
    Evaluate the model — computes loss, top-1 accuracy, top-5 accuracy.
    """
    if device is None:
        device = get_device()
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * y_batch.size(0)

            # Top-1
            _, pred = outputs.max(1)
            correct_top1 += pred.eq(y_batch).sum().item()

            # Top-5
            _, pred5 = outputs.topk(5, 1, True, True)
            correct_top5 += pred5.eq(
                y_batch.unsqueeze(1).expand_as(pred5)
            ).any(1).sum().item()

            total += y_batch.size(0)

    return {
        'val_loss': total_loss / max(total, 1),
        'top1_accuracy': 100.0 * correct_top1 / max(total, 1),
        'top5_accuracy': 100.0 * correct_top5 / max(total, 1),
    }


def predict(model, X, device=None):
    """Return class predictions for input tensor X."""
    if device is None:
        device = get_device()
    model.eval()
    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        X = X.to(device)
        outputs = model(X)
        _, preds = outputs.max(1)
    return preds.cpu()


def save_artifacts(model, history, metrics, output_dir=None):
    """Save model weights, training history, and metrics."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(),
               os.path.join(output_dir, 'model.pt'))

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({k: v if not isinstance(v, list) else v
                    for k, v in {**history, **metrics}.items()}, f, indent=2)

    print(f"[INFO] Artifacts saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main (self-verifying)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Task: CNN with Mixup on CIFAR-100")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print(f"Metadata: {get_task_metadata()}")

    # Data
    train_loader, val_loader = make_dataloaders(
        batch_size=128, n_train=2000, n_val=500)

    # Model
    model = build_model(device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Train
    history = train(model, train_loader, val_loader,
                    epochs=3, lr=0.001, alpha=1.0, device=device)

    # Final evaluation
    train_metrics = evaluate(model, train_loader, device=device)
    val_metrics = evaluate(model, val_loader, device=device)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train — loss: {train_metrics['val_loss']:.4f}, "
          f"top1: {train_metrics['top1_accuracy']:.2f}%, "
          f"top5: {train_metrics['top5_accuracy']:.2f}%")
    print(f"Val   — loss: {val_metrics['val_loss']:.4f}, "
          f"top1: {val_metrics['top1_accuracy']:.2f}%, "
          f"top5: {val_metrics['top5_accuracy']:.2f}%")

    # Save
    save_artifacts(model, history, val_metrics)

    # Assertions
    exit_code = 0
    if val_metrics['top1_accuracy'] < 5.0:
        print("[FAIL] Top-1 accuracy too low (< 5%)")
        exit_code = 1
    else:
        print(f"[PASS] Top-1 accuracy: {val_metrics['top1_accuracy']:.2f}%")

    if val_metrics['top5_accuracy'] < 15.0:
        print("[FAIL] Top-5 accuracy too low (< 15%)")
        exit_code = 1
    else:
        print(f"[PASS] Top-5 accuracy: {val_metrics['top5_accuracy']:.2f}%")

    sys.exit(exit_code)
