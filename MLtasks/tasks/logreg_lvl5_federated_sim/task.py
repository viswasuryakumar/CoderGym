"""
Federated Learning Simulation with Logistic Regression

Mathematical Formulation:
- Federated Averaging (FedAvg) by McMahan et al., 2017:
    1. Server broadcasts global model w_t to K clients
    2. Each client k trains locally: w_k = w_t - lr * grad(L_k(w_t))
    3. Server aggregates: w_{t+1} = sum(n_k / n) * w_k  (weighted average)
- Local loss for client k: L_k(w) = -(1/n_k) sum [y log(sigma(Xw)) + (1-y) log(1-sigma(Xw))]
- Comparison with centralized training as baseline

This task simulates federated learning across 5 clients with non-IID data
splits, using logistic regression. It demonstrates the FedAvg algorithm and
compares against a centralized baseline.
"""

import sys
import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the task."""
    return {
        'task_name': 'federated_learning_simulation',
        'description': 'Federated Averaging (FedAvg) simulation with logistic regression',
        'series': 'Logistic Regression',
        'level': 5,
        'n_clients': 5,
        'input_dim': 10,
        'num_classes': 2,
        'model_type': 'logistic_regression',
        'loss_type': 'binary_cross_entropy',
        'optimization': 'fedavg',
        'new_feature': 'federated_learning',
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get the appropriate device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _generate_non_iid_data(n_total=2000, n_features=10, n_clients=5):
    """
    Generate non-IID binary classification data split across clients.

    Each client gets data with different class ratios and slightly
    different feature distributions to simulate non-IID settings.
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_total, n_features=n_features, n_informative=6,
        n_redundant=2, n_clusters_per_class=2, random_state=42,
        flip_y=0.05)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Non-IID split: sort by label, then distribute unevenly
    sorted_idx = np.argsort(y)
    X, y = X[sorted_idx], y[sorted_idx]

    # Give clients imbalanced shares
    client_data = []
    n_per_client = n_total // n_clients
    for k in range(n_clients):
        # Each client gets data from a shifted window (overlapping, non-IID)
        start = int(k * n_total * 0.7 / n_clients)
        end = start + n_per_client
        end = min(end, n_total)
        idx = np.arange(start, end)
        # Add some random samples to mix slightly
        n_random = n_per_client // 5
        random_idx = np.random.choice(n_total, n_random, replace=False)
        idx = np.unique(np.concatenate([idx, random_idx]))
        client_data.append((X[idx], y[idx]))

    return client_data, X, y


def make_dataloaders(batch_size=64, n_total=2000, n_features=10,
                     n_clients=5, val_ratio=0.2):
    """
    Create per-client dataloaders + a global validation loader.

    Returns:
        client_loaders: list of DataLoaders (one per client)
        val_loader: global validation DataLoader
        centralized_train_loader: all training data combined
    """
    try:
        client_data, X_all, y_all = _generate_non_iid_data(
            n_total, n_features, n_clients)
    except ImportError:
        # Fallback without sklearn
        X_all = np.random.randn(n_total, n_features).astype(np.float32)
        w_true = np.random.randn(n_features).astype(np.float32)
        logits = X_all @ w_true
        y_all = (logits > 0).astype(np.float32)
        # Equal split
        client_data = []
        per_client = n_total // n_clients
        for k in range(n_clients):
            s = k * per_client
            e = s + per_client
            client_data.append((X_all[s:e], y_all[s:e]))

    # Hold out val_ratio of each client's data for a global val set
    client_loaders = []
    val_X_list, val_y_list = [], []
    train_X_list, train_y_list = [], []

    for X_k, y_k in client_data:
        n_k = len(y_k)
        n_val_k = max(int(n_k * val_ratio), 1)
        perm = np.random.permutation(n_k)
        X_k, y_k = X_k[perm], y_k[perm]

        X_train_k = torch.from_numpy(X_k[n_val_k:])
        y_train_k = torch.from_numpy(y_k[n_val_k:])
        X_val_k = torch.from_numpy(X_k[:n_val_k])
        y_val_k = torch.from_numpy(y_k[:n_val_k])

        train_X_list.append(X_train_k)
        train_y_list.append(y_train_k)
        val_X_list.append(X_val_k)
        val_y_list.append(y_val_k)

        train_ds = TensorDataset(X_train_k, y_train_k)
        client_loaders.append(
            DataLoader(train_ds, batch_size=batch_size, shuffle=True))

    # Global validation
    X_val = torch.cat(val_X_list)
    y_val = torch.cat(val_y_list)
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # Centralized training (all clients combined)
    X_train_all = torch.cat(train_X_list)
    y_train_all = torch.cat(train_y_list)
    centralized_train_loader = DataLoader(
        TensorDataset(X_train_all, y_train_all),
        batch_size=batch_size, shuffle=True)

    return client_loaders, val_loader, centralized_train_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LogisticRegressionModel(nn.Module):
    """Simple logistic regression: sigmoid(Xw + b)."""
    def __init__(self, input_dim=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def build_model(device=None, input_dim=10):
    """Build and return the logistic regression model."""
    if device is None:
        device = get_device()
    return LogisticRegressionModel(input_dim=input_dim).to(device)


# ---------------------------------------------------------------------------
# Federated helpers
# ---------------------------------------------------------------------------

def _train_local(model, data_loader, local_epochs=5, lr=0.01, device=None):
    """Train a model locally on one client's data."""
    if device is None:
        device = get_device()
    model.to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for _ in range(local_epochs):
        for X_b, y_b in data_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss = criterion(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def _fedavg_aggregate(global_model, client_models, client_sizes):
    """
    Federated Averaging: weighted average of client model parameters.
    w_global = sum(n_k / n_total) * w_k
    """
    total = sum(client_sizes)
    global_sd = global_model.state_dict()

    for key in global_sd:
        global_sd[key] = torch.zeros_like(global_sd[key], dtype=torch.float32)
        for client_model, n_k in zip(client_models, client_sizes):
            client_sd = client_model.state_dict()
            global_sd[key] += (n_k / total) * client_sd[key].float()

    global_model.load_state_dict(global_sd)
    return global_model


# ---------------------------------------------------------------------------
# Train / Evaluate / Predict / Save
# ---------------------------------------------------------------------------

def train(model, client_loaders, val_loader, rounds=20, local_epochs=5,
          lr=0.01, device=None):
    """
    Federated training loop using FedAvg.

    Args:
        model: global model
        client_loaders: list of per-client DataLoaders
        val_loader: global validation DataLoader
        rounds: number of communication rounds
        local_epochs: local training epochs per round
        lr: local learning rate

    Returns:
        dict with loss_history, val_loss_history, val_acc_history
    """
    if device is None:
        device = get_device()
    model.to(device)

    loss_history, val_loss_history, val_acc_history = [], [], []

    for rnd in range(rounds):
        client_models = []
        client_sizes = []

        for k, loader in enumerate(client_loaders):
            # Clone global model for this client
            local_model = copy.deepcopy(model)
            local_model = _train_local(
                local_model, loader, local_epochs=local_epochs,
                lr=lr, device=device)
            client_models.append(local_model)
            client_sizes.append(len(loader.dataset))

        # Aggregate via FedAvg
        model = _fedavg_aggregate(model, client_models, client_sizes)

        # Evaluate
        val_metrics = evaluate(model, val_loader, device=device)
        val_loss_history.append(val_metrics['loss'])
        val_acc_history.append(val_metrics['accuracy'])

        print(f"Round [{rnd+1}/{rounds}]  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_acc={val_metrics['accuracy']:.2f}%")

    return {
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'val_acc_history': val_acc_history,
    }


def train_centralized(model, train_loader, val_loader, epochs=20,
                      lr=0.01, device=None):
    """
    Train centralized baseline for comparison.
    """
    if device is None:
        device = get_device()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss = criterion(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate(model, data_loader, device=None):
    """
    Evaluate — computes loss, accuracy, precision, recall, F1.
    """
    if device is None:
        device = get_device()
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    tp = fp = fn = tn = 0
    total = 0

    with torch.no_grad():
        for X_b, y_b in data_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss = criterion(logits, y_b)
            total_loss += loss.item() * y_b.size(0)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            tp += ((preds == 1) & (y_b == 1)).sum().item()
            fp += ((preds == 1) & (y_b == 0)).sum().item()
            fn += ((preds == 0) & (y_b == 1)).sum().item()
            tn += ((preds == 0) & (y_b == 0)).sum().item()
            total += y_b.size(0)

    accuracy = 100.0 * (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'loss': total_loss / max(total, 1),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def predict(model, X, device=None):
    """Return binary predictions (0 or 1)."""
    if device is None:
        device = get_device()
    model.eval()
    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        X = X.to(device)
        logits = model(X)
        preds = (torch.sigmoid(logits) >= 0.5).long()
    return preds.cpu()


def save_artifacts(model, history, metrics, output_dir=None):
    """Save model weights, training history, and metrics."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({**history, **metrics}, f, indent=2, default=str)
    print(f"[INFO] Artifacts saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main (self-verifying)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Task: Federated Learning Simulation (FedAvg)")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print(f"Metadata: {get_task_metadata()}")

    # Data
    client_loaders, val_loader, centralized_train_loader = make_dataloaders(
        batch_size=64, n_total=2000, n_features=10, n_clients=5)
    print(f"Number of clients: {len(client_loaders)}")
    for k, loader in enumerate(client_loaders):
        print(f"  Client {k}: {len(loader.dataset)} samples")

    # ---- Federated Training ----
    print("\n--- Federated Training (FedAvg) ---")
    fed_model = build_model(device=device, input_dim=10)
    fed_history = train(fed_model, client_loaders, val_loader,
                        rounds=25, local_epochs=5, lr=0.01, device=device)
    fed_val_metrics = evaluate(fed_model, val_loader, device=device)

    # ---- Centralized Baseline ----
    print("\n--- Centralized Baseline ---")
    cent_model = build_model(device=device, input_dim=10)
    cent_model = train_centralized(
        cent_model, centralized_train_loader, val_loader,
        epochs=50, lr=0.01, device=device)
    cent_val_metrics = evaluate(cent_model, val_loader, device=device)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Federated  — acc: {fed_val_metrics['accuracy']:.2f}%, "
          f"f1: {fed_val_metrics['f1']:.4f}")
    print(f"Centralized — acc: {cent_val_metrics['accuracy']:.2f}%, "
          f"f1: {cent_val_metrics['f1']:.4f}")
    gap = abs(fed_val_metrics['accuracy'] - cent_val_metrics['accuracy'])
    print(f"Accuracy gap: {gap:.2f}%")

    # Save
    combined_metrics = {
        'federated': fed_val_metrics,
        'centralized': cent_val_metrics,
        'accuracy_gap': gap,
    }
    save_artifacts(fed_model, fed_history, combined_metrics)

    # Assertions
    exit_code = 0

    if fed_val_metrics['accuracy'] < 70.0:
        print(f"[FAIL] Federated accuracy {fed_val_metrics['accuracy']:.2f}% < 70%")
        exit_code = 1
    else:
        print(f"[PASS] Federated accuracy: {fed_val_metrics['accuracy']:.2f}%")

    if gap > 15.0:
        print(f"[FAIL] Gap vs centralized: {gap:.2f}% > 15%")
        exit_code = 1
    else:
        print(f"[PASS] Gap vs centralized: {gap:.2f}%")

    sys.exit(exit_code)
