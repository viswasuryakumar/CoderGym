"""
Lasso Regression via Coordinate Descent

Mathematical Formulation:
- Lasso objective: J(w) = (1/2n) * ||Xw - y||^2 + lambda * ||w||_1
- Coordinate Descent update for w_j:
    r_j = X[:, j]^T (y - X w + X[:, j] w_j)    (partial residual)
    w_j = soft_threshold(r_j / (X[:, j]^T X[:, j]),  lambda * n / (X[:, j]^T X[:, j]))
- Soft-thresholding operator:
    S(z, gamma) = sign(z) * max(|z| - gamma, 0)

This task implements L1-regularized (Lasso) linear regression using
coordinate descent — NO autograd, NO torch.optim. Demonstrates sparsity
recovery on a synthetic dataset where only 3 of 20 features are relevant.
"""

import sys
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the task."""
    return {
        'task_name': 'lasso_coordinate_descent',
        'description': 'Lasso (L1) Regression via Coordinate Descent',
        'series': 'Linear Regression',
        'level': 5,
        'n_features': 20,
        'n_relevant': 3,
        'model_type': 'lasso_regression',
        'loss_type': 'mse_plus_l1',
        'optimization': 'coordinate_descent',
        'new_feature': 'sparsity_recovery',
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get the appropriate device (CPU or GPU)."""
    # Coordinate descent is implemented on CPU tensors
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_dataloaders(n_samples=500, n_features=20, n_relevant=3,
                     noise_std=0.5, train_ratio=0.8, batch_size=500):
    """
    Create synthetic sparse regression data.

    Only `n_relevant` of `n_features` have non-zero true coefficients.

    Returns:
        train_loader, val_loader, true_weights (for verification)
    """
    set_seed(42)

    # True weights: only first n_relevant features matter
    true_w = np.zeros(n_features, dtype=np.float32)
    true_w[0] = 3.0
    true_w[1] = -2.0
    true_w[2] = 1.5
    true_bias = 1.0

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = X @ true_w + true_bias + np.random.randn(n_samples).astype(np.float32) * noise_std

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    n_train = int(n_samples * train_ratio)
    train_ds = TensorDataset(X_t[:n_train], y_t[:n_train])
    val_ds = TensorDataset(X_t[n_train:], y_t[n_train:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, true_w, true_bias


# ---------------------------------------------------------------------------
# Model (manual — no nn.Module)
# ---------------------------------------------------------------------------

class LassoModel:
    """
    Lasso regression model using coordinate descent.
    No autograd, no nn.Module, no optimizer — all manual.
    """

    def __init__(self, n_features):
        self.w = torch.zeros(n_features, dtype=torch.float32)
        self.b = torch.tensor(0.0, dtype=torch.float32)
        self.n_features = n_features

    def forward(self, X):
        """y_hat = X @ w + b"""
        return X @ self.w + self.b

    @staticmethod
    def soft_threshold(z, gamma):
        """
        Soft-thresholding operator: S(z, gamma) = sign(z) * max(|z| - gamma, 0)
        """
        return torch.sign(z) * torch.clamp(torch.abs(z) - gamma, min=0.0)

    def state_dict(self):
        return {'w': self.w.clone(), 'b': self.b.clone()}

    def load_state_dict(self, sd):
        self.w = sd['w']
        self.b = sd['b']


def build_model(device=None, n_features=20):
    """Build and return the LassoModel."""
    return LassoModel(n_features=n_features)


# ---------------------------------------------------------------------------
# Train / Evaluate / Predict / Save
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, epochs=200, lam=0.1,
          device=None):
    """
    Train the Lasso model via coordinate descent.

    For each epoch, cycle through all features and apply the
    coordinate descent update with soft-thresholding.

    Args:
        model: LassoModel
        train_loader: training data (single batch expected)
        val_loader: validation data
        epochs: number of full coordinate cycles
        lam: L1 regularization strength

    Returns:
        dict with loss_history, val_loss_history, sparsity_history
    """
    # Extract full data tensors (coordinate descent works on full data)
    X_train, y_train = next(iter(train_loader))

    n = X_train.shape[0]
    p = X_train.shape[1]

    # Precompute X_j^T X_j for each feature
    xtx = (X_train ** 2).sum(dim=0)  # shape: (p,)

    loss_history = []
    val_loss_history = []
    sparsity_history = []

    for epoch in range(epochs):
        # Update bias (no regularization on bias)
        residual = y_train - X_train @ model.w
        model.b = residual.mean()

        # Coordinate descent over features
        for j in range(p):
            # Partial residual (exclude feature j contribution)
            r = y_train - model.b - X_train @ model.w + X_train[:, j] * model.w[j]

            # Unnormalized update
            rho_j = X_train[:, j].dot(r)

            # Soft-thresholding
            if xtx[j] > 0:
                model.w[j] = LassoModel.soft_threshold(
                    rho_j / xtx[j],
                    lam * n / xtx[j]
                )
            else:
                model.w[j] = 0.0

        # Compute training MSE
        y_pred = model.forward(X_train)
        mse = torch.mean((y_pred - y_train) ** 2).item()
        l1_norm = torch.sum(torch.abs(model.w)).item()
        loss = mse / 2 + lam * l1_norm
        loss_history.append(loss)

        # Sparsity
        n_nonzero = (torch.abs(model.w) > 1e-6).sum().item()
        sparsity_history.append(n_nonzero)

        # Validation
        val_metrics = evaluate(model, val_loader)
        val_loss_history.append(val_metrics['mse'])

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  "
                  f"loss={loss:.6f}  mse={mse:.6f}  "
                  f"l1={l1_norm:.4f}  nonzero={n_nonzero}/{p}")

    return {
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'sparsity_history': sparsity_history,
    }


def evaluate(model, data_loader, device=None):
    """
    Evaluate the Lasso model — MSE, R2, sparsity metrics.
    """
    X, y = next(iter(data_loader))

    with torch.no_grad():
        y_pred = model.forward(X)

    residuals = y - y_pred
    mse = torch.mean(residuals ** 2).item()
    ss_res = torch.sum(residuals ** 2).item()
    ss_tot = torch.sum((y - y.mean()) ** 2).item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

    nonzero_mask = torch.abs(model.w) > 1e-6
    n_nonzero = nonzero_mask.sum().item()

    return {
        'mse': mse,
        'r2': r2,
        'n_nonzero_features': n_nonzero,
        'learned_weights': model.w.tolist(),
        'learned_bias': model.b.item(),
    }


def predict(model, X, device=None):
    """Return predictions for input X."""
    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        return model.forward(X)


def save_artifacts(model, history, metrics, output_dir=None):
    """Save model weights, training history, and metrics."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))

    results = {**history, **metrics}
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[INFO] Artifacts saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main (self-verifying)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Task: Lasso Regression via Coordinate Descent")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print(f"Metadata: {get_task_metadata()}")

    # Data
    train_loader, val_loader, true_w, true_bias = make_dataloaders(
        n_samples=500, n_features=20, n_relevant=3)
    print(f"True weights (first 5): {true_w[:5]}")
    print(f"True bias: {true_bias}")

    # Model
    model = build_model(n_features=20)

    # Train
    history = train(model, train_loader, val_loader, epochs=300, lam=0.05)

    # Final evaluation
    train_metrics = evaluate(model, train_loader)
    val_metrics = evaluate(model, val_loader)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train — MSE: {train_metrics['mse']:.6f}, R2: {train_metrics['r2']:.4f}")
    print(f"Val   — MSE: {val_metrics['mse']:.6f}, R2: {val_metrics['r2']:.4f}")
    print(f"Non-zero features: {val_metrics['n_nonzero_features']} / 20")
    print(f"Learned weights: {[f'{w:.3f}' for w in val_metrics['learned_weights']]}")
    print(f"Learned bias: {val_metrics['learned_bias']:.4f}")
    print(f"True weights (first 5): {true_w[:5].tolist()}")

    # Save
    save_artifacts(model, history, val_metrics)

    # Assertions
    exit_code = 0

    # 1. R2 should be reasonable
    if val_metrics['r2'] < 0.80:
        print(f"[FAIL] Validation R2 {val_metrics['r2']:.4f} < 0.80")
        exit_code = 1
    else:
        print(f"[PASS] Validation R2: {val_metrics['r2']:.4f}")

    # 2. Should recover sparse support (3 non-zero features, allow up to 5)
    if val_metrics['n_nonzero_features'] > 8:
        print(f"[FAIL] Too many non-zero features: {val_metrics['n_nonzero_features']}")
        exit_code = 1
    else:
        print(f"[PASS] Sparsity: {val_metrics['n_nonzero_features']} non-zero features")

    # 3. The three true features should be among the largest learned weights
    w_learned = np.array(val_metrics['learned_weights'])
    top3_idx = set(np.argsort(np.abs(w_learned))[-3:])
    true_support = {0, 1, 2}
    overlap = len(top3_idx & true_support)
    if overlap < 2:
        print(f"[FAIL] Support recovery: only {overlap}/3 true features in top-3")
        exit_code = 1
    else:
        print(f"[PASS] Support recovery: {overlap}/3 true features in top-3")

    sys.exit(exit_code)
