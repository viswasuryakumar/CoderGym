"""
LSTM Sentiment Analysis

Mathematical Formulation:
- LSTM Cell:
    f_t = sigma(W_f . [h_{t-1}, x_t] + b_f)       (forget gate)
    i_t = sigma(W_i . [h_{t-1}, x_t] + b_i)       (input gate)
    c_t = f_t * c_{t-1} + i_t * tanh(W_c . [h_{t-1}, x_t] + b_c)
    o_t = sigma(W_o . [h_{t-1}, x_t] + b_o)       (output gate)
    h_t = o_t * tanh(c_t)
- Binary Cross-Entropy for sentiment classification
- Embedding layer maps word indices to dense vectors

This task trains an LSTM for binary sentiment classification (positive/negative)
on synthetically generated review-like sequences.
"""

import sys
import os
import json
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
        'task_name': 'rnn_sentiment_lstm',
        'description': 'LSTM for binary sentiment classification',
        'series': 'Recurrent Neural Networks',
        'level': 1,
        'vocab_size': 5000,
        'embedding_dim': 64,
        'hidden_dim': 128,
        'num_classes': 2,
        'model_type': 'lstm',
        'loss_type': 'binary_cross_entropy',
        'optimization': 'adam',
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the appropriate device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Synthetic sentiment dataset
# ---------------------------------------------------------------------------

# Word pools — positive and negative sentiment "words" (represented as token IDs)
_POS_TOKENS_RANGE = (100, 600)    # token IDs associated with positive sentiment
_NEG_TOKENS_RANGE = (600, 1100)   # token IDs associated with negative sentiment
_NEUTRAL_RANGE = (1100, 3000)     # neutral filler tokens
_PAD_IDX = 0

def _generate_sentiment_data(n_samples=4000, vocab_size=5000, seq_len=50,
                              pos_ratio=0.5):
    """
    Generate synthetic sentiment sequences.

    Positive reviews have more tokens from the positive range;
    negative reviews have more from the negative range.
    """
    X = np.full((n_samples, seq_len), _PAD_IDX, dtype=np.int64)
    y = np.zeros(n_samples, dtype=np.float32)

    n_pos = int(n_samples * pos_ratio)

    for i in range(n_samples):
        actual_len = np.random.randint(seq_len // 2, seq_len + 1)
        is_positive = i < n_pos

        if is_positive:
            # 60-70% positive tokens, rest neutral
            n_sent = int(actual_len * np.random.uniform(0.6, 0.7))
            sent_tokens = np.random.randint(
                _POS_TOKENS_RANGE[0], _POS_TOKENS_RANGE[1], n_sent)
            y[i] = 1.0
        else:
            n_sent = int(actual_len * np.random.uniform(0.6, 0.7))
            sent_tokens = np.random.randint(
                _NEG_TOKENS_RANGE[0], _NEG_TOKENS_RANGE[1], n_sent)
            y[i] = 0.0

        n_neutral = actual_len - n_sent
        neutral_tokens = np.random.randint(
            _NEUTRAL_RANGE[0], _NEUTRAL_RANGE[1], max(n_neutral, 0))

        tokens = np.concatenate([sent_tokens, neutral_tokens])
        np.random.shuffle(tokens)
        X[i, :len(tokens)] = tokens[:seq_len]

    # Shuffle
    perm = np.random.permutation(n_samples)
    return X[perm], y[perm]


def make_dataloaders(batch_size=64, n_samples=4000, seq_len=50,
                     train_ratio=0.8):
    """
    Create train/val dataloaders for sentiment classification.

    Returns:
        train_loader, val_loader
    """
    X, y = _generate_sentiment_data(n_samples=n_samples, seq_len=seq_len)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    n_train = int(n_samples * train_ratio)
    train_ds = TensorDataset(X_tensor[:n_train], y_tensor[:n_train])
    val_ds = TensorDataset(X_tensor[n_train:], y_tensor[n_train:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SentimentLSTM(nn.Module):
    """
    Embedding → LSTM → FC for binary sentiment classification.
    """
    def __init__(self, vocab_size=5000, embedding_dim=64, hidden_dim=128,
                 num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len) of token IDs
        emb = self.embedding(x)            # (batch, seq_len, emb_dim)
        lstm_out, (h_n, c_n) = self.lstm(emb)
        # Use last hidden state
        out = self.dropout(h_n[-1])        # (batch, hidden_dim)
        logit = self.fc(out).squeeze(-1)   # (batch,)
        return logit


def build_model(device=None, vocab_size=5000, embedding_dim=64,
                hidden_dim=128):
    """Build and return the SentimentLSTM model."""
    if device is None:
        device = get_device()
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    return model


# ---------------------------------------------------------------------------
# Train / Evaluate / Predict / Save
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, epochs=15, lr=0.001, device=None):
    """
    Train the LSTM sentiment classifier.

    Returns dict with loss_history, val_loss_history, val_acc_history.
    """
    if device is None:
        device = get_device()
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history, val_loss_history, val_acc_history = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for LSTM stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        val_metrics = evaluate(model, val_loader, device=device)
        val_loss_history.append(val_metrics['loss'])
        val_acc_history.append(val_metrics['accuracy'])

        print(f"Epoch [{epoch+1}/{epochs}]  "
              f"train_loss={avg_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_acc={val_metrics['accuracy']:.2f}%  "
              f"val_f1={val_metrics['f1']:.4f}")

    return {
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'val_acc_history': val_acc_history,
    }


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
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * y_batch.size(0)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            tp += ((preds == 1) & (y_batch == 1)).sum().item()
            fp += ((preds == 1) & (y_batch == 0)).sum().item()
            fn += ((preds == 0) & (y_batch == 1)).sum().item()
            tn += ((preds == 0) & (y_batch == 0)).sum().item()
            total += y_batch.size(0)

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
    """Return binary predictions (0 or 1) for input token sequences."""
    if device is None:
        device = get_device()
    model.eval()
    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            X = torch.LongTensor(X)
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
    print("Task: LSTM Sentiment Analysis")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print(f"Metadata: {get_task_metadata()}")

    # Data
    train_loader, val_loader = make_dataloaders(
        batch_size=64, n_samples=4000, seq_len=50)

    # Model
    model = build_model(device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Train
    history = train(model, train_loader, val_loader,
                    epochs=15, lr=0.001, device=device)

    # Final evaluation
    train_metrics = evaluate(model, train_loader, device=device)
    val_metrics = evaluate(model, val_loader, device=device)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train — loss: {train_metrics['loss']:.4f}, "
          f"acc: {train_metrics['accuracy']:.2f}%, "
          f"f1: {train_metrics['f1']:.4f}")
    print(f"Val   — loss: {val_metrics['loss']:.4f}, "
          f"acc: {val_metrics['accuracy']:.2f}%, "
          f"f1: {val_metrics['f1']:.4f}")

    # Save
    save_artifacts(model, history, val_metrics)

    # Assertions
    exit_code = 0
    if val_metrics['accuracy'] < 65.0:
        print(f"[FAIL] Validation accuracy {val_metrics['accuracy']:.2f}% < 65%")
        exit_code = 1
    else:
        print(f"[PASS] Validation accuracy: {val_metrics['accuracy']:.2f}%")

    if val_metrics['f1'] < 0.55:
        print(f"[FAIL] Validation F1 {val_metrics['f1']:.4f} < 0.55")
        exit_code = 1
    else:
        print(f"[PASS] Validation F1: {val_metrics['f1']:.4f}")

    sys.exit(exit_code)
