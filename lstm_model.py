"""
LSTM model for health time series forecasting.
Built with PyTorch; uses a sliding-window dataset.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple
import copy


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class HealthSequenceDataset(Dataset):
    """
    Sliding-window dataset.
    Each sample: (window of seq_len days) → (next day's target value)
    """

    def __init__(
        self,
        data: np.ndarray,           # shape (n_days, n_features)
        target_idx: int,            # column index of the target variable
        seq_len: int = 14,
    ):
        self.seq_len = seq_len
        self.target_idx = target_idx
        self.X, self.y = self._build_sequences(data)

    def _build_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i : i + self.seq_len])          # input window
            y.append(data[i + self.seq_len, self.target_idx])  # next-step target
        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Stacked LSTM with dropout regularisation.
    Architecture:
        [seq_len, n_features] → LSTM(hidden) × n_layers → FC → [1]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden  = lstm_out[:, -1, :]          # take last timestep
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Forecaster Wrapper
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = ["steps", "heart_rate", "sleep_hours", "calories", "active_minutes"]


class LSTMForecaster:
    """
    Full training / inference pipeline for the LSTM.
    Handles scaling, dataset creation, training loop, and early stopping.
    """

    def __init__(
        self,
        seq_len: int = 14,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 10,
        device: Optional[str] = None,
    ):
        self.seq_len    = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.lr          = lr
        self.batch_size  = batch_size
        self.max_epochs  = max_epochs
        self.patience    = patience
        self.device      = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.scaler     = MinMaxScaler()
        self.model      = None
        self.target_idx = 0  # "steps" is first column
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []

    def _prep_data(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in FEATURE_COLS if c in df.columns]
        return df[cols].ffill().bfill().values

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "LSTMForecaster":
        X_train_raw = self._prep_data(train_df)
        X_val_raw   = self._prep_data(val_df)

        # Fit scaler on training data only — no data leakage
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_val_scaled   = self.scaler.transform(X_val_raw)

        train_ds = HealthSequenceDataset(X_train_scaled, self.target_idx, self.seq_len)
        val_ds   = HealthSequenceDataset(X_val_scaled,   self.target_idx, self.seq_len)

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        n_features = X_train_scaled.shape[1]
        self.model = LSTMModel(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        best_val_loss  = float("inf")
        best_weights   = None
        patience_count = 0

        for epoch in range(self.max_epochs):
            # ── Train ──
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_ds)

            # ── Validate ──
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    val_loss += criterion(pred, yb).item() * len(xb)
            val_loss /= len(val_ds)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            scheduler.step(val_loss)

            if epoch % 10 == 0:
                print(f"   Epoch {epoch:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = copy.deepcopy(self.model.state_dict())
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    print(f"   Early stopping at epoch {epoch}")
                    break

        self.model.load_state_dict(best_weights)
        print(f"✅ LSTM trained. Best val_loss: {best_val_loss:.4f}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on a dataframe (must have at least seq_len rows)."""
        if self.model is None:
            raise RuntimeError("Call .fit() first.")
        X = self._prep_data(df)
        X_scaled = self.scaler.transform(X)

        dataset = HealthSequenceDataset(X_scaled, self.target_idx, self.seq_len)
        loader  = DataLoader(dataset, batch_size=64, shuffle=False)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                out = self.model(xb).cpu().numpy()
                preds.append(out)

        preds_scaled = np.concatenate(preds).reshape(-1, 1)

        # Inverse-transform predictions back to original scale
        dummy = np.zeros((len(preds_scaled), self.scaler.n_features_in_))
        dummy[:, self.target_idx] = preds_scaled[:, 0]
        return self.scaler.inverse_transform(dummy)[:, self.target_idx]

    def save(self, path: str):
        torch.save({"model_state": self.model.state_dict(), "scaler": self.scaler}, path)

    def load(self, path: str, input_size: int):
        checkpoint = torch.load(path, map_location=self.device)
        self.scaler = checkpoint["scaler"]
        self.model  = LSTMModel(input_size=input_size).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
