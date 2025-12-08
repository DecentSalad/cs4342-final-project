import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class StockDataset(Dataset):
    def __init__(self, ohlcv, lookback=10, forecast_days=5):
        self.ohlcv = ohlcv
        self.lookback = lookback
        self.forecast_days = forecast_days

        self.mean = ohlcv.mean(axis=0)
        self.std = ohlcv.std(axis=0)

        self.close_prices = self.ohlcv[:, 3]
        self.X, self.y = self._create_sequences()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

    def _create_sequences(self):
        X, y = [], []
        for i in range(len(self.close_prices) - self.lookback - self.forecast_days):
            features = self.ohlcv_norm[i:i + self.lookback]

            current_price = self.ohlcv[i + self.lookback - 1, 3]
            future_closes = self.ohlcv[i + self.lookback:i + self.lookback + self.forecast_days, 3]
            pct_changes = (future_closes - current_price) / current_price

            X.append(features)
            y.append(pct_changes)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def denormalize(self, normalized_prices):
        return normalized_prices * self.std[3] + self.mean[3]