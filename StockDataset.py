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
        self.ohlcv_norm = (ohlcv - self.mean) / (self.std + 1e-8)

        self.close_prices = self.ohlcv_norm[:, 3]
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
            future_closes = self.close_prices[i + self.lookback:i + self.lookback + self.forecast_days]
            X.append(features)
            y.append(future_closes)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)