import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, dataset, lookback=10, forecast_days=5):
        """
        dataset: 4-tuple [X, base_prices, D, T]
            X: np.array of shape (num_samples, lookback+forecast_days, 5) - percent changes
            base_prices: np.array of shape (num_samples,) - last price in lookback window
            D: np.array of shape (num_samples, lookback+forecast_days) - dates
            T: np.array of shape (num_samples,) - ticker symbols
        """
        self.X, self.base_prices, self.D, self.T = dataset
        self.lookback = lookback
        self.forecast_days = forecast_days

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]

        x = sample[:self.lookback]

        y = sample[self.lookback:, 3]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        base_price = torch.tensor(self.base_prices[idx], dtype=torch.float32)

        return x, y, base_price