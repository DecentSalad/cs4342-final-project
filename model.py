import numpy as np
import torch
from torch.utils.data import DataLoader

from StockReturnsDataset import StockReturnsDataset
from yfinance_test import get_daily_returns

def prepare_data(ticker: str, lookback=10, forecast_days=5, batch_size=32):
    returns = get_daily_returns(ticker, period='2y')

    split_idx = int(len(returns) * 0.8)
    train_returns = returns[:split_idx]
    test_returns = returns[split_idx:]

    train_dataset = StockReturnsDataset(train_returns, lookback, forecast_days)
    test_dataset = StockReturnsDataset(test_returns, lookback, forecast_days)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Feature shape: {train_dataset.X.shape}")

    return train_loader, test_loader

# prepare_data('AAPL')