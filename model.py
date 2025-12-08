import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import yfinance as yf

from StockDataset import StockDataset
from visualization import visualize_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, forecast_days=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, forecast_days)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_model(model, train_loader, test_loader, epochs=10, epsilon=0.001, lambda_reg=0.001):
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=epsilon,
        weight_decay=lambda_reg
    )

    model.to(device)

    for epoch in range(epochs):
        model.train()

        train_loss = 0

        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()

            predictions = model(features)
            loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0

        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)

                predictions = model(features)
                loss = criterion(predictions, targets)

                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    return model

def get_ohlcv(tickers: list[str], period='1mo', interval='1d'):
    if isinstance(tickers, str):
        tickers = [tickers]

    all_data = []

    for t in tickers:
        df = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=True)
        all_data.append(df.to_numpy())

    return np.vstack(all_data) if all_data else np.array([])

def predict(model, dataset, num_predictions=5):
    model.eval()
    model.to(device)

    predictions_list = []
    actuals_list = []

    with torch.no_grad():
        for i in range(min(num_predictions, len(dataset))):
            features, targets = dataset[i]
            features = features.unsqueeze(0).to(device)  # Add batch dimension

            prediction = model(features)

            predictions_list.append(prediction.cpu().numpy()[0])
            actuals_list.append(targets.numpy())

    return np.array(predictions_list), np.array(actuals_list)


def predict_future(model, recent_data, forecast_days=5):
    model.eval()
    model.to(device)

    with torch.no_grad():
        features = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0).to(device)
        predictions = model(features)

    return predictions.cpu().numpy()[0]

if __name__ == "__main__":
    print('hello, world!')

    # hyperparameters
    lookback = 10
    forecast_days = 5

    hidden_size = 64

    batch_size = 32
    epochs = 10
    epsilon = 0.001
    lambda_reg = 0.001

    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL']

    ohlcv = get_ohlcv(tickers, period='2y', interval='1d')
    print(len(ohlcv))

    train_size = int(len(ohlcv) * 0.8)
    train_ohlcv = ohlcv[:train_size]
    test_ohlcv = ohlcv[train_size:]

    d_train = StockDataset(train_ohlcv, lookback=lookback, forecast_days=forecast_days)
    d_test = StockDataset(test_ohlcv, lookback=lookback, forecast_days=forecast_days)

    print(
        f"Total training examples: {len(d_train)}\n",
        f"Total test examples: {len(d_test)}"
    )

    train_loader = DataLoader(d_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(d_test, batch_size=batch_size, shuffle=False)

    model = StockLSTM(input_size=5, hidden_size=hidden_size, forecast_days=forecast_days)

    trained_model = train_model(model, train_loader, test_loader, epochs=epochs, epsilon=epsilon, lambda_reg=lambda_reg)

    predictions, actuals = predict(trained_model, d_test, num_predictions=86)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Actuals shape: {actuals.shape}")

    print("\nFirst 5 predictions:")
    print(predictions[:5])

    predictions_dollars = d_test.denormalize(predictions)
    actuals_dollars = d_test.denormalize(actuals)

    print(f"\nFirst 5 predictions in dollars: {predictions_dollars[:5]}")
    print(f"First 5 actuals in dollars: {actuals_dollars[:5]}")
    print(predictions_dollars[:5])

    [visualize_test(predictions_dollars[i], actuals_dollars[i]) for i in range(5)]
