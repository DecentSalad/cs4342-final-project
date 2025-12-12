import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os

def display_chart(ticker: str, period: str = '1mo', interval: str = '1d'):
    df = yf.download(ticker, period=period, interval=interval)

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'])
    plt.title(f"{ticker} Closing Prices")
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.tight_layout()
    plt.show()

def get_daily_returns(ticker: str, period: str = '1mo', progress=False):
    df = yf.download(ticker, period=period, interval='1d', progress=progress, auto_adjust=True)['Close']
    close_prices = df.to_numpy().ravel()
    returns = np.diff(close_prices) / close_prices[:-1]  # returns = (P_t - P_{t-1}) / P_{t-1}

    return returns

def get_samples(ticker: str, period='2y', lookback=10, forecast_days=5, cache_dir="stock_data"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{period}_{lookback}_{forecast_days}.csv")

    if os.path.exists(cache_file):
        print('loading saved data')

        df = pd.read_csv(cache_file)
        samples = np.array([np.fromstring(s, sep=' ').reshape(lookback + forecast_days, -1) for s in df['samples']])
        base_prices = np.array(df['base_prices'])
        sample_dates = np.array([d.split('|') for d in df['sample_dates']])
        sample_tickers = np.array(df['sample_tickers'])
        return samples, base_prices, sample_dates, sample_tickers

    print(f'downloading data for {ticker}')

    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    prices = data.values
    dates = data.index.strftime('%Y-%m-%d').tolist()

    sample_len = lookback + forecast_days

    samples, base_prices, sample_dates, sample_tickers = [], [], [], []

    for i in range(len(prices) - sample_len + 1):
        sample_window = prices[i: i + sample_len]
        date_window = dates[i: i + sample_len]

        # pct_change[t] = (price[t] - price[t-1]) / price[t-1]
        pct_changes = np.zeros_like(sample_window)
        pct_changes[0] = 0

        for j in range(1, len(sample_window)):
            pct_changes[j] = (sample_window[j] - sample_window[j - 1]) / sample_window[j - 1]

        pct_changes = np.nan_to_num(pct_changes, nan=0.0, posinf=0.0, neginf=0.0)

        last_lookback_price = sample_window[lookback - 1, 3]

        samples.append(pct_changes)
        base_prices.append(last_lookback_price)
        sample_dates.append(date_window)
        sample_tickers.append(ticker)

    df = pd.DataFrame({
        "samples": [' '.join(map(str, s.flatten())) for s in samples],
        "base_prices": base_prices,
        "sample_dates": ['|'.join(d) for d in sample_dates],
        "sample_tickers": sample_tickers
    })
    df.to_csv(cache_file, index=False)

    return np.array(samples), np.array(base_prices), np.array(sample_dates), np.array(sample_tickers)