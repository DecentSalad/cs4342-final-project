from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

def plot_mse_loss(train_losses, test_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()


def print_predictions(ticker, predictions, n_days):
    """Print formatted prediction results."""
    print(f"\n{'=' * 60}")
    print(f"Predictions for {ticker}")
    print(f"{'=' * 60}")
    print(f"Current Price: ${predictions['current_price']:.2f}")
    print(f"\nPredicted prices for the next {n_days} days:")
    print(f"{'Day':<6} {'Date':<12} {'Predicted Price':<18} {'Daily Return':<15}")
    print(f"{'-' * 60}")

    # Generate dates starting from tomorrow
    start_date = datetime.now() + timedelta(days=1)

    for i, (price, ret) in enumerate(zip(predictions['predicted_prices'],
                                         predictions['predicted_returns']), 1):
        pred_date = start_date + timedelta(days=i - 1)
        date_str = pred_date.strftime('%Y-%m-%d')
        print(f"{i:<6} {date_str:<12} ${price:<17.2f} {ret * 100:>6.2f}%")

    total_change = predictions['predicted_prices'][-1] - predictions['current_price']
    total_change_pct = (total_change / predictions['current_price']) * 100

    print(f"\n{'-' * 60}")
    print(f"Total predicted change: ${total_change:+.2f} ({total_change_pct:+.2f}%)")
    print(f"{'=' * 60}\n")


def plot_predicted_prices(ticker, predictions, lookback_days=10):
    """
    Plot historical prices and predicted future prices.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    predictions : dict
        Dictionary containing 'current_price', 'predicted_prices', and 'predicted_returns'
    lookback_days : int
        Number of historical days to display (default: 30)
    """
    # Get historical data
    df = yf.download(ticker, period=f'{lookback_days + 5}d', interval='1d',
                     progress=False, auto_adjust=True)['Close']
    historical_prices = df.to_numpy().ravel()
    historical_dates = df.index

    # Generate future dates
    last_date = historical_dates[-1]
    future_dates = [last_date + timedelta(days=i + 1)
                    for i in range(len(predictions['predicted_prices']))]

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot historical prices
    plt.plot(historical_dates, historical_prices,
             label='Historical Prices', color='blue', linewidth=2)

    # Plot predicted prices (connect from current price)
    combined_dates = [historical_dates[-1]] + future_dates
    combined_prices = [predictions['current_price']] + predictions['predicted_prices']

    plt.plot(combined_dates, combined_prices,
             label='Predicted Prices', color='red', linewidth=2,
             linestyle='--', marker='o', markersize=6)

    # Add vertical line at prediction start
    plt.axvline(x=historical_dates[-1], color='gray',
                linestyle=':', alpha=0.7, label='Prediction Start')

    # Formatting
    plt.title(f'{ticker} - Historical and Predicted Prices', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display price change info
    total_change = predictions['predicted_prices'][-1] - predictions['current_price']
    total_change_pct = (total_change / predictions['current_price']) * 100

    plt.text(0.02, 0.98,
             f"Current: ${predictions['current_price']:.2f}\n"
             f"Predicted: ${predictions['predicted_prices'][-1]:.2f}\n"
             f"Change: {total_change_pct:+.2f}%",
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.show()