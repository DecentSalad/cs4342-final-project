import matplotlib.pyplot as plt

def visualize_test(predictions, actuals):
    plt.figure(figsize=(8, 5))

    plt.plot(predictions, label="Predictions", marker="o")
    plt.plot(actuals, label="Actuals", marker="s")

    # Add labels and title
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Predictions vs Actuals")
    plt.legend()
    plt.grid(True)

    plt.show()