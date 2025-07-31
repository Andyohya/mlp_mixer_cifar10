import matplotlib.pyplot as plt

def plot_metrics(metric_list, title="Training"):
    plt.plot(metric_list)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


