"""This module provides general useful functions"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_config(path: str = "scripts/config.yaml") -> dict:
    """
    Load configuration file.

    :param path: Path to configuration file.
    :return: Configuration dictionary.
    """
    with open(path, "r") as file:
        return yaml.safe_load(file)


def save_plot_and_file(
    values: list,
    labels: dict,
    window: int = 30,
    path_to_pickle: str = "results/pickle_files/ttt/ttt",
    path_to_plot: str = "results/plots/ttt",
) -> None:
    """
    Save a plot of the moving average and a pickle file of raw values.

    :param values: List of raw values to save.
    :param labels: Dictionary of labels for the plot.
    :param window: Window size for moving average.
    :param path_to_pickle: Path to the pickle file.
    :param path_to_plot: Path to the plots.
    """
    if os.path.exists(path_to_pickle) or os.path.exists(path_to_plot) or not values:
        return

    arr = np.array(values)

    # Compute moving average (even with fewer values than the window)
    actual_window = min(window, len(arr))
    smoothed = np.convolve(arr, np.ones(actual_window) / actual_window, mode="valid")
    x = np.arange(actual_window - 1, len(arr))

    # Plot only the moving average
    plt.figure(figsize=(10, 5))
    plt.plot(
        x, smoothed, label=f"Moving Avg ({actual_window})", color="orange", linewidth=2
    )

    plt.title(labels["title"])
    plt.xlabel("Episode")
    plt.ylabel(labels["y_label"])
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{path_to_plot}_plot.png")
    plt.close()

    # Save raw values as pickle
    with open(f"{path_to_pickle}_values.pkl", "wb") as f:
        pickle.dump(values, f)


def log_progress(
    i: int,
    episodes: int,
    hyperparams: dict,
    ttts: list,
    interval: int = 50,
    window: int = 50,
) -> None:
    """
    Logs training progress.


    :param i: Current episode index.
    :param episodes: Total number of episodes.
    :param hyperparams: Dictionary containing training hyperparameters.
    :param ttts: List of time-to-target (or similar metric).
    :param interval: How often to print detailed info.
    :param window: Number of entries to average for first/last comparison.
    """
    # Progress bar
    percent = (i + 1) / episodes * 100
    prog_bar = "=" * int(percent // 2)  # 50-char bar
    sys.stdout.write(f"\rProgress: [{prog_bar:<50}] {percent:.1f}%")
    sys.stdout.flush()

    # Print extra info every `interval` episodes or on final episode
    if (i + 1) % interval == 0 or (i + 1) == episodes:
        ttts_array = np.array(ttts)
        first_mean = (
            np.mean(ttts_array[:window]) if len(ttts_array) >= 1 else float("nan")
        )
        last_mean = (
            np.mean(ttts_array[-window:]) if len(ttts_array) >= 1 else float("nan")
        )

        sys.stdout.write(
            f"\nEpsilon: {hyperparams['epsilon']:.4f} | "
            f"TTT first {window}: {first_mean:.2f} | "
            f"TTT last {window}: {last_mean:.2f}\n"
        )
        sys.stdout.flush()
