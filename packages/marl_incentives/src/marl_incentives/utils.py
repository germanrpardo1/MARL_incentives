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
    path_to_pickle: str,
    path_to_plot: str,
    window: int = 30,
) -> None:
    """
    Save a plot of the moving average and a pickle file of raw values.

    :param values: List of raw values to save.
    :param labels: Dictionary of labels for the plot.
    :param path_to_pickle: Path to the pickle file.
    :param path_to_plot: Path to the plots.
    :param window: Window size for moving average.
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
    plt.savefig(f"{path_to_plot}")
    plt.close()

    # Save raw values as pickle
    with open(f"{path_to_pickle}", "wb") as f:
        pickle.dump(values, f)


def plot_multiple_curves(
    title: str,
    y_label: str,
    budgets: list,
    weights: dict,
    base_name: str,
    baseline_path: str,
    window_size: int = 30,
    ext: str = "pdf",
) -> None:
    """
    Plot smoothed curves from multiple budgets by reading values from pickle files and save the plot.

    :param title: Title of the plot.
    :param y_label: Label for the Y-axis.
    :param budgets: List of budget values to plot.
    :param base_name: Metric name used in file naming and path creation.
    :param weights: Weight dictionary used in file naming.
    :param baseline_path: Path to the baseline file.
    :param window_size: Size of the smoothing window.
    :param ext: Extension for saved plot file (e.g., 'pdf' or 'png').
    """
    for budget in budgets:
        file_path = make_file_paths(
            base_name=base_name,
            subfolder="pickle_files",
            budget=budget,
            weights=weights,
            ext="pkl",
        )

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, "rb") as f:
            values = pickle.load(f)

        arr = np.array(values)
        actual_window = min(window_size, len(arr))
        smoothed = np.convolve(
            arr, np.ones(actual_window) / actual_window, mode="valid"
        )
        x = np.arange(actual_window - 1, len(arr))
        plt.plot(x, smoothed, label=f"Budget {budget}", linewidth=2)

    with open(baseline_path, "rb") as f:
        values = pickle.load(f)

    arr = np.array(values)[0:500] / 1000
    actual_window = min(window_size, len(arr))
    smoothed = np.convolve(arr, np.ones(actual_window) / actual_window, mode="valid")
    x = np.arange(actual_window - 1, len(arr))
    plt.plot(x, smoothed, label="Baseline", linewidth=2)

    plt.legend()
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel(y_label)

    # Use first budget to build save path
    save_path = make_file_paths(
        base_name=base_name,
        subfolder="plots",
        budget=budgets[0],
        weights=weights,
        ext=ext,
    )
    if os.path.exists(save_path):
        plt.close()
        return
    # Make sure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, format=ext, bbox_inches="tight")
    plt.close()


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


def make_file_paths(
    base_name: str, subfolder: str, budget: int, weights: dict, ext: str
) -> str:
    """
    Construct a full file path for saving plots or pickle files based on experiment parameters.

    :param base_name: The base name for the metric (e.g., "ttt", "emissions").
    :param subfolder: Subdirectory under 'results' (e.g., "plots", "pickle_files").
    :param budget: The total budget used in the experiment.
    :param weights: Dictionary containing weights, expects keys 'individual_tt' and 'individual_emissions'.
    :param ext: File extension (e.g., "png", "pkl").
    :return: Full file path as a string.
    """
    w_tt = round(weights["individual_tt"], 3)
    w_em = round(weights["individual_emissions"], 3)
    filename = f"{base_name}_{budget}_ttt_obj_{w_tt}_emissions_obj_{w_em}.{ext}"
    return os.path.join(f"results/{subfolder}/{base_name}", filename)


def get_travel_time(edge_id, timestamp, weights):
    if edge_id not in weights:
        return float(0)

    for begin, end, travel_time in weights[edge_id]:
        if begin <= timestamp < end:
            return travel_time

    return float(0)


def calculate_route_cost(actions, weights):
    costs_r = {}

    for i, (trip, routes) in enumerate(actions.items()):
        trip_costs = []
        for _, route in routes:
            timestamp = i * 0.09  # Initial departure time for each trip
            total_cost = 0

            for edge in route:
                travel_time = get_travel_time(edge, timestamp, weights)
                total_cost += travel_time
                timestamp += travel_time  # Update timestamp as we move through edges

            trip_costs.append(round(total_cost, 2))

        costs_r[trip] = trip_costs

    return costs_r
