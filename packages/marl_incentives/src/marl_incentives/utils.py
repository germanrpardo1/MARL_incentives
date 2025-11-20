"""This module provides general useful functions"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_config(path: str = "scripts/config_file.yaml") -> dict:
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


def smooth_curve(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth a 1D array using a moving average.

    :param values: Input array of values.
    :param window_size: Smoothing window size.
    :return: (x, smoothed values)
    """
    if len(values) == 0:
        return np.array([]), np.array([])

    actual_window = min(window_size, len(values))
    smoothed = np.convolve(values, np.ones(actual_window) / actual_window, mode="valid")
    x = np.arange(actual_window - 1, len(values))
    return x, smoothed


def load_pickle_array(path: str) -> np.ndarray | None:
    """
    Load a numpy array from a pickle file.

    :param path: Path to the pickle file.
    :return: Numpy array or None if file missing.
    """
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return None

    with open(path, "rb") as f:
        try:
            return np.array(pickle.load(f))
        except Exception as e:
            print(f"[ERROR] Failed to load pickle file {path}: {e}")
            return None


def plot_multiple_curves(
    title: str,
    y_label: str,
    budgets: list[int],
    weights: dict,
    base_name: str,
    baseline_path: str | None,
    window_size: int = 30,
    ext: str = "pdf",
) -> None:
    """
    Plot smoothed curves from multiple budgets and a baseline.

    :param title: Title of the plot.
    :param y_label: Label for the Y-axis.
    :param budgets: List of budget values to plot.
    :param base_name: Metric name used in file naming and path creation.
    :param weights: Weight dictionary used in file naming.
    :param baseline_path: Path to the baseline file.
    :param window_size: Size of the smoothing window.
    :param ext: Extension for saved plot file (e.g., 'pdf' or 'png').
    """
    plt.figure(figsize=(8, 5))

    # Plot curves for each budget
    for budget in budgets:
        file_path = make_file_paths(
            base_name=base_name,
            subfolder="pickle_files",
            budget=budget,
            weights=weights,
            ext="pkl",
        )

        values = load_pickle_array(file_path)
        if values is None:
            continue

        x, smoothed = smooth_curve(values, window_size)
        _label = f"Budget {budget}" if budget != 100000000 else "Unlimited budget"
        plt.plot(x, smoothed, label=_label, linewidth=2)

    # Plot baseline
    baseline_values = load_pickle_array(baseline_path)
    if baseline_values is not None:
        baseline_values = baseline_values[:500]
        if (
            base_name == "emissions"
            or base_name == "exp_replay_emissions"
            or base_name == "compliance_rate_exp_replay_emissions"
        ):
            baseline_values /= 1000
        x, smoothed = smooth_curve(baseline_values, window_size)
        plt.plot(x, smoothed, label="Baseline", linewidth=2)

    # Finalize plot
    plt.legend()
    plt.title(title, fontsize=13)
    plt.xlabel("Episodes", fontsize=13)
    plt.ylabel(y_label, fontsize=13)

    # Save
    save_path = make_file_paths(
        base_name=base_name,
        subfolder="plots",
        budget=budgets[0],
        weights=weights,
        ext=ext,
    )
    if os.path.exists(save_path):
        print(f"[INFO] Plot already exists, skipping: {save_path}")
        plt.close()
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format=ext, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved plot: {save_path}")


def plot_weight_sensitivity(
    weights_list: list[dict],
    budgets: list[int],
    ttt_base_name: str,
    emissions_base_name: str,
    window_size: int = 100,
    ext: str = "pdf",
    save_path: str = "results/plots/weight_sensitivity",
) -> None:
    """
    Plot mean TTT and emissions sensitivity across different weight configs.

    :param weights_list: List of weight dictionaries (e.g., {'individual_tt': x, 'individual_emissions': y}).
    :param budgets: List of budgets. Uses the last budget.
    :param ttt_base_name: Base filename for TTT pickle files.
    :param emissions_base_name: Base filename for emissions pickle files.
    :param window_size: Number of final values to average.
    :param ext: File extension for the saved plot.
    :param save_path: Path to save the plot.
    """
    save_path += "." + ext

    def compute_mean(
        base_name: str, weights: dict, budget: int, label: str
    ) -> float | None:
        """Helper to compute the mean of the last window_size entries from a pickle file."""
        path = make_file_paths(
            base_name=base_name,
            subfolder="pickle_files",
            budget=budget,
            weights=weights,
            ext="pkl",
        )
        values = load_pickle_array(path)
        if values is None:
            print(f"[WARN] Missing {label} data for weights {weights}")
            return None
        return float(np.mean(values[-window_size:]))

    budget = budgets[-1]
    ttt_means, emissions_means, x_labels = [], [], []

    for weights in weights_list:
        ttt_mean = compute_mean(ttt_base_name, weights, budget, "TTT")
        emissions_mean = compute_mean(emissions_base_name, weights, budget, "emissions")

        if ttt_mean is None or emissions_mean is None:
            continue

        ttt_means.append(ttt_mean)
        emissions_means.append(emissions_mean)

        wt_tt = weights.get("individual_tt", 0.0)
        wt_em = weights.get("individual_emissions", 0.0)
        x_labels.append(f"{wt_tt:.2f}/{wt_em:.2f}")

    if not ttt_means or not emissions_means:
        print("[WARN] No valid data found for weight sensitivity plot.")
        return

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    blue = "tab:blue"
    ax1.plot(x_labels, ttt_means, marker="o", label="TTT", color=blue, linewidth=2)
    orange = "tab:orange"
    ax2.plot(
        x_labels,
        emissions_means,
        marker="s",
        label="CO2 Emissions [kg]",
        color=orange,
        linewidth=2,
    )

    ax1.set_xlabel("Weights (TTT / Emissions)", fontsize=12)
    ax1.set_ylabel("Total Travel Time [h]", color=blue, fontsize=12)
    ax2.set_ylabel("CO2 Emissions [kg]", color=orange, fontsize=12)

    ax1.tick_params(axis="y", labelcolor=blue)
    ax2.tick_params(axis="y", labelcolor=orange)

    plt.title("Weight Sensitivity Analysis", fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format=ext, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved plot: {save_path}")


def log_progress(
    i: int,
    episodes: int,
    ttts: list,
    interval: int = 50,
    window: int = 50,
) -> None:
    """
    Logs training progress.


    :param i: Current episode index.
    :param episodes: Total number of episodes.
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


def save_metric(
    values: list[float],
    labels: dict,
    base_name: str,
    y_label: str,
    budget: int,
    weights: dict,
) -> None:
    """
    Save a plot and corresponding pickle file for a given metric using
        consistent naming conventions.

    :param values: List of metric values over time (e.g., per episode).
    :param labels: Dictionary of plot labels (e.g., title, y-axis label).
    :param base_name: The metric name (used in file naming and plot title).
    :param y_label: Y-axis label for the plot.
    :param budget: Total budget used in the experiment.
    :param weights: Dictionary of weights used in the experiment.
    """
    labels["title"] = f"{base_name.replace('_', ' ').title()} per episode"
    labels["y_label"] = y_label

    # Generate file paths
    pickle_path = make_file_paths(base_name, "pickle_files", budget, weights, "pkl")
    plot_path = make_file_paths(base_name, "plots", budget, weights, "png")

    # Ensure directories exist
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Save the plot and pickle
    save_plot_and_file(
        values=values,
        labels=labels,
        window=30,
        path_to_pickle=pickle_path,
        path_to_plot=plot_path,
    )


def unpack_config(config: dict) -> tuple[dict, dict, dict, int, dict]:
    """Complete."""
    # Weights of the objective function
    weights = {
        "ttt": config["TTT_weight"],
        "individual_tt": config["individual_travel_time_weight"],
        "individual_emissions": config["emissions_weight"],
        "total_emissions": config["total_emissions_weight"],
    }

    # RL hyper-parameters
    hyperparams = {
        "epsilon": config["epsilon"],
        "decay": config["decay"],
        "alpha": config["alpha"],
    }

    # Dictionary with all paths
    paths_dict = config["paths_dict"]
    # Define edge data granularity
    edge_data_frequency = config["edge_data_frequency"]
    # Parameters to run SUMO
    sumo_params = config["sumo_config"]

    return weights, hyperparams, paths_dict, edge_data_frequency, sumo_params


def update_average_travel_times(drivers: list, weights: dict):
    """
    Update the average travel time for all routes of all drivers.

    :param drivers: List of drivers.
    :param weights: Dictionary containing all the edge's travel times.
    """
    for driver in drivers:
        driver.update_average_travel_time(weights=weights)


def logistic_prob(x: list, coefficients: list) -> np.ndarray:
    """
    Compute logistic regression probabilities given features and coefficients.

    :param x: Feature matrix (shape: [n_samples, n_features]).
               Should NOT include an intercept column.
    :param coefficients: Model coefficients, including intercept as the first element.
                         Example: [intercept, beta1, beta2, ...]
    :return: Predicted probabilities for each sample.
    """
    x = np.asarray(x)
    coef = np.asarray(coefficients)

    intercept = coef[0]
    betas = coef[1:]

    linear_combination = intercept + np.dot(x, betas)
    probs = 1 / (1 + np.exp(-linear_combination))
    return probs
