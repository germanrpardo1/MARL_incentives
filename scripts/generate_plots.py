"""This script generates the plots."""

from typing import Dict, List

import marl_incentives.utils as ut


def plot_multiple_budgets(config_file: Dict) -> None:
    """
    Plot curves with different budgets for travel time and emissions.

    :param config_file: Dictionary loaded from YAML containing:
        - total_budget: list of budget values
        - paths: dict of file paths (dua_iterate_emissions, dua_iterate_time, weight_sensitivity_plot)
        - weights: list of weight dictionaries for TT vs emissions
        - labels: dict of axis labels
        - titles: dict of plot titles
    """
    budgets: List[int] = config_file["total_budget"]
    paths: Dict[str, str] = config_file["paths"]
    weights_list: List[Dict[str, float]] = config_file["weights"]
    labels: Dict[str, str] = config_file["labels"]
    titles: Dict[str, str] = config_file["titles"]

    def plot_curve_set(title: str, weights: Dict[str, float]) -> None:
        """Helper to plot both TTT and emissions curves with given weights."""
        ut.plot_multiple_curves(
            title=title,
            y_label=labels["ttt"],
            budgets=budgets,
            base_name="ttt",
            weights=weights,
            baseline_path=paths["dua_iterate_time"],
        )
        ut.plot_multiple_curves(
            title=title,
            y_label=labels["emissions"],
            budgets=budgets,
            base_name="emissions",
            weights=weights,
            baseline_path=paths["dua_iterate_emissions"],
        )

    # Plot single-objective and multi-objective cases
    plot_curve_set(titles["time"], weights_list[0])  # Travel time only
    for weights in weights_list[1:-1]:  # Mixed weights
        plot_curve_set(titles["multi"], weights)
    plot_curve_set(titles["emissions"], weights_list[-1])  # Emissions only

    # Plot weight sensitivity analysis
    ut.plot_weight_sensitivity(
        weights_list=weights_list,
        budgets=budgets,
        ttt_base_name="ttt",
        emissions_base_name="emissions",
        save_path=paths["weight_sensitivity_plot"],
    )


if __name__ == "__main__":
    # Load config_file
    config = ut.load_config(path="scripts/plots_config.yaml")

    plot_multiple_budgets(config)
