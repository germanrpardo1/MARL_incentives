"""This script generates the plots."""

import marl_incentives.utils as ut


def plot_multiple_budgets(config_file: dict) -> None:
    """
    Plot curves with different budgets for travel time and emissions.

    :param config_file: Dictionary loaded from YAML containing:
        - total_budget: list of budget values
        - paths: dict of file paths (dua_iterate_emissions, dua_iterate_time, weight_sensitivity_plot)
        - weights: list of weight dictionaries for TT vs emissions
        - labels: dict of axis labels
        - titles: dict of plot titles
    """
    budgets: list[int] = config_file["total_budget"]
    paths: dict[str, str] = config_file["paths"]
    weights_list: list[dict[str, float]] = config_file["weights"]
    labels: dict[str, str] = config_file["labels"]
    titles: dict[str, str] = config_file["titles"]
    files_extension: str = config_file["files_extension"]
    window_size_sensitivity: int = config_file["window_size_sensitivity"]
    window_size: int = config_file["window_size"]

    def plot_curve_set(title: str, weights: dict[str, float]) -> None:
        """Helper to plot both TTT and emissions curves with given weights."""
        ut.plot_multiple_curves(
            title=title,
            y_label=labels["ttt"],
            budgets=budgets,
            base_name="compliance_rate_exp_replay_ttt",
            weights=weights,
            baseline_path=paths["dua_iterate_time"],
            window_size=window_size,
            ext=files_extension,
        )
        ut.plot_multiple_curves(
            title=title,
            y_label=labels["emissions"],
            budgets=budgets,
            base_name="compliance_rate_exp_replay_emissions",
            weights=weights,
            baseline_path=paths["dua_iterate_emissions"],
            window_size=window_size,
            ext=files_extension,
        )
        ut.plot_multiple_curves(
            title=title,
            y_label=labels["acceptance_rates"],
            budgets=budgets,
            base_name="compliance_rate_exp_replay_acceptance_rates",
            weights=weights,
            baseline_path="None",
            window_size=window_size,
            ext=files_extension,
        )
        ut.plot_multiple_curves(
            title=title,
            y_label=labels["used_budget"],
            budgets=budgets,
            base_name="compliance_rate_exp_replay_used_budget",
            weights=weights,
            baseline_path="None",
            window_size=window_size,
            ext=files_extension,
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
        ext=files_extension,
        window_size=window_size_sensitivity,
    )


if __name__ == "__main__":
    # Load config_file
    config = ut.load_config(path="scripts/generate_plots.yaml")

    plot_multiple_budgets(config)
