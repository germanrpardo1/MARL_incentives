"""This script generates the plots."""

import marl_incentives.utils as ut


def plot_multiple_budgets(config_file: dict) -> None:
    """
    Plot curves with different budgets.

    :param config: Dictionary of metric parameters (e.g., budget).
    """
    ttt_y_label = "TTT [h]"
    emissions_y_label = "Emissions [kg]"
    ut.plot_multiple_curves(
        title="Minimising travel time",
        y_label=ttt_y_label,
        budgets=config_file["total_budget"],
        base_name="ttt",
        weights={"individual_tt": 1, "individual_emissions": 0},
    )

    ut.plot_multiple_curves(
        title="Minimising travel time",
        y_label=emissions_y_label,
        budgets=config_file["total_budget"],
        base_name="emissions",
        weights={"individual_tt": 1, "individual_emissions": 0},
    )

    ut.plot_multiple_curves(
        title="Minimising emissions",
        y_label=ttt_y_label,
        budgets=config_file["total_budget"],
        base_name="ttt",
        weights={"individual_tt": 0, "individual_emissions": 1},
    )

    ut.plot_multiple_curves(
        title="Minimising emissions",
        y_label=emissions_y_label,
        budgets=config_file["total_budget"],
        base_name="emissions",
        weights={"individual_tt": 0, "individual_emissions": 1},
    )

    ut.plot_multiple_curves(
        title="Multiobjective reward",
        y_label=ttt_y_label,
        budgets=config_file["total_budget"],
        base_name="ttt",
        weights={"individual_tt": 0.5, "individual_emissions": 0.5},
    )

    ut.plot_multiple_curves(
        title="Multiobjective reward",
        y_label=emissions_y_label,
        budgets=config_file["total_budget"],
        base_name="emissions",
        weights={"individual_tt": 0.5, "individual_emissions": 0.5},
    )


if __name__ == "__main__":
    # Load config
    config = ut.load_config(path="scripts/config.yaml")

    plot_multiple_budgets(config)
