"""This script generates the plots."""

import marl_incentives.utils as ut


def plot_multiple_budgets(config_file: dict) -> None:
    """
    Plot curves with different budgets.

    :param config_file: Dictionary of metric parameters (e.g., budget).
    """
    ttt_y_label = "TTT [h]"
    emissions_y_label = "Emissions [kg]"
    dua_iterate_emissions = "results/pickle_files/emissions/duaIterate_emissions.pkl"
    dua_iterate_time = "results/pickle_files/ttt/duaIterate_time.pkl"
    title_multi_obj = "Multiobjective reward"

    ut.plot_multiple_curves(
        title="Minimising travel time",
        y_label=ttt_y_label,
        budgets=config_file["total_budget"],
        base_name="ttt",
        weights={"individual_tt": 1, "individual_emissions": 0},
        baseline_path=dua_iterate_time,
    )

    ut.plot_multiple_curves(
        title="Minimising travel time",
        y_label=emissions_y_label,
        budgets=config_file["total_budget"],
        base_name="emissions",
        weights={"individual_tt": 1, "individual_emissions": 0},
        baseline_path=dua_iterate_emissions,
    )

    ut.plot_multiple_curves(
        title=title_multi_obj,
        y_label=ttt_y_label,
        budgets=config_file["total_budget"],
        base_name="ttt",
        weights={"individual_tt": 0.75, "individual_emissions": 0.25},
        baseline_path=dua_iterate_time,
    )

    ut.plot_multiple_curves(
        title=title_multi_obj,
        y_label=emissions_y_label,
        budgets=config_file["total_budget"],
        base_name="emissions",
        weights={"individual_tt": 0.75, "individual_emissions": 0.25},
        baseline_path=dua_iterate_emissions,
    )

    ut.plot_multiple_curves(
        title=title_multi_obj,
        y_label=ttt_y_label,
        budgets=config_file["total_budget"],
        base_name="ttt",
        weights={"individual_tt": 0.5, "individual_emissions": 0.5},
        baseline_path=dua_iterate_time,
    )

    ut.plot_multiple_curves(
        title=title_multi_obj,
        y_label=emissions_y_label,
        budgets=config_file["total_budget"],
        base_name="emissions",
        weights={"individual_tt": 0.5, "individual_emissions": 0.5},
        baseline_path=dua_iterate_emissions,
    )

    ut.plot_multiple_curves(
        title=title_multi_obj,
        y_label=ttt_y_label,
        budgets=config_file["total_budget"],
        base_name="ttt",
        weights={"individual_tt": 0.25, "individual_emissions": 0.75},
        baseline_path=dua_iterate_time,
    )

    ut.plot_multiple_curves(
        title=title_multi_obj,
        y_label=emissions_y_label,
        budgets=config_file["total_budget"],
        base_name="emissions",
        weights={"individual_tt": 0.25, "individual_emissions": 0.75},
        baseline_path=dua_iterate_emissions,
    )

    ut.plot_multiple_curves(
        title="Minimising emissions",
        y_label=ttt_y_label,
        budgets=config_file["total_budget"],
        base_name="ttt",
        weights={"individual_tt": 0, "individual_emissions": 1},
        baseline_path=dua_iterate_time,
    )

    ut.plot_multiple_curves(
        title="Minimising emissions",
        y_label=emissions_y_label,
        budgets=config_file["total_budget"],
        base_name="emissions",
        weights={"individual_tt": 0, "individual_emissions": 1},
        baseline_path=dua_iterate_emissions,
    )
    weights_list = [
        {"individual_tt": 1, "individual_emissions": 0},
        {"individual_tt": 0.75, "individual_emissions": 0.25},
        {"individual_tt": 0.5, "individual_emissions": 0.5},
        {"individual_tt": 0.25, "individual_emissions": 0.75},
        {"individual_tt": 0, "individual_emissions": 1},
    ]

    ut.plot_weight_sensitivity(
        weights_list=weights_list,
        budgets=config_file["total_budget"],
        ttt_base_name="ttt",
        emissions_base_name="emissions",
        save_path="results/plots/weight_sensitivity.pdf",
    )


if __name__ == "__main__":
    # Load config
    config = ut.load_config(path="scripts/config.yaml")

    plot_multiple_budgets(config)
