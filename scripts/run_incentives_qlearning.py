"""This script runs the MARL algorithm with and without incentives."""

import os

from marl_incentives import environment as env
from marl_incentives import traveller as tr
from marl_incentives import utils as ut


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
    pickle_path = ut.make_file_paths(base_name, "pickle_files", budget, weights, "pkl")
    plot_path = ut.make_file_paths(base_name, "plots", budget, weights, "png")

    # Ensure directories exist
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Save the plot and pickle
    ut.save_plot_and_file(
        values=values,
        labels=labels,
        window=30,
        path_to_pickle=pickle_path,
        path_to_plot=plot_path,
    )


def main(config: dict, total_budget: int) -> None:
    """
    Run the MARL algorithm with or without incentives.

    :param config: Configuration dictionary.
    :param total_budget: Total budget.
    """
    incentives_mode = config["incentives_mode"]
    episodes = config["episodes"]

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

    # Instantiate drivers object
    drivers = tr.Drivers(actions_file_path=paths_dict["output_rou_alt_path"])

    # Get available actions based on pre-computed routes
    actions_and_costs = drivers.get_actions()

    # Unpack trips IDs
    trips_id = list(actions_and_costs.keys())

    ttts = []
    emissions_total = []
    labels_dict = {}

    # Initialise the Q-function
    q_values = (
        tr.initialise_q_function_incentives(actions_costs=actions_and_costs)
        if incentives_mode
        else tr.initialise_q_function_no_incentives(actions_costs=actions_and_costs)
    )

    # Instantiate network object
    network_env = env.Network(
        paths_dict=paths_dict,
        sumo_params=sumo_params,
        edge_data_frequency=edge_data_frequency,
    )

    # Train RL agent
    for i in range(episodes):
        # Select policy function based on whether incentives are used or not
        # Get actions from policy
        routes_edges, actions_index = tr.policy_function(
            incentives_mode=incentives_mode,
            trips_id=trips_id,
            q=q_values,
            actions_costs=actions_and_costs,
            epsilon=hyperparams["epsilon"],
            total_budget=total_budget,
            strategy=config["strategy"],
        )
        # Perform actions given by policy
        total_tt, ind_tt, ind_em, total_em = network_env.step(
            routes_edges=routes_edges,
        )

        # Record TTT and total emissions throughout iterations
        ttts.append(total_tt)
        emissions_total.append(total_em)

        # For each agent update Q function
        # Q(a) = (1 - alpha) * Q(a) + alpha * r
        for trip in trips_id:
            idx = actions_index[trip]
            # Compute reward
            reward = tr.compute_reward(
                trip, ind_tt, ind_em, total_tt, total_em, weights
            )
            # Update Q-value
            q_values[trip][idx] = (1 - hyperparams["alpha"]) * q_values[trip][
                idx
            ] + hyperparams["alpha"] * reward

        # Logging
        ut.log_progress(i=i, episodes=episodes, hyperparams=hyperparams, ttts=ttts)

        # Reduce epsilon
        hyperparams["epsilon"] = max(
            0.01, hyperparams["epsilon"] * hyperparams["decay"]
        )

        # Retrieve updated route costs
        # costs = calculate_route_cost(actions, parse_weights("data/weights.xml"))

    # Save the plot and pickle file for TTT and emissions
    save_metric(ttts, labels_dict, "ttt", "TTT [h]", total_budget, weights)
    save_metric(
        emissions_total,
        labels_dict,
        "emissions",
        "Emissions [kg]",
        total_budget,
        weights,
    )


if __name__ == "__main__":
    # Load config
    config_file = ut.load_config(path="scripts/run_incentives_q_learning.yaml")

    # Loop for different budgets
    for tot_budget in config_file["total_budget"]:
        main(config=config_file, total_budget=tot_budget)
