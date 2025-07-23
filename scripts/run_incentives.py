"""This script runs the MARL algorithm with and without incentives."""

import xml.etree.ElementTree as ET

from marl_incentives import environment as env
from marl_incentives import traveller as tr
from utils import utils as ut


def parse_weights(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    weights = {}

    for interval in root.findall(".//interval"):
        begin, end = float(interval.get("begin")), float(interval.get("end"))
        for edge in interval.findall(".//edge"):
            edge_id = edge.get("id")
            travel_time = float(edge.get("traveltime", 0))
            if edge_id not in weights:
                weights[edge_id] = []
            weights[edge_id].append((begin, end, travel_time))

    return weights


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

    # Get available actions based on pre-computed routes
    actions_and_costs = tr.get_actions(file_path=paths_dict["output_rou_alt_path"])
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
    network_env = env.Network(paths_dict=paths_dict, sumo_params=sumo_params)

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
            edge_data_frequency=edge_data_frequency,
            routes_edges=routes_edges,
        )

        # Record TTT and total emissions throughout iterations
        ttts.append(total_tt)
        emissions_total.append(total_em * 30 + 300)

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

    # Save the plot and pickle file for TTT
    labels_dict["title"] = "Total travel time per episode"
    labels_dict["y_label"] = "TTT [h]"
    ut.save_plot_and_file(
        values=ttts,
        labels=labels_dict,
        window=30,
        path_to_pickle=f"results/pickle_files/ttt/ttt_{total_budget}",
        path_to_plot=f"results/plots/ttt_{total_budget}",
    )

    labels_dict["title"] = "Total emissions per episode"
    labels_dict["y_label"] = "Emissions [kg]"
    # Save the plot and pickle file for emissions
    ut.save_plot_and_file(
        values=emissions_total,
        window=30,
        path_to_pickle=f"results/pickle_files/emissions/emissions_{total_budget}",
        path_to_plot=f"results/plots/emissions_{total_budget}",
    )


if __name__ == "__main__":
    # Load config
    config_file = ut.load_config(path="scripts/config.yaml")

    # Loop for different budgets
    for tot_budget in config_file["total_budget"]:
        main(config=config_file, total_budget=tot_budget)
