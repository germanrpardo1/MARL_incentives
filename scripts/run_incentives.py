"""This script runs the MARL algorithm with and without incentives."""

import sys

import numpy as np
from typing import Any
import xml.etree.ElementTree as ET

import subprocess
import sumolib
from utils import utils as ut
from cfgv import Array

import co2Emissions.calculate_emissions as em


def eps_greedy_policy_no_incentives(
    q: Array, actions_costs: tuple[list, list], num_routes: int, epsilon: float = 0.1
) -> tuple[list, int]:
    """
    Epsilon-greedy policy for selecting a route without incentives.

    :param q: Q-values corresponding to a given trip_id
    :param actions_costs: Routes and costs corresponding to a given trip_id
    :param num_routes: Number of routes available
    :param epsilon: Probability of choosing a random action

    :return: (route_edges, selected_action, action_index)
    """
    # Unpack routes and costs
    routes, _ = actions_costs

    # Define random generator
    rng = np.random.default_rng()
    # Perform random action with probability epsilon
    if rng.random() <= epsilon:
        random_int = rng.integers(num_routes)
        route_idx = int(random_int)  # Random action index
    # Perform action with maximum Q-value with probability 1 - epsilon
    else:
        route_idx = np.argmin(q)

    # Get route edges to write them in the .XML file
    route_edges = routes[route_idx][1]
    return route_edges, route_idx


def eps_greedy_policy_incentives(
    q: Array, actions_costs: tuple[list, list], num_routes: int, epsilon: float = 0.1
) -> tuple[list, int, int, float]:
    """
    Epsilon-greedy policy for selecting a route with incentives.

    :param q: Q-values corresponding to a given trip_id
    :param actions_costs: Routes and costs corresponding to a given trip_id
    :param num_routes: Number of routes available
    :param epsilon: Probability of choosing a random action

    :return: (route_edges, selected_action, action_index, applied_incentive)
    """
    # Unpack routes and costs
    routes, costs = actions_costs

    # Define random generator
    rng = np.random.default_rng()
    # Perform random action with probability epsilon
    if rng.random() <= epsilon:
        random_int = rng.integers(num_routes + 1)
        action_index = int(random_int)  # Random action index
    # Perform action with maximum Q-value with probability 1 - epsilon
    else:
        action_index = np.argmin(q)

    adjusted_costs = costs.copy()
    # Compute incentive to apply
    if action_index + 1 <= num_routes:
        incentive = costs[action_index] - min(costs) + 1
        # Apply incentive to cost
        adjusted_costs[action_index] -= incentive
    else:
        incentive = 0

    # Choose the route with the minimum adjusted cost
    selected_action = int(np.argmin(adjusted_costs))
    route_edges = routes[selected_action][1]

    return route_edges, selected_action, action_index, incentive


def step(
    edge_data_frequency: int,
    routes_edges: dict,
    paths_dict: dict,
    sumo_params: dict,
) -> tuple[float, dict, dict, float]:
    """
    Perform one SUMO simulation step:
    - Write route and configuration files
    - Run the SUMO simulation
    - Process travel time and emission outputs

    :param edge_data_frequency: Frequency of edge data (in seconds)
    :param routes_edges: Route edges for each trip_id
    :param paths_dict: Dict of paths (routes file, edge data, emissions, etc.)
    :param sumo_params: SUMO simulation configuration
    :return: Tuple of (total travel time, normalised individual travel times,
                      normalised individual emissions, normalised total emissions)
    """
    # Write input and configuration files
    ut.write_routes(routes_edges, paths_dict["routes_file_path"])
    ut.write_edge_data_config(
        paths_dict["edge_data_path"],
        sumo_params["edges_weights_path"],
        edge_data_frequency,
    )
    ut.write_sumo_config(**sumo_params)

    # Run SUMO simulation
    run_simulation(paths_dict["log_path"], sumo_params["config_path"])

    # Process outputs
    total_tt = ut.get_ttt(paths_dict["stats_path"])
    individual_tt = ut.normalise_dict(
        ut.get_individual_travel_times(paths_dict["trip_info_path"])
    )
    individual_emissions = ut.normalise_dict(
        em.calculate_emissions_per_vehicle(paths_dict["emissions_per_vehicle_path"])
    )
    total_emissions = ut.normalise_scalar(
        min_val=300, max_val=330, val=em.co2_main(paths_dict["emissions_path"]) / 1000
    )

    return total_tt, individual_tt, individual_emissions, total_emissions


def get_actions(file_path: str) -> dict:
    """
    Parse an XML file and extract available vehicle routes and their costs.

    :param file_path: Path to the XML file.

    :return: A dictionary mapping vehicle IDs to lists of (index, edge list) tuples
        and costs
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    vehicle_routes_and_costs = {}  # vehicle_id -> [(index, edges), costs]
    vehicle_ids = []  # List of vehicle IDs

    # Loop through all the vehicles
    for vehicle in root.findall("vehicle"):
        # Get vehicle ID
        vehicle_id = vehicle.get("id")
        vehicle_ids.append(vehicle_id)

        routes = []
        costs = []

        route_distribution = vehicle.find("routeDistribution")
        if route_distribution is not None:
            # Loop through all routes for the given vehicle
            for i, route in enumerate(route_distribution.findall("route")):
                edge_list = route.get("edges", "").split()
                cost = float(route.get("cost", "0"))
                routes.append((i, edge_list))
                costs.append(cost)

        vehicle_routes_and_costs[vehicle_id] = (routes, costs)

    return vehicle_routes_and_costs


def run_simulation(log_path: str, sumo_config_path: str) -> None:
    """
    Run a SUMO simulation.

    :param log_path: Path to log file.
    :param sumo_config_path: Path to SUMO configuration file.
    """
    sys.stdout = sumolib.TeeFile(sys.stdout, open(log_path, "w+"))
    log = open(log_path, "w+")

    log.flush()
    sys.stdout.flush()

    sumo_cmd = ["sumo", "-c", sumo_config_path]

    sumo_cmd = list(map(str, sumo_cmd))
    subprocess.call(sumo_cmd, stdout=log, stderr=log)


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


def initialise_q_function_no_incentives(actions_costs: dict) -> dict:
    """
    Initialise the Q-function.

    :param actions_costs: Dictionary that stores the possible routes for each
        trip and their cost.

    :return: initialised Q-function for every trip.
    """
    return {
        trip: np.zeros((len(action_cost[0])))
        for trip, action_cost in actions_costs.items()
    }


def initialise_q_function_incentives(actions_costs: dict) -> dict:
    """
    Initialise the Q-function.

    :param actions_costs: Dictionary that stores the possible routes for each
        trip and their cost.

    :return: initialised Q-function for every trip.
    """
    return {
        trip: np.zeros((len(action_cost[0]) + 1))
        for trip, action_cost in actions_costs.items()
    }


def policy_no_incentives(
    trips_id: list, q: dict, actions_costs: dict, epsilon: float
) -> tuple[dict, dict]:
    """
    Policy function for the RL algorithm using epsilon-greedy strategy
        for when no incentives are applied.

    :param trips_id: List of trip IDs
    :param q: Q-function mapping trip_id to Q-value array
    :param actions_costs: Dictionary mapping trip_id to (routes, costs)
    :param epsilon: Probability of choosing a random action

    :return: Tuple containing:
             - route_edges: mapping trip_id to selected route edges
             - actions_index: mapping trip_id to selected (route_idx, incentive_level)
    """
    route_edges = {}
    actions_index = {}

    for trip_id in trips_id:
        _, costs = actions_costs[trip_id]
        num_routes = len(costs)

        # Select action using epsilon-greedy strategy
        selected_edges, selected_index = eps_greedy_policy_no_incentives(
            q[trip_id], actions_costs[trip_id], num_routes, epsilon
        )

        route_edges[trip_id] = selected_edges
        actions_index[trip_id] = selected_index

    return route_edges, actions_index


def policy_incentives(
    trips_id: list,
    q: dict,
    actions_costs: dict,
    epsilon: float,
    total_budget: float,
) -> tuple[dict, dict]:
    """
    Policy function for the RL algorithm using epsilon-greedy strategy
        for when incentives are applied.

    :param trips_id: List of trip IDs
    :param q: Q-function mapping trip_id to Q-value array
    :param actions_costs: Dictionary mapping trip_id to (routes, costs)
    :param epsilon: Probability of choosing a random action
    :param total_budget: Maximum total incentive budget

    :return: Tuple containing:
             - route_edges: mapping trip_id to selected route edges
             - actions_index: mapping trip_id to selected (route_idx, incentive_level)
    """
    route_edges = {}
    actions_index = {}
    current_budget = 0

    for trip_id in trips_id:
        routes, costs = actions_costs[trip_id]
        num_routes = len(costs)

        # Select action using epsilon-greedy strategy
        selected_edges, selected_action, selected_index, incentive = (
            eps_greedy_policy_incentives(
                q[trip_id], actions_costs[trip_id], num_routes, epsilon
            )
        )

        # Enforce budget constraint
        if current_budget + incentive > total_budget:
            selected_action = int(np.argmin(costs))
            selected_edges = routes[selected_action][1]
            incentive = 0

        # Update budget and tracking dictionaries
        current_budget += incentive
        route_edges[trip_id] = selected_edges
        actions_index[trip_id] = selected_index

    return route_edges, actions_index


def compute_reward(
    trip: str,
    ind_tt: dict,
    ind_em: dict,
    total_tt: float,
    total_em: float,
    weights: dict,
) -> float:
    """
    Compute the multi-objective reward.

    :param trip: Trip ID.
    :param ind_tt: Individual travel times.
    :param ind_em: Individual emissions.
    :param total_tt: Total travel time.
    :param total_em: Total emissions.
    :param weights: Weight of incentives.
    :return: Multi-objective reward.
    """
    return (
        weights["individual_tt"] * ind_tt[trip]
        + weights["ttt"] * total_tt
        + weights["individual_emissions"] * ind_em[trip]
        + weights["total_emissions"] * total_em
    )


def policy_function(incentives_mode: bool, **kwargs: Any) -> tuple[dict, dict]:
    """
    Select the policy based on whether incentives are applied or not.

    :param incentives_mode: True if incentives are used, False otherwise.
    :param kwargs: Keyword arguments passed to policy function.
    :return: Policy function.
    """
    if incentives_mode:
        return policy_incentives(**kwargs)
    else:
        # Remove total_budget if it exists (safe removal)
        kwargs.pop("total_budget", None)
        return policy_no_incentives(**kwargs)


def main() -> None:
    """
    Run the MARL algorithm with or without incentives
    depending on the incentives_mode parameter.
    """
    # Load config
    config = ut.load_config(path="scripts/config.yaml")
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

    # Total budget for the incentives
    total_budget = config["total_budget"]
    # Dictionary with all paths
    paths_dict = config["paths_dict"]
    # Define edge data granularity
    edge_data_frequency = config["edge_data_frequency"]
    # Parameters to run SUMO
    sumo_params = config["sumo_config"]

    # Get available actions based on pre-computed routes
    actions_and_costs = get_actions(file_path=paths_dict["output_rou_alt_path"])
    # Unpack trips IDs
    trips_id = list(actions_and_costs.keys())

    ttts = []
    emissions_total = []

    # Initialise the Q-function
    q_values = (
        initialise_q_function_incentives(actions_costs=actions_and_costs)
        if incentives_mode
        else initialise_q_function_no_incentives(actions_costs=actions_and_costs)
    )

    # Train the RL agent
    for _ in range(episodes):
        # Select policy function based on whether incentives are used or not
        # Get actions from policy
        routes_edges, actions_index = policy_function(
            incentives_mode=incentives_mode,
            trips_id=trips_id,
            q=q_values,
            actions_costs=actions_and_costs,
            epsilon=hyperparams["epsilon"],
            total_budget=total_budget,
        )
        # Perform actions given by policy
        total_tt, ind_tt, ind_em, total_em = step(
            edge_data_frequency=edge_data_frequency,
            routes_edges=routes_edges,
            paths_dict=paths_dict,
            sumo_params=sumo_params,
        )

        ttts.append(total_tt)
        emissions_total.append(total_em * 30 + 300)

        # For each agent update Q function
        # Q(a) = (1 - alpha) * Q(a) + alpha * r
        for trip in trips_id:
            idx = actions_index[trip]
            # Compute reward
            reward = compute_reward(trip, ind_tt, ind_em, total_tt, total_em, weights)
            # Update Q-value
            q_values[trip][idx] = (1 - hyperparams["alpha"]) * q_values[trip][
                idx
            ] + hyperparams["alpha"] * reward

        hyperparams["epsilon"] = max(
            0.01, hyperparams["epsilon"] * hyperparams["decay"]
        )
        # Retrieve updated route costs
        # costs = calculate_route_cost(actions, parse_weights("data/weights.xml"))


# TODO(German): look at reward - only normalising some and TTT no
if __name__ == "__main__":
    main()
