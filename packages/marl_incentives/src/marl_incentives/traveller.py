"""Module that represents a single traveller."""

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
from cfgv import Array

# Initialise a random generator instance
_rng = np.random.default_rng()


class Drivers:
    """Class that represents the drivers."""

    def __init__(self, actions_file_path: str) -> None:
        """
        Constructor method for the Driver class.

        :param actions_file_path: Path to the XML file.
        """
        self.actions_file_path = actions_file_path

    def get_actions(self) -> dict:
        """
        Parse an XML file and extract available vehicle routes and their costs.

        :return: A dictionary mapping vehicle IDs to lists of (index, edge list) tuples
            and costs
        """
        tree = ET.parse(self.actions_file_path)
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

    # Perform random action with probability epsilon
    if _rng.random() <= epsilon:
        random_int = _rng.integers(num_routes)
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

    # Perform random action with probability epsilon
    if _rng.random() <= epsilon:
        random_int = _rng.integers(num_routes + 1)
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
    # Here, the route selection strategy should always be minimum cost
    # as we assume that travellers take the incentivised route deterministically
    # when participation rate is added, this will need modification
    selected_action = route_selection_strategy(costs=adjusted_costs, strategy="argmin")
    route_edges = routes[selected_action][1]

    return route_edges, selected_action, action_index, incentive


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


def route_selection_strategy(costs: list, strategy: str = "argmin") -> int:
    """
    Select the strategy for route selection when there is no budget left.

    :param costs: Costs corresponding to all the routes of a given trip_id.
    :param strategy: Strategy used to select routes.
    :return: Route index.
    """

    costs = np.array(costs)

    if strategy == "argmin":
        return int(np.argmin(costs))

    if strategy == "prob_distribution":
        probs = costs / np.sum(costs)
        return int(_rng.choice(len(costs), p=probs))

    if strategy == "logit":
        # Inverse utility (lower cost -> higher probability)
        exp_utilities = np.exp(-costs / max(costs))
        probs = exp_utilities / np.sum(exp_utilities)
        return int(_rng.choice(len(costs), p=probs))

    raise ValueError(f"Unknown strategy: {strategy}")


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
    strategy: str,
) -> tuple[dict, dict]:
    """
    Policy function for the RL algorithm using epsilon-greedy strategy
        for when incentives are applied.

    :param trips_id: List of trip IDs
    :param q: Q-function mapping trip_id to Q-value array
    :param actions_costs: Dictionary mapping trip_id to (routes, costs)
    :param epsilon: Probability of choosing a random action
    :param total_budget: Maximum total incentive budget
    :param strategy: Strategy used to select routes.

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

        # If there is no budget left, select route according to route strategy
        if current_budget + incentive > total_budget:
            selected_action = route_selection_strategy(costs=costs, strategy=strategy)
            selected_edges = routes[selected_action][1]
            incentive = 0

        # Update budget and tracking dictionaries
        current_budget += incentive
        route_edges[trip_id] = selected_edges
        actions_index[trip_id] = selected_index

    return route_edges, actions_index


def policy_function(incentives_mode: bool, **kwargs: Any) -> tuple[dict, dict]:
    """
    Select the policy based on whether incentives are applied or not.

    :param incentives_mode: True if incentives are used, False otherwise.
    :param kwargs: Keyword arguments passed to policy function.
    :return: Policy function.
    """
    if incentives_mode:
        return policy_incentives(**kwargs)

    # Remove total_budget and strategy if they exist
    kwargs.pop("total_budget", None)
    kwargs.pop("strategy", None)
    return policy_no_incentives(**kwargs)


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
