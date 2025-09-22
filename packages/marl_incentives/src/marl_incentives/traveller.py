"""Module that represents a single traveller."""

import xml.etree.ElementTree as ET

import numpy as np

# Initialise a random generator instance
_rng = np.random.default_rng()


class Driver:
    """Class that represents a single driver."""

    def __init__(
        self,
        trip_id: str,
        routes: list[tuple[int, list]],
        costs: list,
        incentives_mode: bool,
        strategy: str = "argmin",
    ) -> None:
        """
        Constructor method for a single driver.

        :param trip_id: The trip ID.
        :param routes: The routes of the driver. Each entry of the form (index, edges).
        :param costs: The costs of the routes.
        :param incentives_mode: Whether the driver should include incentives or not.
        :param strategy: The strategy to assign routes.
        """
        self.trip_id = trip_id
        self.routes = routes
        self.costs = costs
        self.strategy = strategy
        # Initialise the Q-table
        self.q_values = (
            # Q table for incentives mode
            np.zeros((len(self.costs) + 1))
            if incentives_mode
            # Q table for no incentives
            else np.zeros(len(self.costs))
        )

    def eps_greedy_policy_no_incentives(self, epsilon: float) -> tuple[list, int]:
        """
        Epsilon-greedy policy for selecting a route without incentives.

        :return: (route_edges, selected_action, action_index)
        """
        num_routes = len(self.costs)
        # Perform random action with probability epsilon
        if _rng.random() <= epsilon:
            random_int = _rng.integers(num_routes)
            route_idx = int(random_int)  # Random action index
        # Perform action with maximum Q-value with probability 1 - epsilon
        else:
            route_idx = np.argmin(self.q_values)

        # Get route edges to write them in the .XML file
        route_edges = self.routes[route_idx][1]
        return route_edges, route_idx

    def eps_greedy_policy_incentives(
        self, epsilon: float
    ) -> tuple[list, int, int, float]:
        """
        Epsilon-greedy policy for selecting a route with incentives.

        :return: (route_edges, selected_action, action_index, applied_incentive)
        """
        num_routes = len(self.costs)

        # Perform random action with probability epsilon
        if _rng.random() <= epsilon:
            random_int = _rng.integers(num_routes + 1)
            action_index = int(random_int)  # Random action index
        # Perform action with maximum Q-value with probability 1 - epsilon
        else:
            action_index = np.argmin(self.q_values)

        adjusted_costs = self.costs.copy()
        # Compute incentive to apply
        if action_index + 1 <= num_routes:
            incentive = self.costs[action_index] - min(self.costs) + 1
            # Apply incentive to cost
            adjusted_costs[action_index] -= incentive
        else:
            incentive = 0

        # Choose the route with the minimum adjusted cost
        # Here, the route selection strategy should always be minimum cost
        # as we assume that travellers take the incentivised route deterministically
        # when participation rate is added, this will need modification
        selected_action = self.route_selection_strategy(strategy="argmin")
        route_edges = self.routes[selected_action][1]

        return route_edges, selected_action, action_index, incentive

    def route_selection_strategy(self, strategy: str) -> int:
        """
        Select the strategy for route selection when there is no budget left.

        :param strategy: Route selection strategy.
        :return: Route index.
        """
        costs = np.array(self.costs)

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

        raise ValueError(f"Unknown strategy: {self.strategy}")

    def compute_reward(
        self,
        ind_tt: dict,
        ind_em: dict,
        total_tt: float,
        total_em: float,
        weights: dict,
    ) -> float:
        """
        Compute the multi-objective reward.

        :param ind_tt: Individual travel times.
        :param ind_em: Individual emissions.
        :param total_tt: Total travel time.
        :param total_em: Total emissions.
        :param weights: Weight of incentives.
        :return: Multi-objective reward.
        """
        return (
            weights["individual_tt"] * ind_tt[self.trip_id]
            + weights["ttt"] * total_tt
            + weights["individual_emissions"] * ind_em[self.trip_id]
            + weights["total_emissions"] * total_em
        )


def policy_no_incentives(drivers: list[Driver], epsilon: float) -> tuple[dict, dict]:
    """
    Policy function for the RL algorithm using epsilon-greedy strategy
        for when no incentives are applied.

    :param drivers: List of objects of type Driver.
    :return: Tuple containing:
             - route_edges: mapping trip_id to selected route edges
             - actions_index: mapping trip_id to selected (route_idx, incentive_level)
    """
    route_edges = {}
    actions_index = {}
    for driver in drivers:
        # Select action using epsilon-greedy strategy
        selected_edges, selected_index = driver.eps_greedy_policy_no_incentives(
            epsilon=epsilon
        )

        route_edges[driver.trip_id] = selected_edges
        actions_index[driver.trip_id] = selected_index

    return route_edges, actions_index


def policy_incentives(
    drivers: list[Driver], total_budget: float, epsilon: float
) -> tuple[dict, dict]:
    """
    Policy function for the RL algorithm using epsilon-greedy strategy
        for when incentives are applied.

    :param drivers: List of objects of type Driver.
    :param total_budget: Maximum total incentive budget
    :return: Tuple containing:
             - route_edges: mapping trip_id to selected route edges
             - actions_index: mapping trip_id to selected (route_idx, incentive_level)
    """
    route_edges = {}
    actions_index = {}
    current_budget = 0

    for driver in drivers:
        # Select action using epsilon-greedy strategy
        selected_edges, selected_action, selected_index, incentive = (
            driver.eps_greedy_policy_incentives(epsilon=epsilon)
        )

        # If there is no budget left, select route according to route strategy
        if current_budget + incentive > total_budget:
            selected_action = driver.route_selection_strategy(driver.strategy)
            selected_edges = driver.routes[selected_action][1]
            incentive = 0

        # Update budget and tracking dictionaries
        current_budget += incentive
        route_edges[driver.trip_id] = selected_edges
        actions_index[driver.trip_id] = selected_index

    return route_edges, actions_index


def initialise_drivers(
    actions_file_path: str, incentives_mode: bool, strategy: str, epsilon: float
) -> list[Driver]:
    """
    Initialise all the drivers of type Driver.

    :param actions_file_path: Path to the XML file.
    :param incentives_mode: Whether the incentives are applied.
    :param strategy: Strategy used to select routes.
    :param epsilon: The epsilon parameter for RL.
    :return: A list of drivers of type Driver.
    """
    drivers = []

    tree = ET.parse(actions_file_path)
    root = tree.getroot()

    # Loop through all the vehicles
    for vehicle in root.findall("vehicle"):
        routes = []
        costs = []
        route_distribution = vehicle.find("routeDistribution")
        if route_distribution is not None:
            # Loop through all routes for the given vehicle
            for i, route in enumerate(route_distribution.findall("route")):
                routes.append((i, route.get("edges", "").split()))
                costs.append(float(route.get("cost", "0")))

        drivers.append(
            Driver(
                trip_id=vehicle.get("id"),
                routes=routes,
                costs=costs,
                strategy=strategy,
                incentives_mode=incentives_mode,
            )
        )

    return drivers
