"""Module that represents a single traveller."""

import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.optim as optim
from dqn_neural_network import DQN

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
        if action_index < num_routes:
            incentive = self.costs[action_index] - min(self.costs) + 1
            # Apply incentive to cost
            adjusted_costs[action_index] -= incentive
        else:
            incentive = 0

        # Choose the route with the minimum adjusted cost
        # Here, the route selection strategy should always be minimum cost
        # as we assume that travellers take the incentivised route deterministically
        # when participation rate is added, this will need modification
        selected_action = self.route_selection_strategy(
            strategy="argmin", costs=adjusted_costs
        )
        route_edges = self.routes[selected_action][1]

        return route_edges, selected_action, action_index, incentive

    def route_selection_strategy(self, strategy: str, costs) -> int:
        """
        Select the strategy for route selection when there is no budget left.

        :param strategy: Route selection strategy.
        :param costs: The costs of the routes after adjusting.
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

    :param drivers: List of objects of type StateDriver.
    :param epsilon: Epsilon parameter.
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


def initialise_drivers(
    actions_file_path: str,
    incentives_mode: bool,
    strategy: str,
) -> list[Driver]:
    """
    Initialise all the drivers of type StateDriver.

    :param actions_file_path: Path to the XML file.
    :param incentives_mode: Whether the incentives are applied.
    :param strategy: Strategy used to select routes.
    :return: A list of drivers of type StateDriver.
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


def policy_incentives(
    drivers: list[Driver],
    total_budget: float,
    epsilon: float,
    compliance_rate: bool = False,
) -> tuple[dict, dict]:
    """
    Policy function for the RL algorithm using epsilon-greedy strategy
        for when incentives are applied.

    :param drivers: List of objects of type StateDriver.
    :param total_budget: Maximum total incentive budget.
    :param epsilon: Probability to select a random action.
    :param compliance_rate: True if using compliance rate, False otherwise.
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

        # Set random number if using compliance rate
        if compliance_rate:
            if _rng.random() >= 0.8:
                selected_action = driver.route_selection_strategy(
                    driver.strategy, driver.costs
                )
                selected_edges = driver.routes[selected_action][1]
                incentive = 0

        # If there is no budget left, select route according to route strategy
        if current_budget + incentive > total_budget:
            selected_action = driver.route_selection_strategy(
                driver.strategy, driver.costs
            )
            selected_edges = driver.routes[selected_action][1]
            incentive = 0

        # Update budget and tracking dictionaries
        current_budget += incentive
        route_edges[driver.trip_id] = selected_edges
        actions_index[driver.trip_id] = selected_index

    return route_edges, actions_index


class StateDriver:
    """Class that represents a single driver."""

    def __init__(
        self,
        budget: float,
        trip_id: str,
        routes: list[tuple[int, list]],
        costs: list,
        strategy: str = "argmin",
    ) -> None:
        """
        Constructor method for a single driver.

        :param budget: Budget available.
        :param trip_id: The trip ID.
        :param routes: The routes of the driver. Each entry of the form (index, edges).
        :param costs: The costs of the routes.
        :param strategy: The strategy to assign routes.
        """

        self.trip_id = trip_id
        self.routes = routes
        self.costs = costs
        self.strategy = strategy

        self.state_size = 1
        self.action_size = len(self.costs) + 1

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state = torch.tensor([[budget]], dtype=torch.float32, device=self.device)

        # Q-Network
        self.q_network = DQN(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=5e-4)

    def eps_greedy_policy_incentives(
        self, epsilon: float
    ) -> tuple[list, int, int, float]:
        """
        Epsilon-greedy policy for selecting a route with incentives.

        :return: (route_edges, selected_action, action_index, applied_incentive)
        """
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(self.state).cpu().numpy().reshape(-1)
        self.q_network.train()

        num_routes = len(self.costs)

        # Perform random action with probability epsilon
        rnd = _rng.random()
        if rnd <= epsilon:
            random_int = _rng.integers(num_routes + 1)
            action_index = int(random_int)  # Random action index
        # Perform action with maximum Q-value with probability 1 - epsilon
        else:
            action_index = int(action_values.argmin())

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


def policy_incentives_state(
    drivers: list[StateDriver], total_budget: float, epsilon: float
) -> tuple[dict, dict]:
    """
    Policy function for the RL algorithm using epsilon-greedy strategy
        for when incentives are applied.

    :param drivers: List of objects of type StateDriver.
    :param total_budget: Maximum total incentive budget.
    :param epsilon: Probability to select a random action.
    :return: Tuple containing:
             - route_edges: mapping trip_id to selected route edges
             - actions_index: mapping trip_id to selected (route_idx, incentive_level)
    """
    route_edges = {}
    actions_index = {}
    current_used_budget = 0

    for i, driver in enumerate(drivers):
        if i != 0:
            # Update state from the incentive used by the previous driver
            new_state = total_budget - current_used_budget

            driver.state = torch.tensor(
                [[new_state]], dtype=torch.float32, device=driver.device
            )

        # Select action using epsilon-greedy strategy
        selected_edges, selected_action, selected_index, incentive = (
            driver.eps_greedy_policy_incentives(epsilon=epsilon)
        )

        # If there is no budget left, select route according to route strategy
        if current_used_budget + incentive > total_budget:
            selected_action = driver.route_selection_strategy(driver.strategy)
            selected_edges = driver.routes[selected_action][1]
            incentive = 0

        current_used_budget += incentive
        route_edges[driver.trip_id] = selected_edges
        actions_index[driver.trip_id] = selected_index

    return route_edges, actions_index


def initialise_drivers_state(
    actions_file_path: str,
    strategy: str,
    budget: float = None,
) -> list[StateDriver]:
    """
    Initialise all the drivers of type StateDriver.

    :param actions_file_path: Path to the XML file.
    :param strategy: Strategy used to select routes.
    :param budget: Budget for the traveller.
    :return: A list of drivers of type StateDriver.
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
            StateDriver(
                budget=budget,
                trip_id=vehicle.get("id"),
                routes=routes,
                costs=costs,
                strategy=strategy,
            )
        )

    return drivers
