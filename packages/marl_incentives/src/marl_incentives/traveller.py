"""Module that represents a single traveller."""

import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.optim as optim

from marl_incentives import utils as ut
from marl_incentives.dqn_neural_network import DQN

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
        state_variable: bool = False,
        strategy: str = "argmin",
        thompson_sampling_method: bool | None = None,
    ) -> None:
        """
        Constructor method for a single driver.

        :param trip_id: The trip ID.
        :param routes: The routes of the driver. Each entry of the form (index, edges).
        :param costs: The costs of the routes.
        :param incentives_mode: True if the driver should include incentives, False otherwise.
        :param state_variable: True if using state variable, False otherwise.
        :param strategy: The strategy to assign routes.
        :param thompson_sampling_method: True if using Thompson sampling, False otherwise.
        """
        self.trip_id = trip_id
        self.routes = routes
        self.costs = costs
        self.strategy = strategy
        self.action_counts = np.zeros(len(self.costs))
        self.t = 0

        self.estimated_means = np.append(np.array(costs), min(costs))
        self.estimated_stds = np.full(len(self.costs) + 1, 20)

        if not thompson_sampling_method:
            # Initialise the Q-table
            if incentives_mode and state_variable:
                self.q_values = np.zeros((2, len(self.costs) + 1))
                self.state = 0
                self.action_counts = np.zeros(len(self.costs) + 1)
            elif incentives_mode and not state_variable:
                self.q_values = np.zeros((len(self.costs) + 1))
                self.action_counts = np.zeros(len(self.costs) + 1)
            elif not incentives_mode and state_variable:
                self.q_values = np.zeros((2, len(self.costs)))
                self.state = 0
                self.action_counts = np.zeros(len(self.costs))
            else:
                self.q_values = np.zeros(len(self.costs))
                self.action_counts = np.zeros(len(self.costs))
        else:
            self.action_counts = np.zeros(len(self.costs) + 1)
            self.means = costs
            self.stds = np.ones(len(self.costs))
            self.reward_stds = np.ones(len(self.costs))

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

        :param epsilon: Probability of taking a random action.
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

    def upper_confidence_bound(self, c: float = 200) -> tuple[list, int, int, float]:
        """pass."""
        # upper_confidence_bounds = [q_values[action] + c * np.sqrt(np.log(iteration + 1) / (num_invocations[action])) if num_invocations[action] > 0 else np.inf for action in actions]
        # return np.random.choice([action_ for action_, value_ in enumerate(upper_confidence_bounds) if value_ == np.max(upper_confidence_bounds)])
        num_routes = len(self.costs)

        # Calculate upper confidence bound
        ucb = self.q_values + c * np.sqrt(np.log(self.t) / (self.action_counts + 1e-9))

        # Perform action with minimum Q-value + UCB
        action_index = int(np.argmin(ucb))

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

    def thompson_sampling(self) -> tuple[list, int, int, float]:
        """pass."""

        samples_travel_times = np.random.normal(
            loc=self.estimated_means, scale=self.estimated_stds
        )
        action_index = int(np.argmin(samples_travel_times))
        num_routes = len(self.costs)

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

    def eps_greedy_policy_incentives_discrete_state(
        self, epsilon: float, n: int
    ) -> tuple[list, int, int, float]:
        """
        Epsilon-greedy policy for selecting a route with incentives.

        :param epsilon: Probability of taking a random action.
        :param n: Current discrete state of the driver - available budget.
        :return: (route_edges, selected_action, action_index, applied_incentive)
        """
        num_routes = len(self.costs)

        # Perform random action with probability epsilon
        if _rng.random() <= epsilon:
            random_int = _rng.integers(num_routes + 1)
            action_index = int(random_int)  # Random action index
        # Perform action with maximum Q-value with probability 1 - epsilon
        else:
            action_index = np.argmin(self.q_values[n])

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

    def update_average_travel_time(self, weights: dict):
        """
        Compute the average travel time per route from all available routes.

        :param weights: Dictionary containing all the edge's travel times.
        """
        avg_travel_times = []

        for route_id, route in self.routes:
            total_edge_average = 0
            for edge in route:
                total_time = 0
                count = 0
                for _, _, travel_time in weights.get(edge, []):
                    total_time += travel_time if travel_time else 0
                    count += 1 if travel_time else 0
                total_edge_average += total_time / count
            # Avoid division by zero
            avg_travel_times.append(
                round(total_edge_average, 2) if total_edge_average else 0
            )

        self.costs = avg_travel_times


def policy_no_incentives(drivers: list[Driver], epsilon: float) -> tuple[dict, dict]:
    """
    Policy function for the RL algorithm using epsilon-greedy strategy
        for when no incentives are applied.

    :param drivers: List of objects of type DQNStateDriver.
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
    state_variable: bool = False,
    thompson_sampling_method: bool | None = None,
) -> list[Driver]:
    """
    Initialise all the drivers of type DQNStateDriver.

    :param actions_file_path: Path to the XML file.
    :param incentives_mode: Whether the incentives are applied.
    :param strategy: Strategy used to select routes.
    :param state_variable: True if using state variable, False otherwise.
    :param thompson_sampling_method: True if using thompson sampling. False otherwise.
    :return: A list of drivers of type DQNStateDriver.
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
                    state_variable=state_variable,
                    thompson_sampling_method=thompson_sampling_method,
                )
            )
    return drivers


def select_default_route(driver: Driver):
    """
    Select route based on the driver's default strategy.

    :param driver: Driver object.
    """
    idx = driver.route_selection_strategy(driver.strategy, driver.costs)
    return idx, driver.routes[idx][1], 0.0


def policy_incentives(
    drivers: list[Driver],
    total_budget: float,
    epsilon: float | None = None,
    compliance_rate: bool = False,
    upper_confidence_bound: bool = False,
    thompson_sampling: bool = False,
) -> tuple[dict[str, list], dict[str, tuple], float, float]:
    """
    Apply an epsilon-greedy policy for route and incentive selection.

    :param drivers: List of Driver objects.
    :param total_budget: Maximum total incentive budget.
    :param epsilon: Probability of selecting a random action.
    :param compliance_rate: Whether to simulate compliance rate randomness.
    :param upper_confidence_bound: Whether to use UCB for action selection.
    :param thompson_sampling: Whether to do Thompson sampling for action selection.
    :return:
        route_edges: mapping trip_id → selected route edges
        actions_index: mapping trip_id → (route_idx, incentive_level)
        current_used_budget: Budget effectively used
    """
    route_edges = {}
    actions_index = {}
    current_used_budget = 0.0
    total_accepted_paths = 0
    total_incentivised_paths = 0

    for driver in drivers:
        num_routes = len(driver.costs)
        path_accepted = True
        # --- Step 1: Base action via epsilon-greedy or UCB ---
        if upper_confidence_bound:
            edges, _, index, incentive = driver.upper_confidence_bound()
        elif thompson_sampling:
            edges, _, index, incentive = driver.thompson_sampling()
        else:
            edges, _, index, incentive = driver.eps_greedy_policy_incentives(epsilon)

        # --- Step 2: Apply compliance rate randomness ---
        # Only applies for when incentives are assigned
        if index < len(driver.costs) and compliance_rate:
            # Hard-coding coefficients of the logit model
            coefficients = [1.99, -0.23]
            # Only feature at the moment: time sacrifice in minutes
            x = [(driver.costs[index] - min(driver.costs)) / 60]  # In minutes
            # Probability of accepting incentivised path
            prob = ut.logistic_prob(x, coefficients)
            prob = 0.8
            if _rng.random() >= prob:
                # Route not accepted, select shortest path
                _, edges, incentive = select_default_route(driver)
                path_accepted = False

        # --- Step 3: Enforce budget limit ---
        if current_used_budget + incentive > total_budget:
            _, edges, incentive = select_default_route(driver)
            path_accepted = False

        # --- Step 4: Update trackers ---
        current_used_budget += incentive
        route_edges[driver.trip_id] = edges
        actions_index[driver.trip_id] = index
        if index < num_routes:
            total_incentivised_paths += 1
            if path_accepted:
                total_accepted_paths += 1

        if upper_confidence_bound or thompson_sampling:
            driver.action_counts[index] += 1

    return (
        route_edges,
        actions_index,
        current_used_budget,
        total_accepted_paths / total_incentivised_paths,
    )


def policy_incentives_discrete_state(
    drivers: list[Driver],
    total_budget: float,
    epsilon: float,
    compliance_rate: bool = False,
) -> tuple[dict[str, list], dict[str, tuple]]:
    """
    Apply an epsilon-greedy policy for route and incentive selection.

    :param drivers: List of Driver objects.
    :param total_budget: Maximum total incentive budget.
    :param epsilon: Probability of selecting a random action.
    :param compliance_rate: Whether to simulate compliance rate randomness.
    :return:
        route_edges: mapping trip_id → selected route edges
        actions_index: mapping trip_id → (route_idx, incentive_level)
    """

    route_edges = {}
    actions_index = {}
    current_used_budget = 0.0

    for driver in drivers:
        remaining = total_budget - current_used_budget
        ratio_left = remaining / total_budget

        if ratio_left <= 0.20:
            driver.state = 0
        else:
            driver.state = 1

        state = total_budget - current_used_budget
        n = int(np.floor((state / total_budget) * 2) - 1)
        # driver.state = n
        # --- Step 1: Base action via epsilon-greedy ---
        edges, _, index, incentive = driver.eps_greedy_policy_incentives_discrete_state(
            epsilon, n
        )

        # --- Step 2: Apply compliance rate randomness ---
        # Only applies for when incentives are assigned
        if index < len(driver.costs) and compliance_rate:
            # Hard-coding coefficients of the logit model
            coefficients = [1.99, -0.23]
            # Only feature at the moment: time sacrifice in minutes
            x = [(driver.costs[index] - min(driver.costs)) / 60]  # In minutes
            # Probability of accepting incentivised path
            prob = ut.logistic_prob(x, coefficients)
            if _rng.random() >= prob:
                # Route not accepted, select shortest path
                _, edges, incentive = select_default_route(driver)

        # --- Step 3: Enforce budget limit ---
        if current_used_budget + incentive > total_budget:
            _, edges, incentive = select_default_route(driver)

        # --- Step 4: Update trackers ---
        current_used_budget += incentive
        route_edges[driver.trip_id] = edges
        actions_index[driver.trip_id] = index

    return route_edges, actions_index


class DQNStateDriver:
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


def policy_incentives_dqn_state(
    drivers: list[DQNStateDriver], total_budget: float, epsilon: float
) -> tuple[dict, dict]:
    """
    Policy function for the RL algorithm using epsilon-greedy strategy
        for when incentives are applied.

    :param drivers: List of objects of type DQNStateDriver.
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


def initialise_drivers_dqn_state(
    actions_file_path: str,
    strategy: str,
    budget: float = None,
) -> list[DQNStateDriver]:
    """
    Initialise all the drivers of type DQNStateDriver.

    :param actions_file_path: Path to the XML file.
    :param strategy: Strategy used to select routes.
    :param budget: Budget for the traveller.
    :return: A list of drivers of type DQNStateDriver.
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
            DQNStateDriver(
                budget=budget,
                trip_id=vehicle.get("id"),
                routes=routes,
                costs=costs,
                strategy=strategy,
            )
        )

    return drivers
