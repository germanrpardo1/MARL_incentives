"""This script runs the incentives MARL algorithm."""

import sys
from pathlib import Path

import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import subprocess
import sumolib
import yaml
from cfgv import Array

from co2Emissions.xmlreader import co2_main


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
    write_routes(routes_edges, paths_dict["routes_file_path"])
    write_edge_data_config(paths_dict["edge_data_path"], edge_data_frequency)
    write_sumo_config(**sumo_params)

    # Run SUMO simulation
    run_simulation(paths_dict["log_path"], sumo_params["config_path"])

    # Process outputs
    total_tt = get_ttt(paths_dict["stats_path"])
    individual_tt = normalise_dict(
        get_individual_travel_times(paths_dict["trip_info_path"])
    )
    individual_emissions = normalise_dict(
        calculate_emissions_per_vehicle(paths_dict["emissions_per_vehicle_path"])
    )
    total_emissions = normalise_scalar(
        min_val=300, max_val=330, val=co2_main(paths_dict["emissions_path"]) / 1000
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


def write_sumo_config(
    config_path: str,
    network_path: str,
    routes_path: str,
    edges_weights_path: str,
    edge_frequency_path: str,
) -> None:
    """
    Write SUMO configuration file.

    :param config_path: Path to the configuration file.
    :param network_path: Path to the network file.
    :param routes_path: Path to the routes file.
    :param edges_weights_path: Path to the edges weights file.
    :param edge_frequency_path: Path to the edge frequency file.
    """
    sumo_cmd = [
        "sumo",
        "-n",
        network_path,
        "-r",
        routes_path,
        "--save-configuration",
        config_path,
        "--edgedata-output",
        str(edges_weights_path),
        "--tripinfo-output",
        str(Path("data") / "tripinfo.xml"),
        "--log",
        str(Path("data") / "log.xml"),
        "--no-step-log",
        "--additional-files",
        edge_frequency_path,
        "--begin",
        "0",
        "--route-steps",
        "200",
        "--time-to-teleport",
        "300",
        "--time-to-teleport.highways",
        "0",
        "--no-internal-links",
        "False",
        "--eager-insert",
        "False",
        "--verbose",
        "True",
        "--no-warnings",
        "True",
        "--statistic-output",
        str(Path("data") / "stats.xml"),
        "--fcd-output",
        str(Path("data") / "fcd.xml"),
        "--fcd-output.acceleration",
    ]

    subprocess.call(sumo_cmd, stdout=subprocess.PIPE)


def write_edge_data_config(filename: str, freq: int) -> None:
    """
    Write config for edge data granularity.

    :param filename: Path to the output file.
    :param freq: Edge data granularity.
    """
    # Create the root element
    root = ET.Element("a")

    # edgeData element
    ET.SubElement(
        root,
        "edgeData",
        {
            "id": "edge_data",
            "freq": str(freq),
            "excludeEmpty": "True",
            "minSamples": "1",
        },
    )
    # Create the XML tree
    ET.ElementTree(root)
    # Convert to string
    xml_str = ET.tostring(root, encoding="unicode")
    # Parse the string with minidom for pretty printing
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="    ")
    # Write to file
    with open(filename, "w", encoding="utf-8") as file:
        file.write(pretty_xml_str)


def write_routes(routes_edges: dict, file: str = "data/output.rou.xml") -> None:
    """
    Write all the routes to a .XML file.

    :param routes_edges: Dictionary containing all the edges for every trip
    :param file: Path to the output file
    """
    # Create the root element
    routes_element = ET.Element("routes")
    routes_element.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    routes_element.set(
        "xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd"
    )

    # Add the vType element
    vtype_element = ET.SubElement(routes_element, "vType")
    vtype_element.set("id", "type1")
    vtype_element.set("length", "5.00")
    vtype_element.set("maxSpeed", "40.00")
    vtype_element.set("accel", "0.4")
    vtype_element.set("decel", "4.8")
    vtype_element.set("sigma", "0.5")

    # Add vehicles and their routes to the XML
    counter = 0
    for vehicle_id, edges in routes_edges.items():
        vehicle_element = ET.SubElement(routes_element, "vehicle")
        vehicle_element.set("id", vehicle_id)
        vehicle_element.set("type", "type1")
        vehicle_element.set(
            "depart", str(0.09 * counter)
        )  # Modify departure time as needed

        route_element = ET.SubElement(vehicle_element, "route")
        route_element.set("edges", " ".join(edges))

        counter += 1
    # Convert the ElementTree to a string
    xml_str = ET.tostring(routes_element, "utf-8")

    # Prettify the XML string
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="    ")

    # Write to a .rou.xml file
    with open(file, "w") as f:
        f.write(pretty_xml_str)


def get_ttt(file="data/stats.xml"):
    """
    Get total travel time from the stats file.

    :param file: Path to the stats file
    :return: Total travel time in hours.
    """
    with open(file, "rb") as f:
        root = ET.parse(f).getroot()
    return float(root.find("vehicleTripStatistics").get("totalTravelTime")) / (60**2)


def get_individual_travel_times(file="data/tripinfo.xml") -> dict:
    """
    Get individual travel times from the tripinfo file.

    :param file: Path to the tripinfo file
    :return: Individual travel times for each trip_id.
    """
    # Parse the XML data
    tree = ET.parse(file)
    root = tree.getroot()

    # Initialize an empty dictionary to store id and duration
    trip_durations = {}

    # Iterate through each trip_info element
    for trip_info in root.findall("tripinfo"):
        trip_id = trip_info.get("id")
        duration = trip_info.get("duration")
        if trip_id and duration:
            trip_durations[trip_id] = float(duration)
    return trip_durations


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


def calculate_emissions_per_vehicle(file_path="data/emissions_per_vehicle.txt") -> dict:
    """
    Calculate emissions per vehicle after SUMO simulation.

    :param file_path: Path to the emissions file
    :return: Emissions per vehicle.
    """
    vehicle_emission = {}

    # Read the TXT file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Skip the header and process each line
    for line in lines[1:]:
        # Validate line format
        if ";" in line:
            parts = line.strip().split(";")
            if len(parts) == 2:
                vehicle, emission = parts
                try:
                    vehicle_emission[vehicle] = float(emission)
                except ValueError:
                    pass
                    # print(f"Skipping invalid emission value in line: {line.strip()}")
            else:
                pass
                # print(f"Skipping malformed line: {line.strip()}")
        else:
            pass
            # print(f"Skipping malformed line: {line.strip()}")

    return vehicle_emission


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


def normalise_dict(dict_to_normalise: dict) -> dict:
    """
    Normalise the values in the dictionary.

    :param dict_to_normalise: Dictionary for which the values will be normalised.
    :return: Normalised dictionary.
    """
    min_v, max_v = min(dict_to_normalise.values()), max(dict_to_normalise.values())
    return {
        k: (v - min_v) / (max_v - min_v) if max_v != min_v else 0
        for k, v in dict_to_normalise.items()
    }


def normalise_scalar(min_val: float, max_val: float, val: float) -> float:
    """
    Normalise a given scalar value.

    :param min_val: Minimum scalar value.
    :param max_val: Maximum scalar value.
    :param val: Scalar value to normalise.

    :return: Normalised scalar value.
    """
    return (val - min_val) / (max_val - min_val)


def main_incentives():
    with open("scripts/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    incentives_mode = config["incentives_mode"]

    # Number of episodes for the RL algorithm
    episodes = config["episodes"]

    # Setting weights of the objective function
    ttt_weight = config["TTT_weight"]
    individual_travel_time_weight = config["individual_travel_time_weight"]
    individual_emissions_weight = config["emissions_weight"]
    total_emissions_weight = config["total_emissions_weight"]

    # RL hyper-parameters
    epsilon = config["epsilon"]
    decay = config["decay"]
    alpha = config["alpha"]

    # Setting parameters for budget
    total_budget = config["total_budget"]

    # Paths for all relevant files
    paths_dict = config["paths_dict"]

    # Frequency of the edge data
    edge_data_frequency = config["edge_data_frequency"]

    # Parameters to run SUMO
    sumo_params = config["sumo_config"]

    # Getting available actions based on pre-computed routes
    actions_and_costs = get_actions(file_path=paths_dict["output_rou_alt_path"])

    # Unpack all the trip IDs for future use
    trips_id = list(actions_and_costs.keys())

    ttts = []
    emissions_total = []

    # Initialise the Q-function
    if incentives_mode:
        q_values = initialise_q_function_incentives(actions_costs=actions_and_costs)
    else:
        q_values = initialise_q_function_no_incentives(actions_costs=actions_and_costs)

    ## vehicle_id -> [(index, edges), costs]
    # Start training the agent
    for _ in range(episodes):
        if incentives_mode:
            # Policy function decides the next actions to take
            routes_edges, actions_index = policy_incentives(
                trips_id=trips_id,
                q=q_values,
                actions_costs=actions_and_costs,
                epsilon=epsilon,
                total_budget=total_budget,
            )
        else:
            # Policy function decides the next actions to take
            routes_edges, actions_index = policy_no_incentives(
                trips_id=trips_id,
                q=q_values,
                actions_costs=actions_and_costs,
                epsilon=epsilon,
            )

        # Run a simulation to evaluate the actions selected
        (
            total_travel_time,
            individual_travel_times,
            individual_emissions,
            tot_emission,
        ) = step(
            edge_data_frequency=edge_data_frequency,
            routes_edges=routes_edges,
            paths_dict=paths_dict,
            sumo_params=sumo_params,
        )

        ttts.append(total_travel_time)
        emissions_total.append(tot_emission * 30 + 300)

        # For each agent update Q function
        # Q(a) = Q(a) + alpha * (r - Q(a))
        for trip in trips_id:
            index = actions_index[trip]

            # Compute reward
            reward = (
                individual_travel_time_weight * individual_travel_times[trip]
                + ttt_weight * total_travel_time
                + individual_emissions_weight * individual_emissions[trip]
                + total_emissions_weight * tot_emission
            )

            # Update Q-value
            q_values[trip][index] = (1 - alpha) * q_values[trip][index] + alpha * reward

        epsilon = max(0.01, epsilon * decay)

        # Retrieve updated route costs
        # costs = calculate_route_cost(actions, parse_weights("data/weights.xml"))


if __name__ == "__main__":
    main_incentives()
