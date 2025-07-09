"""This script runs the incentives MARL algorithm."""

import sys
from pathlib import Path

import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import subprocess
import sumolib
import copy
import yaml
from co2Emissions.xmlreader import co2_main
import pickle


# TODO(German): Handle for when budget=False (or check it's handled)
def eps_greedy_policy(q, trip_id, actions_costs, n_incentives, epsilon=0.1):
    """
    Epsilon greedy policy.

    :param q: Q-function
    :param trip_id: Trip id for which action is selected
    :param actions_costs: Routes and costs for each trip
    :param n_incentives: Number of incentives to select
    :param epsilon: Epsilon value for random action selection
    :return: Route edges, action selected, action index and incentive value.
    """
    if np.random.uniform() <= epsilon:
        # Select random route and whether to incentivise or not
        action_index = (
            np.random.randint(len(actions_costs[trip_id][1])),
            np.random.randint(n_incentives),
        )
    else:
        # Select action with highest Q-value
        action_index = np.unravel_index(np.argmax(q[trip_id]), np.shape(q[trip_id]))

    if action_index[1] > 0:
        incentive = (
            actions_costs[trip_id][1][action_index[0]]
            - min(actions_costs[trip_id][1])
            + 1
        )
    else:
        incentive = 0

    # Subtract incentive from original cost for selected route
    actions_costs[trip_id][1][action_index[0]] -= incentive
    # Select route after incentives
    action = np.argmin(actions_costs[trip_id][1])
    route_edges = actions_costs[trip_id][0][action][1]
    return route_edges, action, action_index, incentive


def step(routes_edges):
    write_routes(routes_edges)
    write_edge_data_config(
        filename="data/edge_data.add.xml", freq=500, file="edge_data.add.xml"
    )
    write_sumo_config(
        filename="data/config.sumocfg",
        net_file="data/kamppi.net.xml",
        route_files="data/output.rou.xml",
        weight_file="data/weights.xml",
        additional_files="edge_data.add.xml",
    )
    run_simulation()
    tot_emission = co2_main("data/fcd.xml")

    return (
        (get_TTT() / (60**2)),
        normalise_dict(travelTimes()),
        normalise_dict(emissions_func()),
        normalise_scalar(300, 330, tot_emission / 1000),
    )


def get_actions(file_path: str) -> dict:
    """
    Parse an XML file and extract available vehicle routes and their costs.

    :param file_path: Path to the XML file.
    :return:
        - A dictionary mapping vehicle IDs to lists of (index, edge list) tuples
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


def run_simulation():
    sys.stdout = sumolib.TeeFile(sys.stdout, open("data/log.xml", "w+"))
    log = open("data/log.xml", "w+")

    log.flush()
    sys.stdout.flush()

    sumo_cmd = ["sumo", "-c", "data/config.sumocfg"]

    sumo_cmd = list(map(str, sumo_cmd))
    subprocess.call(sumo_cmd, stdout=log, stderr=log)


def write_sumo_config(filename, net_file, route_files, weight_file, additional_files):
    sumo_cmd = [
        "sumo",
        "-n",
        net_file,
        "-r",
        route_files,
        "--save-configuration",
        filename,
        "--edgedata-output",
        str(weight_file),
        "--tripinfo-output",
        str(Path("data") / "tripinfo.xml"),
        "--log",
        str(Path("data") / "log.xml"),
        "--no-step-log",
        "--additional-files",
        additional_files,
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


def write_edge_data_config(filename, freq, file):
    # Create the root element
    root = ET.Element("a")

    # edgeData element
    ET.SubElement(
        root,
        "edgeData",
        {
            "id": "edge_data",
            "freq": str(freq),
            "file": file,
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


def write_routes(vehicle_routes, file="data/output.rou.xml"):
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
    for vehicle_id, edges in vehicle_routes.items():
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


def get_TTT(file="data/stats.xml"):
    tree = ET.parse(file)
    root = tree.getroot()

    vehicle_trip_stats = root.find("vehicleTripStatistics")
    total_travel_time = float(vehicle_trip_stats.get("totalTravelTime"))
    return total_travel_time


def travelTimes(file="data/tripinfo.xml"):
    # Parse the XML data
    tree = ET.parse(file)
    root = tree.getroot()

    # Initialize an empty dictionary to store id and duration
    trip_durations = {}

    # Iterate through each tripinfo element
    for tripinfo in root.findall("tripinfo"):
        trip_id = tripinfo.get("id")
        duration = tripinfo.get("duration")
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


def emissions_func(file_path="data/emissions_per_vehicle.txt"):
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


def initialise_q_function(actions_costs: dict, incentives_n: int) -> dict:
    """
    Initialise the Q-function.

    :param actions_costs: dictionary that stores the possible routes for each
        trip and their cost
    :param incentives_n: number of incentives options
    :return: initialised Q-function for every trip.
    """

    return {
        trip: np.zeros((len(action_cost[0]), incentives_n))
        for trip, action_cost in actions_costs.items()
    }


def policy(
    trips_id: list,
    q: dict,
    actions_costs: dict,
    n_incentives: int,
    epsilon: float,
    total_budget: float,
    *,
    budget: bool,
) -> tuple[dict, dict]:
    """
    Policy function for the RL algorithm.

    :param trips_id: list of trip IDs
    :param q: Q-function
    :param actions_costs: dictionary that stores the possible routes and costs
    :param n_incentives: number of incentives options
    :total_budget: total budget
    :param budget: True if using incentives, False otherwise
    :return: Routes' edges and actions' indexes
    """
    route_edges = {}
    action = {}
    action_index = {}
    b = 0
    for trip in trips_id:
        route_edges[trip], action[trip], action_index[trip], incentive = (
            eps_greedy_policy(q, trip, actions_costs, n_incentives, epsilon=epsilon)
        )
        if budget:
            if (b + incentive) <= total_budget:
                b += incentive
            else:
                action_index[trip] = (action_index[trip][0], 0)
                action[trip] = np.argmin(actions_costs[trip][1])
                route_edges[trip] = actions_costs[trip][0][action[trip]][1]

    return route_edges, action_index


def prob_calc(costs, theta=-0.01):
    sum_ = sum(np.exp(i * theta) for i in costs)
    probs = [np.exp(cost * theta) / (sum_) for cost in costs]
    return probs


def normalise_dict(d):
    min_v, max_v = min(d.values()), max(d.values())
    return {
        k: (v - min_v) / (max_v - min_v) if max_v != min_v else 0 for k, v in d.items()
    }


def normalise_scalar(min, max, val):
    return (val - min) / (max - min)


def running_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


# Set the window size (for running average)
window_size = 20  # Choose a window size


def main():
    with open("scripts/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Number of episodes for the RL algorithm
    episodes = config["episodes"]

    # Setting weights of the objective function
    ttt_weight = config["TTT_weight"]
    individual_travel_time_weight = config["individual_travel_time_weight"]
    emissions_weight = config["emissions_weight"]
    total_emissions_weight = config["total_emissions_weight"]

    n_incentives = config["n_incentives"]

    # RL hyper-parameters
    epsilon = config["epsilon"]
    decay = config["decay"]
    alpha = config["alpha"]

    # Setting parameters for budget
    total_budget = config["B"]
    budget = config["budget"]

    # Path for output_rou_alt
    output_rou_alt_path = config["output_rou_alt_path"]

    # Getting available actions based on pre-computed routes
    actions_and_costs = get_actions(file_path=output_rou_alt_path)

    # Unpack all the trip IDs for future use
    trips_id = list(actions_and_costs.keys())

    ttts = []
    emissions_total = []

    # Initialise the Q-function
    q = initialise_q_function(
        actions_costs=actions_and_costs, incentives_n=n_incentives
    )
    ## vehicle_id -> [(index, edges), costs]
    # Start training the agent
    for ep in range(episodes):
        routes_edges, action_index = policy(
            trips_id=trips_id,
            q=q,
            actions_costs=actions_and_costs,
            n_incentives=n_incentives,
            epsilon=epsilon,
            total_budget=total_budget,
            budget=budget,
        )

        # Run a simulation to evaluate the actions selected
        ttt, ind_travel_times, emissions, tot_emission = step(routes_edges=routes_edges)

        ttts.append(ttt)
        emissions_total.append(tot_emission * 30 + 300)

        # For each agent update Q function
        # Q(a) = Q(a) + alpha * (r - Q(a))
        for trip in trips_id:
            r = -(
                individual_travel_time_weight * ind_travel_times[trip]
                + ttt_weight * ttt
                + emissions_weight * emissions[trip]
                + total_emissions_weight * tot_emission
            )

            q[trip][action_index[trip]] = (1 - alpha) * q[trip][
                action_index[trip]
            ] + alpha * r

        epsilon = max(0.01, epsilon * decay)

        # Retrieve updated route costs
        # TODO(German): fix line below
        # costs = calculate_route_cost(actions, parse_weights("data/weights.xml"))


if __name__ == "__main__":
    main()
