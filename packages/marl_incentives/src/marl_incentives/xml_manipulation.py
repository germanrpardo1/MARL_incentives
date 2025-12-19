"""Module that provides relevant functions to manipulate .XML files and configs for SUMO"""

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom


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


def write_sumo_config(
    config_path: str,
    network_path: str,
    routes_path: str,
) -> None:
    """
    Write SUMO configuration file.

    :param config_path: Path to the configuration file.
    :param network_path: Path to the network file.
    :param routes_path: Path to the routes file.
    """
    # This is the directory your script is in
    script_dir = Path(__file__).resolve().parent.parent
    # Navigate up to the project root and into 'data'
    ### Fix
    data_path = script_dir.parent.parent.parent / "data" / "edge_data.add.xml"
    config_path = script_dir.parent.parent.parent / config_path
    network_path = script_dir.parent.parent.parent / network_path
    routes_path = script_dir.parent.parent.parent / routes_path

    sumo_cmd = [
        "sumo",
        "-n",
        network_path,
        "-r",
        routes_path,
        "--save-configuration",
        config_path,
        "--edgedata-output",
        data_path,
        "--tripinfo-output",
        str(Path("data") / "tripinfo.xml"),
        "--log",
        str(Path("data") / "log.xml"),
        "--no-step-log",
        "--additional-files",
        data_path,
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


def write_edge_data_config(filename: str, weights_path: str, freq: int) -> None:
    """
    Write config_file for edge data granularity.

    :param filename: Path to the output file.
    :param weights_path: Path to the weights file.
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
            "file": weights_path,
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


def parse_weights(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    weights = {}

    for interval in root.findall(".//interval"):
        begin, end = float(interval.get("begin")), float(interval.get("end"))
        for edge in interval.findall(".//edge"):
            edge_id = edge.get("id")
            travel_time = edge.get("traveltime")
            if edge_id not in weights:
                weights[edge_id] = []
            if travel_time:
                weights[edge_id].append((begin, end, float(travel_time)))

    return weights
