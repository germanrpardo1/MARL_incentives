"""
This module calculates the CO2 emissions for the fcd.xml file based on NGM dynamic model

The inputs are speed, acceleration, vehicle type and fuel type

The outputs are two txt file:
1- Emissions_per_second.txt contains the total emissions and the instantaneous emissions
2- Emissions_per_lane.txt contains the total emissions and the total emissions per lane
"""

from collections import defaultdict

from lxml import etree

from marl_incentives.co2modeler_v1 import co2modeler  # Import the CO2 modeler


def co2_main(path, vehicle_type="light_passenger", fuel="gasoline"):
    """
    Parse an XML file and calculate total and per-vehicle CO2 emissions.

    :param path: Path to the XML file containing vehicle data.
    :param vehicle_type: Type of vehicle to model ('light_passenger' by default).
    :param fuel: Type of fuel used ('gasoline' by default).
    :return: Tuple of (total emissions, dictionary of emissions per vehicle).
    """
    total_emissions = 0.0
    emissions_per_vehicle = defaultdict(float)
    model = co2modeler
    to_float = float

    with open(path, "rb") as f:
        for _, elem in etree.iterparse(f, tag="vehicle"):
            attrs = elem.attrib
            vehicle_id = attrs["id"]
            speed = to_float(attrs["speed"])
            acceleration = to_float(attrs["acceleration"])

            emission = model(speed, acceleration, vehicle_type, fuel)
            emissions_per_vehicle[vehicle_id] += emission
            total_emissions += emission

            elem.clear()  # Free memory

    return total_emissions, dict(emissions_per_vehicle)
