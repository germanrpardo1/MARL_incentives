"""
This module calculates the CO2 emissions for the fcd.xml file based on NGM dynamic model

The inputs are speed, acceleration, vehicle type and fuel type

The outputs are two txt file:
1- Emissions_per_second.txt contains the total emissions and the instantaneous emissions
2- Emissions_per_lane.txt contains the total emissions and the total emissions per lane
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from lxml import etree

from marl_incentives.co2modeler_v1 import co2modeler  # Import the CO2 modeler


def co2_main(
    path: str | Path,
    vehicle_type: str = "light_passenger",
    fuel: str = "gasoline",
    include_speeds: bool = False,
) -> tuple[float, dict[str, float]] | tuple[float, dict[str, float], dict[str, float]]:
    """
    Parse an XML file and calculate total and per-vehicle CO2 emissions.

    :param path: Path to the XML file containing vehicle data.
    :param vehicle_type: Type of vehicle to model ('light_passenger' by default).
    :param fuel: Type of fuel used ('gasoline' by default).
    :param include_speeds: Whether to include mean speed for each vehicle.
    :return: Total and per-vehicle emissions, plus per-vehicle mean speeds when
        ``include_speeds`` is true.
    """
    total_emissions = 0.0
    emissions_per_vehicle = defaultdict(float)
    speed_sums = defaultdict(float)
    speed_counts = defaultdict(int)
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
            speed_sums[vehicle_id] += speed
            speed_counts[vehicle_id] += 1

            elem.clear()  # Free memory

    result = total_emissions, dict(emissions_per_vehicle)
    if not include_speeds:
        return result

    average_speeds = {
        vehicle_id: speed_sums[vehicle_id] / count
        for vehicle_id, count in speed_counts.items()
    }
    return *result, average_speeds
