# This code is to calculate the CO2 emissions for the fcd.xml file based on NGM dynamic model

# The inputs are speed, acceleration, vehicle type and fuel type

# The outputs are two txt file:
# 1- Emissions_per_second.txt contains the total emissions and the instantaneous emissions
# 2- Emissions_per_lane.txt contains the total emissions and the total emissions per lane

from collections import defaultdict
from xml.etree.ElementTree import iterparse

from marl_incentives.co2modeler_v1 import co2modeler  # Import the CO2 modeler


def co2_main(path):
    total_emissions = 0.0
    emissions_per_vehicle = defaultdict(float)

    type_passenger = "light_passenger"
    fuel = "gasoline"

    float_ = float
    model = co2modeler

    for event, elem in iterparse(path, events=("end",)):
        if elem.tag == "vehicle":
            get_attr = elem.get
            vehicle_id = get_attr("id")
            speed = float_(get_attr("speed"))
            acceleration = float_(get_attr("acceleration"))

            emission = float_(model(speed, acceleration, type_passenger, fuel))
            emissions_per_vehicle[vehicle_id] += emission
            total_emissions += emission

            elem.clear()  # Free memory

    return total_emissions, dict(emissions_per_vehicle)
