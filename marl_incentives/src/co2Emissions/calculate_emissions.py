# This code is to calculate the CO2 emissions for the fcd.xml file based on NGM dynamic model

# The inputs are speed, acceleration, vehicle type and fuel type

# The outputs are two txt file:
# 1- Emissions_per_second.txt contains the total emissions and the instantaneous emissions
# 2- Emissions_per_lane.txt contains the total emissions and the total emissions per lane

import xml.etree.ElementTree as ET
from co2Emissions.co2modeler_v1 import co2modeler  # Import the CO2 modeler


# if __name__ == "__main__":
def co2_main(path):
    # current_directory = os.path.dirname(__file__)
    xml_file = path  # os.path.join(current_directory, 'fcd.xml') # chagne this file to the dieserd xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    total_emissions = 0  # reset the variables
    emissions_per_vehicle = {}

    for timestep in root.findall(".//timestep"):
        subtotal_emission = 0  # reset the instantaneous emissions
        for vehicle in timestep.findall(".//vehicle"):  # Get the vehicles informations
            # type = (vehicle.get('type'))         # uncomment this one if you have the vehicle type in the xml file
            type = "light_passenger"  # comment this one if you have the vehicle type in the xml file
            # fuel = (vehicle.get('fuel'))         # uncomment this one if you have the fuel type in the xml file
            fuel = (
                "gasoline"  # comment this one if you have the fuel type in the xml file
            )
            speed = float(vehicle.get("speed"))  # Get the vehicles speed
            acceleration = float(
                vehicle.get("acceleration")
            )  # Get the vehicles acceleration
            emission = float(
                co2modeler(speed, acceleration, type, fuel)
            )  # Calculate the emissions assuming gasoline passenger vehicle

            if (
                vehicle.get("id") not in emissions_per_vehicle
            ):  # Create key for each lane
                emissions_per_vehicle[vehicle.get("id")] = 0
            emissions_per_vehicle[vehicle.get("id")] += emission
            # lane = vehicle.get('lane')  # Get the lane
            # if lane not in emissions_per_lane: # Create key for each lane
            #    emissions_per_lane[lane] = 0

            # emissions_per_lane[lane] += emission # Sort emissions per lane

            subtotal_emission += (
                emission  # Add the emissions to the instantaneous emissions
            )

        total_emissions += (
            subtotal_emission  # Add the instantaneous emissions to the total emissions
        )
        # emission_per_second.append((time,subtotal_emission))

    # Write the instantaneous emissions
    # with open(output_file,'w') as file:
    #    file.write(f"Total emissions: {total_emissions}\n \n")
    #    file.write("time(s); Emission \n")
    #    for time,emission in emission_per_second:
    #        file.write(f"{time};{emission}\n")

    # Write the emissions per lane
    # with open(output_file_2,'w') as file:
    #    file.write(f"Total emissions: {total_emissions}\n \n")
    #    file.write("lane; Emission \n")
    #    for lane,emission in emissions_per_lane.items():
    #        file.write(f"{lane};{emission}\n")

    # Write the vehicle emissions
    with open("emissions_per_vehicle.txt", "w") as file:
        file.write(f"Total emissions: {total_emissions}\n \n")
        file.write("Vehicle; Emission \n")
        for vehicle, emission in emissions_per_vehicle.items():
            file.write(f"{vehicle};{emission}\n")

    return total_emissions


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
