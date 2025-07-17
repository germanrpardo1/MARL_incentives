"""Module that represents the SUMO network"""

import subprocess
import sys

import co2Emissions.calculate_emissions as em
import sumolib
import utils.utils as ut


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
    ut.write_routes(routes_edges, paths_dict["routes_file_path"])
    ut.write_edge_data_config(
        paths_dict["edge_data_path"],
        paths_dict["edges_weights_path"],
        edge_data_frequency,
    )
    ut.write_sumo_config(**sumo_params)

    # Run SUMO simulation
    run_simulation(paths_dict["log_path"], sumo_params["config_path"])

    # Process outputs
    total_tt = ut.get_ttt(paths_dict["stats_path"])
    individual_tt = ut.normalise_dict(
        ut.get_individual_travel_times(paths_dict["trip_info_path"])
    )
    individual_emissions = ut.normalise_dict(
        em.calculate_emissions_per_vehicle(paths_dict["emissions_per_vehicle_path"])
    )
    total_emissions = ut.normalise_scalar(
        min_val=300, max_val=330, val=em.co2_main(paths_dict["emissions_path"]) / 1000
    )

    return total_tt, individual_tt, individual_emissions, total_emissions
