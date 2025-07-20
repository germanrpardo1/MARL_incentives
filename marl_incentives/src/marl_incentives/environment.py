"""Module that represents the SUMO network"""

import subprocess
import sys

import co2Emissions.calculate_emissions as em
import sumolib
from utils import utils as ut
from utils import xml_manipulation as xmlm


class Network:
    def __init__(
        self,
        paths_dict: dict,
        sumo_params: dict,
    ):
        """
        Constructor method for the Network class.

        :param paths_dict: Dict of paths (routes file, edge data, emissions, etc.)
        :param sumo_params: SUMO simulation configuration
        """

        self.paths_dict = paths_dict
        self.sumo_params = sumo_params

    def run_simulation(self) -> None:
        """Run a SUMO simulation."""
        sys.stdout = sumolib.TeeFile(
            sys.stdout, open(self.paths_dict["log_path"], "w+")
        )
        log = open(self.paths_dict["log_path"], "w+")

        log.flush()
        sys.stdout.flush()

        sumo_cmd = ["sumo", "-c", self.sumo_params["config_path"]]

        sumo_cmd = list(map(str, sumo_cmd))
        subprocess.call(sumo_cmd, stdout=log, stderr=log)

    def step(
        self,
        edge_data_frequency: int,
        routes_edges: dict,
    ) -> tuple[float, dict, dict, float]:
        """
        Perform one SUMO simulation step:
        - Write route and configuration files
        - Run the SUMO simulation
        - Process travel time and emission outputs

        :param edge_data_frequency: Frequency of edge data (in seconds)
        :param routes_edges: Route edges for each trip_id
        :return: Tuple of (total travel time, normalised individual travel times,
                          normalised individual emissions, normalised total emissions)
        """
        # Write input and configuration files
        xmlm.write_routes(routes_edges, self.paths_dict["routes_file_path"])
        xmlm.write_edge_data_config(
            self.paths_dict["edge_data_path"],
            self.paths_dict["edges_weights_path"],
            edge_data_frequency,
        )
        xmlm.write_sumo_config(**self.sumo_params)

        # Run SUMO simulation
        self.run_simulation()

        # Process outputs
        total_tt = xmlm.get_ttt(self.paths_dict["stats_path"])
        individual_tt = ut.normalise_dict(
            xmlm.get_individual_travel_times(self.paths_dict["trip_info_path"])
        )
        individual_emissions = ut.normalise_dict(
            em.calculate_emissions_per_vehicle(
                self.paths_dict["emissions_per_vehicle_path"]
            )
        )
        total_emissions = ut.normalise_scalar(
            min_val=300,
            max_val=330,
            val=em.co2_main(self.paths_dict["emissions_path"]) / 1000,
        )

        return total_tt, individual_tt, individual_emissions, total_emissions
