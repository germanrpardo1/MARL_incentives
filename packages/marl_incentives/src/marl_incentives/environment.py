"""Module that represents the SUMO network"""

import subprocess
import sys

import sumolib

import marl_incentives.calculate_emissions as em
import marl_incentives.replay_buffer as rb
from marl_incentives import xml_manipulation as xmlm


class Network:
    """Class that represents the network for the simulation."""

    def __init__(
        self,
        paths_dict: dict,
        sumo_params: dict,
        edge_data_frequency: int,
        buffer_capacity: int = 100,
        batch_size: int = 32,
        state_mode: bool = False,
    ) -> None:
        """
        Constructor method for the Network class.

        :param paths_dict: Dict of paths (routes file, edge data, emissions, etc.)
        :param sumo_params: SUMO simulation configuration
        :param edge_data_frequency: Frequency of edge data (in seconds)
        :param buffer_capacity: Maximum size of the replay buffer.
        :param batch_size: Batch size for every sample taken from the buffer.
        :param state_mode: Whether using state or not.
        """
        self.paths_dict = paths_dict
        self.sumo_params = sumo_params
        self.edge_data_frequency = edge_data_frequency
        if not state_mode:
            self.buffer = rb.ReplayBuffer(
                capacity=buffer_capacity, batch_size=batch_size
            )
        else:
            self.buffer = rb.StateReplayBuffer(capacity=buffer_capacity)

    def run_simulation(self) -> None:
        """Run a SUMO simulation."""
        log_file = open(self.paths_dict["log_path"], "w+", encoding="utf-8")
        sys.stdout = sumolib.TeeFile(sys.__stdout__, log_file)

        log_file.flush()
        sys.__stdout__.flush()

        sumo_cmd = ["sumo", "-c", self.sumo_params["config_path"]]
        sumo_cmd = list(map(str, sumo_cmd))
        subprocess.call(sumo_cmd, stdout=log_file, stderr=log_file)

    def step(
        self,
        routes_edges: dict,
    ) -> tuple[float, dict, dict, float]:
        """
        Perform one SUMO simulation step:
        - Write route and configuration files
        - Run the SUMO simulation
        - Process travel time and emission outputs

        :param routes_edges: Route edges for each trip_id
        :return: Tuple of (total travel time, normalised individual travel times,
                          normalised individual emissions, normalised total emissions)
        """
        # Write input and configuration files
        xmlm.write_routes(routes_edges, self.paths_dict["routes_file_path"])
        xmlm.write_edge_data_config(
            self.paths_dict["edge_data_path"],
            self.paths_dict["edges_weights_path"],
            self.edge_data_frequency,
        )
        xmlm.write_sumo_config(**self.sumo_params)

        # Run SUMO simulation
        self.run_simulation()

        # Get travel times
        total_tt = xmlm.get_ttt(self.paths_dict["stats_path"])
        individual_tt = xmlm.get_individual_travel_times(
            self.paths_dict["trip_info_path"]
        )
        # Get emissions
        total_emissions, individual_emissions = em.co2_main(
            self.paths_dict["emissions_path"]
        )

        return total_tt, individual_tt, individual_emissions, total_emissions / 1000
