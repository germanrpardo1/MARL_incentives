import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from marl_incentives import traveller, utils
from marl_incentives.replay_buffer import StateReplayBuffer
from marl_incentives.sumo_surrogate import SurrogateModel

WEIGHTS = {
    "individual_tt": 1.0,
    "ttt": 0.0,
    "individual_emissions": 0.0,
    "total_emissions": 0.0,
}


class CoreFixTests(unittest.TestCase):
    def test_speed_rewards_match_paper_definitions(self):
        speeds = {"slow": 1.0, "middle_a": 2.0, "middle_b": 3.0, "fast": 4.0}
        self.assertEqual(
            traveller.compute_speed_reward("fast", speeds, "speed_relative"), 1.5
        )
        self.assertEqual(
            traveller.compute_speed_reward("fast", speeds, "speed_percentile"), 5.0
        )
        self.assertEqual(
            traveller.compute_speed_reward("slow", speeds, "speed_percentile"),
            -1.0,
        )

    def test_binary_replay_uses_each_drivers_stored_state(self):
        drivers = [
            traveller.Driver("a", [(0, ["edge"])], [1.0], True, True),
            traveller.Driver("b", [(0, ["edge"])], [1.0], True, True),
        ]
        StateReplayBuffer.update_q_values_discrete_state(
            drivers=drivers,
            state_index=[0, 1],
            action_index={"a": 0, "b": 0},
            reward=[0.0, {"a": 2.0, "b": 3.0}, {"a": 0.0, "b": 0.0}, 0.0],
            weights=WEIGHTS,
            alpha=1.0,
        )
        self.assertEqual(drivers[0].q_values[0, 0], 2.0)
        self.assertEqual(drivers[1].q_values[1, 0], 3.0)
        self.assertEqual(drivers[0].q_values[1, 0], 0.0)

    def test_global_seed_repeats_numpy_and_torch_draws(self):
        utils.set_global_seed(7)
        first = (np.random.random(), torch.rand(1).item())
        utils.set_global_seed(7)
        second = (np.random.random(), torch.rand(1).item())
        self.assertEqual(first, second)

    def test_surrogate_dimensions_are_data_driven(self):
        model = SurrogateModel(num_agents=3, max_actions=4, hidden_dim=8)
        output = model(torch.zeros((2, 3), dtype=torch.long))
        self.assertEqual(tuple(output.shape), (2, 4))
        with self.assertRaises(ValueError):
            model(torch.zeros((2, 4), dtype=torch.long))

    def test_run_config_separates_generated_files(self):
        config = {
            "seed": 9,
            "paths_dict": {"output_rou_alt_path": "data/input.xml"},
            "sumo_config": {
                "network_path": "data/network.xml",
                "routes_path": "data/old-routes.xml",
                "config_path": "data/old.sumocfg",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            old_cwd = Path.cwd()
            try:
                os.chdir(directory)
                prepared = utils.prepare_run_config(config, "example", 100)
            finally:
                os.chdir(old_cwd)

        self.assertEqual(
            prepared["paths_dict"]["output_rou_alt_path"], "data/input.xml"
        )
        self.assertIn("results\\runs\\example\\budget_100", prepared["run_dir"])
        self.assertEqual(prepared["sumo_config"]["seed"], 9)


if __name__ == "__main__":
    unittest.main()
