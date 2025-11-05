"""
This script runs the multi-agent Reinforcement Learning DQN
with incentives.
It is an adaptation of DQN for one-step MDPs, so there is no target network.
It uses experience replay to accelerate learning, and the available budget
is the state variable.
"""

import numpy as np
import torch
import torch.nn.functional as F
from marl_incentives import environment as env
from marl_incentives import traveller as tr
from marl_incentives import utils as ut


def main(config, total_budget: int) -> None:
    """
    Run the MARL algorithm with incentives using DQN.

    :param config: Configuration dictionary.
    :param total_budget: Total budget.
    """
    # Unpack configuration file
    weights, hyperparams, paths_dict, edge_data_frequency, sumo_params = (
        ut.unpack_config(config)
    )

    # Initialise all drivers
    drivers = tr.initialise_drivers_state(
        actions_file_path=paths_dict["output_rou_alt_path"],
        strategy=config["strategy"],
        budget=total_budget,
    )

    ttts = []
    emissions_total = []
    labels_dict = {}

    # Instantiate network object
    network_env = env.Network(
        paths_dict=paths_dict,
        sumo_params=sumo_params,
        edge_data_frequency=edge_data_frequency,
        state_mode=True,
    )

    # Train RL agent
    for i in range(config["episodes"]):
        # Get actions from policy
        routes_edges, actions_index = tr.policy_incentives_state(
            drivers, total_budget=total_budget, epsilon=hyperparams["epsilon"]
        )

        # Perform actions given by policy
        total_tt, ind_tt, _, total_em = network_env.step(
            routes_edges=routes_edges,
        )

        reward_tuple = list(ind_tt.values())
        actions_tuple = list(actions_index.values())
        states_tuple = [driver.state for driver in drivers]
        network_env.buffer.push(states_tuple, actions_tuple, reward_tuple)

        # Record TTT and total emissions throughout iterations
        ttts.append(total_tt)
        emissions_total.append(total_em)

        # If there are enough observations, perform a gradient step
        if len(network_env.buffer) >= network_env.buffer.batch_size:
            # Sample from replay buffer
            states, actions, rewards = network_env.buffer.sample(
                network_env.buffer.batch_size
            )

            # Transpose so we get per-driver lists
            rewards_all_drivers = [list(x) for x in zip(*rewards)]
            actions_all_drivers = [list(x) for x in zip(*actions)]
            states_all_drivers = [list(x) for x in zip(*states)]

            for j, driver in enumerate(drivers):
                rewards_per_driver = rewards_all_drivers[j]
                actions_per_driver = actions_all_drivers[j]
                states_per_driver = states_all_drivers[j]

                # Rewards: [batch_size, 1]
                q_targets = torch.tensor(
                    rewards_per_driver, dtype=torch.float32, device=driver.device
                ).unsqueeze(1)

                # Convert list of numpy arrays to a single numpy array first (faster)
                state_batch = np.array(states_per_driver, dtype=np.float32)

                # Convert to tensor
                state_batch = torch.tensor(
                    state_batch, dtype=torch.float32, device=driver.device
                )

                # If shape is [batch_size, 1, state_dim], squeeze unnecessary dims
                state_batch = state_batch.view(
                    len(states_per_driver), -1
                )  # [batch_size, state_dim]

                # Ensure batch dimension exists
                if state_batch.dim() == 1:  # single state
                    state_batch = state_batch.unsqueeze(0)  # shape [1, state_dim]

                # Forward pass
                q_values = driver.q_network(
                    state_batch
                )  # should be [batch_size, num_actions]

                # Convert actions to tensor
                actions_tensor = torch.tensor(
                    actions_per_driver, dtype=torch.int64, device=driver.device
                )

                # Ensure actions tensor is 2D: [batch_size, 1]
                if actions_tensor.dim() == 1:
                    actions_tensor = actions_tensor.unsqueeze(1)

                # Now gather works
                q_expected = q_values.gather(1, actions_tensor)

                # Loss calculation (Mean Squared Error between predicted Q and target Q)
                loss = F.mse_loss(q_expected, q_targets)

                # Gradient step
                driver.optimizer.zero_grad()
                loss.backward()
                driver.optimizer.step()

            # Reduce epsilon
            hyperparams["epsilon"] = max(
                0.01, hyperparams["epsilon"] * hyperparams["decay"]
            )

        # Logging
        ut.log_progress(
            i=i, episodes=config["episodes"], hyperparams=hyperparams, ttts=ttts
        )

    # Save the plot and pickle file for TTT and emissions
    ut.save_metric(
        ttts, labels_dict, "continuous_state_ttt", "TTT [h]", total_budget, weights
    )
    ut.save_metric(
        emissions_total,
        labels_dict,
        "continuous_state_emissions",
        "Emissions [kg]",
        total_budget,
        weights,
    )


if __name__ == "__main__":
    # Load config
    config_file = ut.load_config(path="scripts/run_incentives_budget_state.yaml")

    # Loop for different budgets
    for tot_budget in config_file["total_budget"]:
        main(config=config_file, total_budget=tot_budget)
