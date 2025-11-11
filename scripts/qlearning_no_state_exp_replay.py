"""
This script runs the multi-agent Reinforcement Learning Q-Learning
with and without incentives (this can be changed in the config).
It uses experience replay to accelerate learning, and it does not have
a state variable.
"""

from marl_incentives import environment as env
from marl_incentives import traveller as tr
from marl_incentives import utils as ut
from marl_incentives import xml_manipulation as xml


def main(config, total_budget: int) -> None:
    """
    Run the MARL algorithm with or without incentives with experience replay.

    :param config: Configuration dictionary.
    :param total_budget: Total budget.
    """
    # Unpack configuration file
    weights, hyperparams, paths_dict, edge_data_frequency, sumo_params = (
        ut.unpack_config(config)
    )

    # Initialise all drivers
    drivers = tr.initialise_drivers(
        actions_file_path=paths_dict["output_rou_alt_path"],
        incentives_mode=config["incentives_mode"],
        strategy=config["strategy"],
    )

    ttts = []
    emissions_total = []
    labels_dict = {}

    # Instantiate network object
    network_env = env.Network(
        paths_dict=paths_dict,
        sumo_params=sumo_params,
        edge_data_frequency=edge_data_frequency,
        buffer_capacity=config["buffer_capacity"],
        batch_size=config["batch_size"],
    )

    # Start training loop for RL agents
    for i in range(config["episodes"]):
        # Get action from policy for every driver with incentives mode
        if config["incentives_mode"]:
            routes_edges, actions_index = tr.policy_incentives(
                drivers=drivers,
                total_budget=total_budget,
                epsilon=hyperparams["epsilon"],
                compliance_rate=config["compliance_rate"],
            )
        # Take action from policy for every driver without incentives mode
        else:
            routes_edges, actions_index = tr.policy_no_incentives(
                drivers, hyperparams["epsilon"]
            )

        # Perform actions given by policy
        total_tt, ind_tt, ind_em, total_em = network_env.step(
            routes_edges=routes_edges,
        )

        reward_tuple = [(60**2) * total_tt / 1100, ind_tt, ind_em, total_em]
        network_env.buffer.push(actions_index, reward_tuple)

        # Record TTT and total emissions throughout iterations
        ttts.append(total_tt)
        emissions_total.append(total_em)

        # If there are enough observations in the buffer, sample and update Qs
        if len(network_env.buffer) >= network_env.buffer.batch_size:
            # Sample past observations from replay buffer
            acts, rewards = network_env.buffer.sample(network_env.buffer.batch_size)
            for a, r in zip(acts, rewards):
                # For each agent update Q function
                # Q(a) = (1 - alpha) * Q(a) + alpha * r
                network_env.buffer.update_q_values(
                    drivers=drivers,
                    action_index=a,
                    reward=r,
                    weights=weights,
                    alpha=hyperparams["alpha"],
                )

                # Reduce epsilon
                hyperparams["epsilon"] = max(
                    0.01, hyperparams["epsilon"] * hyperparams["decay"]
                )

        # Log progress
        ut.log_progress(i=i, episodes=config["episodes"], ttts=ttts)

        # Update travel times
        ut.update_average_travel_times(
            drivers=drivers, weights=xml.parse_weights("data/weights.xml")
        )

    # Save the plot and pickle file for TTT and emissions
    base_name = (
        "compliance_rate_exp_replay" if config["compliance_rate"] else "exp_replay"
    )
    ut.save_metric(
        ttts, labels_dict, base_name + "_ttt", "TTT [h]", total_budget, weights
    )
    ut.save_metric(
        emissions_total,
        labels_dict,
        base_name + "_emissions",
        "Emissions [kg]",
        total_budget,
        weights,
    )


if __name__ == "__main__":
    # Load config
    config_file = ut.load_config(path="scripts/qlearning_no_state_exp_replay.yaml")

    # Loop for different budgets
    for tot_budget in config_file["total_budget"]:
        main(config=config_file, total_budget=tot_budget)
