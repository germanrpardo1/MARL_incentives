"""
This script runs the multi-agent Reinforcement Learning Q-Learning
with and without incentives (this can be changed in the config).
It uses experience replay to accelerate learning, and it has a
discrete state variable.
"""

from marl_incentives import environment as env
from marl_incentives import traveller as tr
from marl_incentives import utils as ut
from marl_incentives import xml_manipulation as xml
from marl_incentives.environment import Network
from marl_incentives.traveller import Driver


def experience_replay(network_env: Network, drivers: list[Driver], weights) -> None:
    """pass."""
    # Sample past observations from replay buffer
    states, acts, rewards = network_env.buffer.sample(network_env.buffer.batch_size)
    for s, a, r in zip(states, acts, rewards):
        # For each agent update Q function
        # Q(s, a) = (1 - alpha) * Q(s, a) + alpha * r
        network_env.buffer.update_q_values_discrete_state(
            drivers=drivers,
            state_index=s,
            action_index=a,
            reward=r,
            weights=weights,
        )


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
        incentives_mode=True,
        strategy=config["strategy"],
        state_variable=True,
    )

    ttts = []
    emissions_total = []
    current_used_budgets = []
    acceptance_rates = []
    labels_dict = {}

    # Instantiate network object
    network_env = env.Network(
        paths_dict=paths_dict,
        sumo_params=sumo_params,
        edge_data_frequency=edge_data_frequency,
        buffer_capacity=config["buffer_capacity"],
        batch_size=config["batch_size"],
        state_mode=True,
    )

    epsilon = hyperparams["epsilon"]
    decay = hyperparams["decay"]

    # Start training loop for RL agents
    for i in range(config["episodes"]):
        # Get action from policy for every driver
        routes_edges, actions_index, current_used_budget, acceptance_rate = (
            tr.policy_incentives_discrete_state(
                drivers=drivers,
                total_budget=total_budget,
                epsilon=epsilon,
                compliance_rate=config["compliance_rate"],
            )
        )
        acceptance_rates.append(acceptance_rate)
        current_used_budgets.append(current_used_budget)
        # Perform actions given by policy
        total_tt, ind_tt, ind_em, total_em = network_env.step(
            routes_edges=routes_edges,
        )

        reward_tuple = [(60**2) * total_tt / 1100, ind_tt, ind_em, total_em]
        states_tuple = {driver.trip_id: driver.state for driver in drivers}
        network_env.buffer.push(states_tuple, actions_index, reward_tuple)

        # Record TTT and total emissions throughout iterations
        ttts.append(total_tt)
        emissions_total.append(total_em)

        # If there are enough observations in the buffer, sample and update Qs
        if len(network_env.buffer) >= network_env.buffer.batch_size:
            experience_replay(network_env, drivers, weights)

        # Reduce epsilon
        epsilon = max(0.01, epsilon * decay)

        # Log progress
        ut.log_progress(i=i, episodes=config["episodes"], ttts=ttts)

        # Update travel times
        ut.update_average_travel_times(
            drivers=drivers, weights=xml.parse_weights("data/weights.xml")
        )

    # Save the plot and pickle file for TTT and emissions
    base_name = (
        "compliance_rate_exp_replay_binary_state"
        if config["compliance_rate"]
        else "exp_replay_binary_state"
    )
    ut.save_metric(
        ttts,
        labels_dict,
        base_name + "_ttt",
        "TTT [h]",
        total_budget,
        weights,
    )
    ut.save_metric(
        emissions_total,
        labels_dict,
        base_name + "_emissions",
        "Emissions [kg]",
        total_budget,
        weights,
    )
    ut.save_metric(
        current_used_budgets,
        labels_dict,
        base_name + "_used_budget",
        "Budget",
        total_budget,
        weights,
    )
    ut.save_metric(
        acceptance_rates,
        labels_dict,
        base_name + "_acceptance_rates",
        "Acceptance rates",
        total_budget,
        weights,
    )


if __name__ == "__main__":
    # Load config
    config_file = ut.load_config(path="scripts/qlearning_binary_state_exp_replay.yaml")

    # Loop for different budgets
    for tot_budget in config_file["total_budget"]:
        main(config=config_file, total_budget=tot_budget)
