"""This script runs the MARL algorithm with and without incentives."""

import os

import torch
import torch.nn.functional as F
from marl_incentives import environment as env
from marl_incentives import traveller_with_budget_state as tr
from marl_incentives import utils as ut
from marl_incentives.experience_replay import ReplayBuffer


def save_metric(
    values: list[float],
    labels: dict,
    base_name: str,
    y_label: str,
    budget: int,
    weights: dict,
) -> None:
    """
    Save a plot and corresponding pickle file for a given metric using
        consistent naming conventions.

    :param values: List of metric values over time (e.g., per episode).
    :param labels: Dictionary of plot labels (e.g., title, y-axis label).
    :param base_name: The metric name (used in file naming and plot title).
    :param y_label: Y-axis label for the plot.
    :param budget: Total budget used in the experiment.
    :param weights: Dictionary of weights used in the experiment.
    """
    labels["title"] = f"{base_name.replace('_', ' ').title()} per episode"
    labels["y_label"] = y_label

    # Generate file paths
    pickle_path = ut.make_file_paths(base_name, "pickle_files", budget, weights, "pkl")
    plot_path = ut.make_file_paths(base_name, "plots", budget, weights, "png")

    # Ensure directories exist
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Save the plot and pickle
    ut.save_plot_and_file(
        values=values,
        labels=labels,
        window=30,
        path_to_pickle=pickle_path,
        path_to_plot=plot_path,
    )


def main(config, total_budget: int) -> None:
    """
    Run the MARL algorithm with or without incentives with experience replay.

    :param config: Configuration dictionary.
    :param total_budget: Total budget.
    """
    # Weights of the objective function
    weights = {
        "ttt": config["TTT_weight"],
        "individual_tt": config["individual_travel_time_weight"],
        "individual_emissions": config["emissions_weight"],
        "total_emissions": config["total_emissions_weight"],
    }

    # RL hyper-parameters
    hyperparams = {
        "epsilon": config["epsilon"],
        "decay": config["decay"],
        "alpha": config["alpha"],
    }

    # Dictionary with all paths
    paths_dict = config["paths_dict"]
    # Define edge data granularity
    edge_data_frequency = config["edge_data_frequency"]
    # Parameters to run SUMO
    sumo_params = config["sumo_config"]

    # Initialise all drivers
    drivers = tr.initialise_drivers(
        actions_file_path=paths_dict["output_rou_alt_path"],
        incentives_mode=config["incentives_mode"],
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
    )

    # Initialise replay buffer
    buffer = ReplayBuffer(capacity=100)

    # Train RL agent
    for i in range(config["episodes"]):
        # Get actions from policy based on whether incentives are used or not
        if config["incentives_mode"]:
            routes_edges, actions_index = tr.policy_incentives(
                drivers, total_budget=total_budget, epsilon=hyperparams["epsilon"]
            )
        else:
            routes_edges, actions_index = tr.policy_no_incentives(
                drivers, hyperparams["epsilon"]
            )

        # Perform actions given by policy
        total_tt, ind_tt, ind_em, total_em = network_env.step(
            routes_edges=routes_edges,
        )

        reward_tuple = [(60**2) * total_tt / 1100, ind_tt, ind_em, total_em]
        buffer.push(actions_index, reward_tuple)

        # Record TTT and total emissions throughout iterations
        ttts.append(total_tt)
        emissions_total.append(total_em)

        # For each agent update Q function
        # Q(a) = (1 - alpha) * Q(a) + alpha * r
        batch_size = 2
        # TODO(german): complete
        if len(buffer) >= batch_size:
            acts, rews = buffer.sample(batch_size)
            total_tt, ind_tt, ind_em, total_em = rews
            for driver in drivers:
                # Compute reward
                # Update Q-networks
                q_targets = torch.tensor(
                    [[ind_tt]], dtype=torch.float32, device=driver.device
                )
                # Calculate expected value from local network
                q_expected = driver.q_network_local(
                    torch.tensor(
                        [[driver.state]], dtype=torch.float32, device=driver.device
                    )
                ).gather(1, acts[driver.trip_id])
                # Loss calculation (we used Mean squared error)
                loss = F.mse_loss(q_expected, q_targets)
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
    save_metric(ttts, labels_dict, "exp_replay_ttt", "TTT [h]", total_budget, weights)
    save_metric(
        emissions_total,
        labels_dict,
        "exp_replay_emissions",
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
