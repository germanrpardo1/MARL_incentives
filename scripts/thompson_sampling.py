"""
This script runs the multi-agent Reinforcement Learning Thompson
sampling algorithm to solve the incentives' problem.
"""

import numpy as np
from marl_incentives import environment as env
from marl_incentives import traveller as tr
from marl_incentives import utils as ut
from marl_incentives import xml_manipulation as xml


def main(config, total_budget: int) -> None:
    """
    Run the MARL algorithm with or without incentives.

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
        thompson_sampling_method=True,
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
    )

    # Start training loop for RL agents
    for i in range(config["episodes"]):
        # Take action from policy for every driver with incentives mode
        routes_edges, actions_index, current_used_budget, acceptance_rate = (
            tr.policy_incentives(
                drivers=drivers,
                total_budget=total_budget,
                epsilon=hyperparams["epsilon"],
                compliance_rate=config["compliance_rate"],
                thompson_sampling=True,
            )
        )
        acceptance_rates.append(acceptance_rate)
        current_used_budgets.append(current_used_budget)

        # Perform actions for each driver
        total_tt, ind_tt, ind_em, total_em = network_env.step(
            routes_edges=routes_edges,
        )

        # Record TTT and total emissions throughout iterations
        ttts.append(total_tt)
        emissions_total.append(total_em)

        # Update Q function for each agent
        for driver in drivers:
            if i == 500:
                driver.estimated_stds = np.full(len(driver.costs) + 1, 1)
            # Unpack actions index
            idx = actions_index[driver.trip_id]
            # Compute reward
            reward = driver.compute_reward(ind_tt, ind_em, total_tt, total_em, weights)

            # Update mean estimate based on the reward
            driver.estimated_means[idx] = driver.estimated_means[idx] + (
                reward - driver.estimated_means[idx]
            ) / (driver.action_counts[idx] + 1)

        # Log progress
        ut.log_progress(i=i, episodes=config["episodes"], ttts=ttts)

        if not config["upper_confidence_bound"]:
            # Reduce epsilon
            hyperparams["epsilon"] = max(
                0.01, hyperparams["epsilon"] * hyperparams["decay"]
            )

        # Update travel times
        ut.update_average_travel_times(
            drivers=drivers, weights=xml.parse_weights("data/weights.xml")
        )

    # Save the plot and pickle file for TTT and emissions
    base_name = (
        "compliance_rate_thompson_sampling"
        if config["compliance_rate"]
        else "thompson_sampling"
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
    config_file = ut.load_config(path="scripts/thompson_sampling.yaml")

    # Loop for different budgets
    for tot_budget in config_file["total_budget"]:
        main(config=config_file, total_budget=tot_budget)
