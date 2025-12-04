"""
This script runs the multi-agent Reinforcement Learning Thompson
sampling algorithm to solve the incentives' problem.
"""

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
        if config["incentives_mode"]:
            routes_edges, actions_index, current_used_budget, tot_accepted_paths = (
                tr.policy_incentives(
                    drivers=drivers,
                    total_budget=total_budget,
                    epsilon=hyperparams["epsilon"],
                    compliance_rate=config["compliance_rate"],
                    thompson_sampling=True,
                )
            )
        # Take action from policy for every driver without incentives mode
        else:
            routes_edges, actions_index = tr.policy_no_incentives(
                drivers=drivers, epsilon=hyperparams["epsilon"]
            )

        # Perform actions for each driver
        total_tt, ind_tt, ind_em, total_em = network_env.step(
            routes_edges=routes_edges,
        )

        # Record TTT and total emissions throughout iterations
        ttts.append(total_tt)
        emissions_total.append(total_em)

        # Update Q function for each agent
        for driver in drivers:
            idx = actions_index[driver.trip_id]

            # Compute reward
            reward = driver.compute_reward(ind_tt, ind_em, total_tt, total_em, weights)

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
    ut.save_metric(ttts, labels_dict, "ttt", "TTT [h]", total_budget, weights)
    ut.save_metric(
        emissions_total,
        labels_dict,
        "emissions",
        "Emissions [kg]",
        total_budget,
        weights,
    )


if __name__ == "__main__":
    # Load config
    config_file = ut.load_config(path="scripts/qlearning_no_state.yaml")

    # Loop for different budgets
    for tot_budget in config_file["total_budget"]:
        main(config=config_file, total_budget=tot_budget)
