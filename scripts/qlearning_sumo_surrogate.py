"""
This script runs the multi-agent Reinforcement Learning Q-Learning
with and without incentives (this can be changed in the config).
It uses experience replay to accelerate learning, and it does not have
a state variable.
"""

from marl_incentives import traveller as tr
from marl_incentives import utils as ut
from marl_incentives import xml_manipulation as xml
from marl_incentives.environment import Network
from sumo_surrogate import SurrogateModel, SimulatorDataset
import torch


def pre_train(surrogate: SurrogateModel, config, DEVICE, drivers, network_env):
    surrogate_dataset = SimulatorDataset(
        drivers=drivers,
        network_env=network_env,
        num_samples=config.get("surrogate_dataset_size", 10),
    )

    surrogate.set_base_dataset(
        surrogate_dataset.X,
        surrogate_dataset.Y,
    )

    train_loader = torch.utils.data.DataLoader(
        surrogate_dataset,
        batch_size=config.get("surrogate_batch_size", 32),
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        surrogate.parameters(),
        lr=config.get("surrogate_lr", 1e-3),
    )

    mse = torch.nn.MSELoss()

    print("Pretraining surrogate...")

    surrogate.train()

    for epoch in range(config.get("surrogate_pretrain_epochs", 10)):
        total_loss = 0.0

        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_Y = batch_Y.to(DEVICE)

            pred = surrogate(batch_X)

            pred_individual = pred[:, : len(drivers)]
            pred_total = pred[:, -1]

            true_individual = batch_Y[:, : len(drivers)]
            true_total = batch_Y[:, -1]

            loss_individual = mse(
                pred_individual,
                true_individual,
            )

            loss_total = mse(
                pred_total,
                true_total,
            )

            loss = loss_individual + 5.0 * loss_total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"[Surrogate Pretrain] Epoch {epoch + 1:02d} | Loss = {avg_loss:.6f}")

    print("Surrogate pretraining complete.")
    return surrogate


def main(config: dict, total_budget: int) -> None:
    """
    Run the MARL algorithm with or without incentives with experience replay.

    :param config: Configuration dictionary.
    :param total_budget: Total budget.
    """
    # Unpack configuration file
    weights, hyperparams, paths_dict, edge_data_frequency, sumo_params = (
        ut.unpack_config(config)
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DYNA_STEPS = config.get("dyna_steps", 20)

    SURROGATE_RETRAIN_FREQ = config.get(
        "surrogate_retrain_freq",
        25,
    )

    SURROGATE_RETRAIN_EPOCHS = config.get(
        "surrogate_retrain_epochs",
        2,
    )

    # Initialise all drivers
    drivers = tr.initialise_drivers(
        actions_file_path=paths_dict["output_rou_alt_path"],
        incentives_mode=config["incentives_mode"],
        strategy=config["strategy"],
    )

    ttts = []
    emissions_total = []
    current_used_budgets = []
    acceptance_rates = []
    labels_dict = {}

    # Instantiate network object
    network_env = Network(
        paths_dict=paths_dict,
        sumo_params=sumo_params,
        edge_data_frequency=edge_data_frequency,
        buffer_capacity=config["buffer_capacity"],
        batch_size=config["batch_size"],
    )

    # Define surrogate model for SUMO
    surrogate = SurrogateModel().to(DEVICE)

    # Pretrain surrogate model
    surrogate = pre_train(surrogate, config, DEVICE, drivers, network_env)

    epsilon = hyperparams["epsilon"]
    decay = hyperparams["decay"]

    # ============================================================
    # START RL TRAINING
    # ============================================================
    for i in range(config["episodes"]):
        print("Start training")
        # Get action from policy for every driver with incentives mode
        if config["incentives_mode"]:
            routes_edges, actions_index, current_used_budget, acceptance_rate = (
                tr.policy_incentives(
                    drivers=drivers,
                    total_budget=total_budget,
                    epsilon=epsilon,
                    compliance_rate=config["compliance_rate"],
                )
            )

            acceptance_rates.append(acceptance_rate)
            current_used_budgets.append(current_used_budget)

        # Take action from policy for every driver without incentives mode
        else:
            routes_edges, actions_index = tr.policy_no_incentives(
                drivers,
                epsilon,
            )

        # --------------------------------------------------------
        # Execute REAL simulator step
        # --------------------------------------------------------

        total_tt, ind_tt, ind_em, total_em = network_env.step(
            routes_edges=routes_edges,
        )

        # --------------------------------------------------------
        # Store metrics
        # --------------------------------------------------------

        ttts.append(total_tt)
        emissions_total.append(total_em)

        # --------------------------------------------------------
        # Build training target for surrogate replay
        # --------------------------------------------------------

        ordered_ind_tt = torch.tensor(
            [ind_tt[d.trip_id] for d in drivers],
            dtype=torch.float32,
        )

        target = torch.cat(
            [
                ordered_ind_tt,
                torch.tensor([total_tt], dtype=torch.float32),
            ]
        )

        joint_action = torch.tensor(
            [actions_index[d.trip_id] for d in drivers],
            dtype=torch.long,
        )

        # --------------------------------------------------------
        # Add REAL experience to replay buffer
        # --------------------------------------------------------

        surrogate.add_experience(
            joint_action,
            target,
        )

        # ========================================================
        # STANDARD Q-LEARNING UPDATE
        # ========================================================
        for driver_idx, driver in enumerate(drivers):
            idx = actions_index[driver.trip_id]
            reward = ind_tt[driver.trip_id]

            # Update Q-value
            driver.q_values[idx] = (1 - hyperparams["alpha"]) * driver.q_values[
                idx
            ] + hyperparams["alpha"] * reward

        # ========================================================
        # DYNA-Q PLANNING STEPS USING SURROGATE
        # ========================================================

        surrogate.eval()

        for _ in range(DYNA_STEPS):
            simulated_joint_action = []

            # ----------------------------------------------------
            # Sample hypothetical action for every driver
            # ----------------------------------------------------

            for driver in drivers:
                n_actions = len(driver.routes)

                sampled_action = torch.randint(
                    low=0,
                    high=n_actions,
                    size=(1,),
                ).item()

                simulated_joint_action.append(sampled_action)

            simulated_joint_action = torch.tensor(
                simulated_joint_action,
                dtype=torch.long,
                device=DEVICE,
            )

            # ----------------------------------------------------
            # Predict outcome using surrogate
            # ----------------------------------------------------

            pred = surrogate.predict(
                simulated_joint_action,
            )

            pred_ind_tt = pred["individual_tt"].squeeze(0)

            # ----------------------------------------------------
            # Q-learning update using MODELLED experience
            # ----------------------------------------------------

            for driver_idx, driver in enumerate(drivers):
                idx = simulated_joint_action[driver_idx].item()
                reward = pred_ind_tt[driver_idx].item()

                # Update Q-value
                driver.q_values[idx] = (1 - hyperparams["alpha"]) * driver.q_values[
                    idx
                ] + hyperparams["alpha"] * reward

        # ========================================================
        # PERIODIC ONLINE SURROGATE RETRAINING
        # ========================================================

        if i > 0 and i % SURROGATE_RETRAIN_FREQ == 0:
            print(f"\n[Episode {i}] Retraining surrogate with replay buffer...")

            surrogate.retrain(
                epochs=SURROGATE_RETRAIN_EPOCHS,
                batch_size=config.get("surrogate_batch_size", 32),
                lr=config.get("surrogate_finetune_lr", 1e-4),
            )

        # ========================================================
        # Reduce epsilon
        # ========================================================

        epsilon = max(
            0.01,
            epsilon * decay,
        )

        # ========================================================
        # Logging
        # ========================================================

        ut.log_progress(
            i=i,
            episodes=config["episodes"],
            ttts=ttts,
        )

        # ========================================================
        # Update travel times
        # ========================================================

        ut.update_average_travel_times(
            drivers=drivers,
            weights=xml.parse_weights("data/weights.xml"),
        )

    # Save the plot and pickle file for TTT and emissions
    base_name = (
        "compliance_rate_surrogate" if config["compliance_rate"] else "_surrogate"
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
    config_file = ut.load_config(path="scripts/qlearning_no_state_exp_replay.yaml")

    # Loop for different budgets
    for tot_budget in config_file["total_budget"]:
        main(config=config_file, total_budget=tot_budget)
