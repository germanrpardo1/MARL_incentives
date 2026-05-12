"""
Surrogate model for large-scale multi-agent routing

INPUT
  Joint action vector for 1100 agents

OUTPUT
  - 1100 individual travel times
  - 1 total network travel time

TOTAL OUTPUT DIM = 1101

Added:
  - Dyna-friendly prediction helpers
  - Online replay buffer
  - Incremental retraining / fine-tuning
  - Replay mixed with original dataset
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from marl_incentives import traveller as tr
from marl_incentives.environment import Network

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path("dataset.pt")
LOAD_DATA = False

NUM_AGENTS = 1100
MAX_ACTIONS = 5
NUM_SAMPLES = 10

INDIVIDUAL_OUTPUTS = NUM_AGENTS
GLOBAL_OUTPUTS = 1

OUTPUT_DIM = INDIVIDUAL_OUTPUTS + GLOBAL_OUTPUTS

EMBED_DIM = 16
HIDDEN_DIM = 2048

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# ACTION SPACE PER AGENT
# ============================================================

agent_num_actions = torch.randint(
    low=2,
    high=6,
    size=(NUM_AGENTS,),
)


class SimulatorDataset(Dataset):
    """
    Generate data for training the SUMO surrogate model:
        simulator(joint_action)

    X shape:
        [N, 1100]

    Y shape:
        [N, 1101]
    """

    def __init__(self, drivers, network_env, num_samples):
        # Load data if already stored
        if LOAD_DATA and DATASET_PATH.exists():
            data = torch.load(DATASET_PATH)

            self.X = data["X"]
            self.Y = data["Y"].float()

        # Generate and store data
        else:
            self.X = self.generate_actions(
                drivers,
                num_samples,
            )

            self.Y = self.generate_targets(
                self.X,
                drivers,
                network_env,
            )

            self.save_dataset()

    def save_dataset(self):
        torch.save(
            {
                "X": self.X,
                "Y": self.Y,
            },
            DATASET_PATH,
        )

    @staticmethod
    def generate_actions(drivers, num_samples):
        actions = torch.zeros(
            (num_samples, NUM_AGENTS),
            dtype=torch.long,
        )

        for i, driver in enumerate(drivers):
            n_actions = len(driver.costs)

            actions[:, i] = torch.randint(
                low=0,
                high=n_actions,
                size=(num_samples,),
            )

        return actions

    @staticmethod
    def generate_targets(X, drivers, network_env):
        targets = []

        for row in X:
            routes_edges = {
                driver.trip_id: driver.routes[row[i]][1]
                for i, driver in enumerate(drivers)
            }

            total_tt, ind_tt, _, _ = network_env.step(routes_edges=routes_edges)

            y = torch.tensor(
                list(ind_tt.values()) + [total_tt],
                dtype=torch.float32,
            )

            targets.append(y)

        return torch.stack(targets)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Action embeddings
        # ----------------------------------------------------

        self.action_embedding = nn.Embedding(
            num_embeddings=MAX_ACTIONS,
            embedding_dim=EMBED_DIM,
        )

        # ----------------------------------------------------
        # Agent embeddings
        # ----------------------------------------------------

        self.agent_embedding = nn.Embedding(
            num_embeddings=NUM_AGENTS,
            embedding_dim=EMBED_DIM,
        )

        input_dim = NUM_AGENTS * EMBED_DIM

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

        # Precompute agent ids
        self.register_buffer(
            "agent_ids",
            torch.arange(NUM_AGENTS),
        )

        # ----------------------------------------------------
        # Replay buffers for online retraining
        # ----------------------------------------------------

        self.base_X = None
        self.base_Y = None

        self.replay_X = []
        self.replay_Y = []

    def forward(self, actions):
        """
        actions shape:
            [batch_size, NUM_AGENTS]
        """

        batch_size = actions.size(0)

        # ----------------------------------------------------
        # Action embeddings
        # ----------------------------------------------------

        action_emb = self.action_embedding(actions)

        # ----------------------------------------------------
        # Agent embeddings
        # ----------------------------------------------------

        agent_emb = self.agent_embedding(self.agent_ids)

        agent_emb = agent_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        x = action_emb + agent_emb

        # Flatten

        x = x.reshape(batch_size, -1)

        return self.mlp(x)

    # ========================================================
    # DYNA / MODEL-BASED RL HELPERS
    # ========================================================

    @torch.no_grad()
    def predict(self, joint_action):
        """
        Convenience inference wrapper.
        """
        self.eval()

        if joint_action.dim() == 1:
            joint_action = joint_action.unsqueeze(0)

        joint_action = joint_action.to(next(self.parameters()).device)

        pred = self.forward(joint_action)

        return {
            "individual_tt": pred[:, :NUM_AGENTS],
            "total_tt": pred[:, -1],
            "raw": pred,
        }

    @torch.no_grad()
    def predict_total(self, joint_action):
        """
        Predict total network travel time only.
        """
        out = self.predict(joint_action)

        return out["total_tt"]

    @torch.no_grad()
    def predict_individual(self, joint_action):
        """
        Predict individual travel times only.
        """
        out = self.predict(joint_action)

        return out["individual_tt"]

    @torch.no_grad()
    def evaluate_joint_action(self, joint_action):
        """
        Reward-style evaluator.

        Higher reward = lower congestion.
        """
        return self.predict_total(joint_action)

    # ========================================================
    # ONLINE RETRAINING
    # ========================================================

    def set_base_dataset(self, X, Y):
        """
        Store original offline dataset.

        Prevents catastrophic forgetting during retraining.
        """

        self.base_X = X.detach().cpu()
        self.base_Y = Y.detach().cpu()

    def add_experience(self, joint_action, target):
        """
        Add newly collected REAL simulator data.

        Args:
            joint_action:
                [NUM_AGENTS]
                or
                [1, NUM_AGENTS]

            target:
                [OUTPUT_DIM]
                or
                [1, OUTPUT_DIM]
        """

        if joint_action.dim() == 2:
            joint_action = joint_action.squeeze(0)

        if target.dim() == 2:
            target = target.squeeze(0)

        self.replay_X.append(joint_action.detach().cpu())

        self.replay_Y.append(target.detach().cpu())

    def save_dataset(self):
        torch.save(
            {
                "X": self.replay_X,
                "Y": self.replay_Y,
            },
            DATASET_PATH,
        )

    def retrain(
        self,
        epochs=1,
        batch_size=32,
        lr=1e-4,
    ):
        """
        Fine-tune surrogate on:
            offline dataset + replay buffer

        This DOES NOT retrain from scratch.
        """

        if self.base_X is None:
            raise ValueError("Base dataset not set. Call model.set_base_dataset(X, Y)")

        # ----------------------------------------------------
        # Build combined dataset
        # ----------------------------------------------------

        X_parts = [self.base_X]
        Y_parts = [self.base_Y]

        if len(self.replay_X) > 0:
            replay_X = torch.stack(self.replay_X)
            replay_Y = torch.stack(self.replay_Y)

            X_parts.append(replay_X)
            Y_parts.append(replay_Y)

        X = torch.cat(X_parts, dim=0)
        Y = torch.cat(Y_parts, dim=0)

        dataset = TensorDataset(X, Y)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
        )

        mse = nn.MSELoss()

        self.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for batch_X, batch_Y in loader:
                batch_X = batch_X.to(DEVICE)
                batch_Y = batch_Y.to(DEVICE)

                pred = self.forward(batch_X)

                pred_individual = pred[:, :NUM_AGENTS]
                pred_total = pred[:, -1]

                true_individual = batch_Y[:, :NUM_AGENTS]
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

            avg_loss = total_loss / len(loader)

            print(f"[Retrain] Epoch {epoch + 1:02d} | Loss = {avg_loss:.6f}")


def main():
    # ========================================================
    # PATHS
    # ========================================================

    paths_dict = {
        "output_rou_alt_path": "data/output.rou.alt.xml",
        "routes_file_path": "data/output.rou.xml",
        "edge_data_path": "data/edge_data.add.xml",
        "log_path": "data/log.xml",
        "emissions_path": "data/fcd.xml",
        "emissions_per_vehicle_path": "data/emissions_per_vehicle.txt",
        "stats_path": "data/stats.xml",
        "trip_info_path": "data/tripinfo.xml",
        "edges_weights_path": "weights.xml",
    }

    sumo_params = {
        "config_path": "data/config.sumocfg",
        "network_path": "data/kamppi.net.xml",
        "routes_path": "data/output.rou.xml",
    }

    # ========================================================
    # DRIVERS + NETWORK
    # ========================================================

    drivers = tr.initialise_drivers(
        actions_file_path=paths_dict["output_rou_alt_path"],
        incentives_mode=True,
        strategy="logit",
    )

    network_env = Network(
        paths_dict=paths_dict,
        sumo_params=sumo_params,
        edge_data_frequency=500,
        buffer_capacity=64,
        batch_size=32,
    )

    # ========================================================
    # DATA
    # ========================================================

    dataset = SimulatorDataset(
        drivers,
        network_env,
        num_samples=NUM_SAMPLES,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = SurrogateModel().to(DEVICE)

    # Store offline dataset
    model.set_base_dataset(
        dataset.X,
        dataset.Y,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    mse = nn.MSELoss()

    # ========================================================
    # INITIAL TRAINING
    # ========================================================

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0

        for X, Y in loader:
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            pred = model(X)

            pred_individual = pred[:, :NUM_AGENTS]
            pred_total = pred[:, -1]

            true_individual = Y[:, :NUM_AGENTS]
            true_total = Y[:, -1]

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

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch + 1:02d} | Loss = {avg_loss:.6f}")

    # ========================================================
    # INFERENCE
    # ========================================================

    model.eval()

    with torch.no_grad():
        joint_action = torch.zeros(
            (1, NUM_AGENTS),
            dtype=torch.long,
        )

        for agent_id in range(NUM_AGENTS):
            n_actions = agent_num_actions[agent_id].item()

            joint_action[0, agent_id] = torch.randint(
                low=0,
                high=n_actions,
                size=(1,),
            )

        joint_action = joint_action.to(DEVICE)

        result = model.predict(joint_action)

        print("\nPredicted total network travel time:")

        print(result["total_tt"].item())

        print("\nFirst 10 individual predictions:")

        print(result["individual_tt"][0, :10])

    # ========================================================
    # ONLINE DYNA RETRAINING EXAMPLE
    # ========================================================

    print("\nAdding new real experience...")

    new_joint_action = torch.randint(
        low=0,
        high=MAX_ACTIONS,
        size=(NUM_AGENTS,),
    )

    # Example:
    # replace with REAL simulator output

    new_target = torch.randn(OUTPUT_DIM)

    model.add_experience(
        new_joint_action,
        new_target,
    )

    # Fine-tune existing model
    model.retrain(
        epochs=2,
        batch_size=32,
        lr=1e-4,
    )


if __name__ == "__main__":
    main()
