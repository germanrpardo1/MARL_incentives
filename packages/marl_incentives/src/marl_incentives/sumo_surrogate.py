"""
Surrogate model for large-scale multi-agent routing

INPUT
  Joint action vector for 1100 agents

OUTPUT
  - 1100 individual travel times
  - 1 total network travel time

TOTAL OUTPUT DIM = 1101

Important:
  Agents may have different numbers of valid actions.

Example:
  agent 0 -> 5 actions
  agent 1 -> 2 actions
  agent 2 -> 4 actions

We use:
  - shared action embeddings
  - agent embeddings
  - MLP surrogate baseline

This is intentionally SIMPLE but clean and scalable.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from marl_incentives import traveller as tr
from marl_incentives.environment import Network

# ============================================================
# CONFIG
# ============================================================

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

# Example:
# each agent has between 2 and 5 valid actions
#
# Replace with your real values.

agent_num_actions = torch.randint(low=2, high=6, size=(NUM_AGENTS,))


class SimulatorDataset(Dataset):
    """
    Replace synthetic target generation with:
        simulator(joint_action)

    X shape:
        [N, 1100]

    Y shape:
        [N, 1101]
    """

    def __init__(self, drivers, network_env, num_samples=10):
        self.X = self.generate_actions(drivers, num_samples)
        self.Y = self.generate_targets(self.X, drivers, network_env)

    @staticmethod
    def generate_actions(drivers, num_samples):
        actions = torch.zeros((num_samples, NUM_AGENTS), dtype=torch.long)

        for i, driver in enumerate(drivers):
            n_actions = len(driver.costs)

            actions[:, i] = torch.randint(low=0, high=n_actions, size=(num_samples,))

        return actions

    @staticmethod
    def generate_targets(X, drivers, network_env):
        # ----------------------------------------------------
        # FAKE SIMULATOR
        # Replace this block with your simulator outputs
        # ----------------------------------------------------
        outputs = []

        for row in X:
            routes_edges = {
                driver.trip_id: driver.routes[row[i]][1]
                for i, driver in enumerate(drivers)
            }

            outputs.append(network_env.step(routes_edges=routes_edges))

        total_tt_list, ind_tt_list, ind_em_list, total_em_list = zip(*outputs)

        Y = torch.cat(
            [
                torch.stack(
                    [torch.tensor(list(ind_tt.values())) for ind_tt in ind_tt_list]
                ),
                torch.tensor(total_tt_list).unsqueeze(1),
            ],
            dim=1,
        )

        return Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Action embeddings
        #
        # Shared across all agents.
        #
        # action 0..4 -> embedding vector
        # ----------------------------------------------------

        self.action_embedding = nn.Embedding(
            num_embeddings=MAX_ACTIONS, embedding_dim=EMBED_DIM
        )

        # ----------------------------------------------------
        # Agent embeddings
        #
        # Gives model identity information.
        #
        # Otherwise:
        #   "agent 3 choosing action 2"
        # and
        #   "agent 700 choosing action 2"
        # look identical.
        # ----------------------------------------------------

        self.agent_embedding = nn.Embedding(
            num_embeddings=NUM_AGENTS, embedding_dim=EMBED_DIM
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
        self.register_buffer("agent_ids", torch.arange(NUM_AGENTS))

    def forward(self, actions):
        """
        actions shape:
            [batch_size, 1100]
        """

        batch_size = actions.size(0)

        # ----------------------------------------------------
        # Action embeddings
        #
        # shape:
        #   [batch, 1100, EMBED_DIM]
        # ----------------------------------------------------

        action_emb = self.action_embedding(actions)

        # ----------------------------------------------------
        # Agent embeddings
        #
        # shape:
        #   [1100, EMBED_DIM]
        # ----------------------------------------------------

        agent_emb = self.agent_embedding(self.agent_ids)

        # Expand to batch dimension
        #
        # shape:
        #   [batch, 1100, EMBED_DIM]

        agent_emb = agent_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # ----------------------------------------------------
        # Combine information
        # ----------------------------------------------------

        x = action_emb + agent_emb

        # Flatten
        #
        # shape:
        #   [batch, 1100 * EMBED_DIM]

        x = x.reshape(batch_size, -1)

        # Predict travel times
        #
        # output shape:
        #   [batch, 1101]

        return self.mlp(x)


def main():
    # Unpack configuration file
    # Store all the paths
    paths_dict = {
        # Path of output.rou.alt file
        "output_rou_alt_path": "data/output.rou.alt.xml",
        # Path of output.rou file
        "routes_file_path": "data/output.rou.xml",
        # Path for edge data frequency config
        "edge_data_path": "data/edge_data.add.xml",
        # Path for SUMO log
        "log_path": "data/log.xml",
        # Path to write emissions data
        "emissions_path": "data/fcd.xml",
        # Path to emissions data per vehicle
        "emissions_per_vehicle_path": "data/emissions_per_vehicle.txt",
        # Path to stats data after simulation
        "stats_path": "data/stats.xml",
        # Path to tripinfo file
        "trip_info_path": "data/tripinfo.xml",
        "edges_weights_path": "weights.xml",
    }

    sumo_params = {
        "config_path": "data/config.sumocfg",
        "network_path": "data/kamppi.net.xml",
        "routes_path": "data/output.rou.xml",
    }
    # Initialise all drivers
    drivers = tr.initialise_drivers(
        actions_file_path=paths_dict["output_rou_alt_path"],
        incentives_mode=True,
        strategy="logit",
    )

    # Instantiate network object
    network_env = Network(
        paths_dict=paths_dict,
        sumo_params=sumo_params,
        edge_data_frequency=500,
        buffer_capacity=64,
        batch_size=32,
    )
    # ============================================================
    # DATA
    # ============================================================

    dataset = SimulatorDataset(drivers, network_env, num_samples=NUM_SAMPLES)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ============================================================
    # MODEL SETUP
    # ============================================================

    model = SurrogateModel().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    mse = nn.MSELoss()

    # ============================================================
    # TRAINING
    # ============================================================

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0

        for X, Y in loader:
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            pred = model(X)

            # ----------------------------------------------------
            # Separate losses
            # ----------------------------------------------------

            pred_individual = pred[:, :NUM_AGENTS]
            pred_total = pred[:, -1]

            true_individual = Y[:, :NUM_AGENTS]
            true_total = Y[:, -1]

            loss_individual = mse(pred_individual, true_individual)

            loss_total = mse(pred_total, true_total)

            # Weight total network prediction more
            loss = loss_individual + 5.0 * loss_total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch + 1:02d} | Loss = {avg_loss:.6f}")

    # ============================================================
    # INFERENCE
    # ============================================================

    model.eval()

    with torch.no_grad():
        # Build one VALID joint action

        joint_action = torch.zeros((1, NUM_AGENTS), dtype=torch.long)

        for agent_id in range(NUM_AGENTS):
            n_actions = agent_num_actions[agent_id].item()

            joint_action[0, agent_id] = torch.randint(low=0, high=n_actions, size=(1,))

        joint_action = joint_action.to(DEVICE)

        prediction = model(joint_action)

        individual_tt = prediction[0, :NUM_AGENTS]
        total_tt = prediction[0, -1]

        print("\nPredicted total network travel time:")
        print(total_tt.item())

        print("\nFirst 10 individual predictions:")
        print(individual_tt[:10])


if __name__ == "__main__":
    main()
