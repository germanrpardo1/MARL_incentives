"""Module for the DQN neural network."""

import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_size, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, budget):
        # budget: tensor shape (batch,1), already normalized
        return self.net(budget)
