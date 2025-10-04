"""Module for the experience replay."""

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 100, batch_size: int = 32) -> None:
        """
        Initialise capacity.

        :param capacity: Maximum size of the replay buffer.
        :param batch_size: Size of each batch.
        """
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self, action: list, reward: list) -> None:
        """
        Add elements to the replay buffer.

        :param action: Action to add to the buffer.
        :param reward: Reward to add to the buffer.
        """
        self.buffer.append((action, reward))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample from the replay buffer.

        :param batch_size: Number of elements to sample.
        :return: Tuple of sampled actions and sampled rewards.
        """
        batch = random.sample(self.buffer, batch_size)
        actions, rewards = zip(*batch)
        return np.array(actions), np.array(rewards)

    @staticmethod
    def update_q_values(
        drivers: list, action_index, reward, weights: dict, alpha: float
    ):
        """complete."""
        total_tt, ind_tt, ind_em, total_em = reward
        for driver in drivers:
            idx = action_index[driver.trip_id]
            # Compute reward
            reward = driver.compute_reward(ind_tt, ind_em, total_tt, total_em, weights)
            # Update Q-value
            driver.q_values[idx] = (1 - alpha) * driver.q_values[idx] + alpha * reward

    def __len__(self) -> int:
        """Get the size of the replay buffer."""
        return len(self.buffer)
