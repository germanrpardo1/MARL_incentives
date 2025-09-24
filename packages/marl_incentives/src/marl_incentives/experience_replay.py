"""Module for the experience replay."""

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 100) -> None:
        """
        Initialise capacity.

        :param capacity: Maximum size of the replay buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: list, action: list, reward: list) -> None:
        """
        Add elements to the replay buffer.

        :param state: State to add to the buffer.
        :param action: Action to add to the buffer.
        :param reward: Reward to add to the buffer.
        """
        self.buffer.append((state, action, reward))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample from the replay buffer.

        :param batch_size: Number of elements to sample.
        :return: Tuple of sampled actions and sampled rewards.
        """
        batch = random.sample(self.buffer, batch_size)
        state, actions, rewards = zip(*batch)
        return np.array(state), np.array(actions), np.array(rewards)

    def __len__(self) -> int:
        """Get the size of the replay buffer."""
        return len(self.buffer)
