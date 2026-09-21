"""Module for the experience replay."""

import random
from collections import deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from marl_incentives.traveller import Driver


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
        drivers: list["Driver"],
        action_index: Mapping[str, int],
        reward: Sequence[object],
        weights: dict[str, float],
        alpha: float | None,
        individual_speeds: dict[str, float] | None = None,
        reward_mode: str = "weighted",
    ) -> None:
        """
        Update each driver's action value from one replayed observation.

        :param drivers: Drivers whose Q-values will be updated.
        :param action_index: Selected action index keyed by driver ID.
        :param reward: Simulation outputs used to compute each driver's reward.
        :param weights: Weights for the multi-objective reward.
        :param alpha: Fixed learning rate, or ``None`` for count-based updates.
        :param individual_speeds: Optional mean speed keyed by driver ID.
        :param reward_mode: Reward definition to apply.
        :return: None.
        """
        total_tt, ind_tt, ind_em, total_em, *extra = reward
        if individual_speeds is None and extra:
            individual_speeds = extra[0]
        for driver in drivers:
            idx = action_index[driver.trip_id]
            # Update action counts
            driver.action_counts[idx] += 1
            # Calculate alpha based on action counts
            learning_rate = alpha or 1 / driver.action_counts[idx]

            # Compute reward
            observed_reward = driver.compute_reward(
                ind_tt,
                ind_em,
                total_tt,
                total_em,
                weights,
                individual_speeds,
                reward_mode,
            )
            # Update Q-value
            driver.q_values[idx] = (
                (1 - learning_rate) * driver.q_values[idx]
                + learning_rate * observed_reward
            )

    def __len__(self) -> int:
        """Get the size of the replay buffer."""
        return len(self.buffer)


class StateReplayBuffer:
    def __init__(self, capacity: int = 500, batch_size: int = 128) -> None:
        """
        Initialise capacity.

        :param capacity: Maximum size of the replay buffer.
        :param batch_size: Size of each batch.
        """
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

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

    @staticmethod
    def update_q_values_discrete_state(
        drivers: list["Driver"],
        state_index: Mapping[str, int] | Sequence[int],
        action_index: Mapping[str, int],
        reward: Sequence[object],
        weights: dict[str, float],
        alpha: float | None,
        individual_speeds: dict[str, float] | None = None,
        reward_mode: str = "weighted",
    ) -> None:
        """
        Update each driver's state-action value from a replayed observation.

        :param drivers: Drivers whose Q-values will be updated.
        :param state_index: Stored state per driver, as a mapping or aligned sequence.
        :param action_index: Selected action index keyed by driver ID.
        :param reward: Simulation outputs used to compute each driver's reward.
        :param weights: Weights for the multi-objective reward.
        :param alpha: Fixed learning rate, or ``None`` for count-based updates.
        :param individual_speeds: Optional mean speed keyed by driver ID.
        :param reward_mode: Reward definition to apply.
        :return: None.
        """
        total_tt, ind_tt, ind_em, total_em, *extra = reward
        if individual_speeds is None and extra:
            individual_speeds = extra[0]
        for driver_index, driver in enumerate(drivers):
            idx = action_index[driver.trip_id]
            index_state = (
                state_index[driver.trip_id]
                if isinstance(state_index, dict)
                else state_index[driver_index]
            )
            # Update state-action pairs counts
            driver.state_action_counts[index_state][idx] += 1
            # Calculate alpha based on state-action counts
            learning_rate = (
                alpha or 1 / driver.state_action_counts[index_state][idx]
            )

            # Compute reward
            observed_reward = driver.compute_reward(
                ind_tt,
                ind_em,
                total_tt,
                total_em,
                weights,
                individual_speeds,
                reward_mode,
            )
            # Update Q-value
            driver.q_values[index_state][idx] = (
                (1 - learning_rate) * driver.q_values[index_state][idx]
                + learning_rate * observed_reward
            )

    def __len__(self) -> int:
        """Get the size of the replay buffer."""
        return len(self.buffer)
