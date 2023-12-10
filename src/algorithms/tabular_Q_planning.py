# Import packages
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import random
from collections import defaultdict


class TabularModel():
    def __init__(self, model: defaultdict[Any, Any]):
        self.model = model

    def predict(self, state, action):
      return self.model[state][action]


# Q-planning class
class Q_Planner():
    """
    Q-planning
    1. Sample random (St, At) from environment
    2. Get (Rt', St') from world model
    3. Q-learning update
    """

    def __init__(self, env, model: TabularModel, step_size=0.1, discount=0.9):
            self.env = env
            self.model = model
            self.step_size = step_size  # Learning rate
            self.discount = discount  # Discount factor

            self.action_space_size = env.action_space.n

            # Initialize Q-table and model
            self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))


    def q_table_update(self, prev_state, prev_action, prev_reward, current_state): # , done # Only if we need to treat terminal states
        """
        Q-table update

        Input:
        prev_state (int): State the agent was previously in
        prev_action (int): Action the agent previously took
        prev_reward (int): Reward the agent previously obtained
        current_state (int): State the agent is currently in
        done (bool): Says if the env has terminated or truncated # Only if we need to treat terminal states

        env: Custom Grid World environment
        num_episodes (int): Number of iterations for training
        """

        update_action = np.argmax(self.q_table[current_state])
        self.q_table[prev_state][prev_action] = self.q_table[prev_state][prev_action] + self.step_size*(prev_reward+self.discount*self.q_table[current_state][update_action]-self.q_table[prev_state][prev_action])


    def training(self, num_steps):
        """
        Agent training loop

        Input:
        num_steps (int): Number of iterations for training
        """

        for _ in range(num_steps):
            self.env.reset()
            state = tuple(self.env.observation_space.sample())
            action = self.env.action_space.sample()
            reward, next_state = self.model.predict(state, action)

            self.q_table_update(state, action, reward, next_state)


    def eval(self, num_episodes: int = 25):
        returns = []
        num_steps_per_episode = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            returns.append(0.)
            num_steps_per_episode.append(0)
            while not done:
                action = np.argmax(self.q_table[state])
                next_state, reward, trunc, end, _ = self.env.step(action)
                done = trunc or end
                returns[-1] += reward
                num_steps_per_episode[-1] += 1
                state = next_state

        return returns, num_steps_per_episode
