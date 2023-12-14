# Import packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import random
import json

from collections import defaultdict

# Tabular dyna-Q class
class Tabular_Q_learning():

    def __init__(self, env: gym.Env, step_size=0.1, discount=0.9, epsilon=0.1):
            self.env = env
            self.step_size = step_size  # Learning rate
            self.discount = discount  # Discount factor
            self.epsilon = epsilon  # Exploration-exploitation trade-off

            self.action_space_size = env.action_space.n

            # Initialize Q-table
            self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))

    def eps_greedy_policy(self, current_state):
        """
        Implementation of epsilon-greedy policy

        Input:
        current_state (int): State in which the agent is currently in

        Returns:
        step_action (int): Action to take in the current_state
        """
        probability = np.random.random()
        if probability < self.epsilon: # Do random action with probability p
            step_action = self.env.action_space.sample()
        else: # Exploit optimal action with probability 1-p
            step_action = np.argmax(self.q_table[current_state])
        # --------------------------------
        return step_action

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

        # TBD if we need to treat terminal states
        # if done == True:
        #   self.q_table[prev_state, prev_action] = self.q_table[prev_state, prev_action] + self.step_size*(prev_reward+self.discount*0-self.q_table[prev_state, prev_action])
        # else: # Essentially if done == False
        #   self.q_table[prev_state, prev_action] = self.q_table[prev_state, prev_action] + self.step_size*(prev_reward+self.discount*self.q_table[current_state, update_action]-self.q_table[prev_state, prev_action])

    def training(self, num_steps: int = 100):
        """
        Agent training loop

        Input:
        env: Custom Grid World environment
        num_episodes (int): Number of iterations for training

        Returns:
        total_rewards (list): Sum of all the rewards for each episode
        nb_steps_episodes (list): Number of steps for each episode
        """

        for episode in range(num_steps):

            # Reset the environment
            state, _ = self.env.reset()
            done = False

            while not done:
                # Get action
                action = self.eps_greedy_policy(state)
                # Take step
                current_state, reward, terminated, truncated,_ = self.env.step(action)
                done = terminated or truncated
                # Update Q-table
                self.q_table_update(state, action, reward, current_state) # , done
                # Update state for next iteration
                state = current_state
    
    def eval(self, num_episodes=100):
        total_rewards = []
        nb_steps_episodes = []

        for episode in range(num_episodes):

            # Reset the environment
            state, _ = self.env.reset()
            done = False

            total_reward_per_episode = 0.0
            nb_steps_per_episode = 0.0

            while not done:
                # Get action
                action = self.eps_greedy_policy(state)
                # Take step
                current_state, reward, terminated, truncated,_ = self.env.step(action)
                done = terminated or truncated
                # Update state for next iteration
                state = current_state
                total_reward_per_episode += reward
                nb_steps_per_episode += 1.0

            total_rewards.append(total_reward_per_episode)
            nb_steps_episodes.append(nb_steps_per_episode)

        # Save data
        file_path = "C:/Users/nicle/OneDrive/Bureau/Automne 2023/INF8250AE - Reinforcement learning/INF8250AE_Project/src/graph_data/tabular_Q_learning.json"
        data = {"total_rewards":total_rewards, "nb_steps_episodes":nb_steps_episodes}
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)

        return total_rewards, nb_steps_episodes

# data = {
#     "name": "John",
#     "age": 30,
#     "city": "New York"
# }

# # Specify the file name
# filename = "data.json"

# # Use json.dump() to write data to the file
# with open(filename, "w") as json_file:
#     json.dump(data, json_file)


# # Specify the file name
# filename = "data.json"

# # Use json.load() to read data from the file
# with open(filename, "r") as json_file:
#     loaded_data = json.load(json_file)

# # Now, loaded_data contains the data from the JSON file
# print(loaded_data)
