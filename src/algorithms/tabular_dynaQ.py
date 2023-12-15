# Import packages
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import random
import json

from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.cm import get_cmap

from collections import defaultdict

# Tabular dyna-Q class
class Tabular_DynaQ():
    """
    Tabular dyna-Q steps
    1. Take step in env
    2. Direct RL
    3. Model-learning
    4. Planning
    """

    def __init__(self, env: gym.Env, step_size=0.1, discount=0.9, epsilon=0.1, planning_steps=5):
            self.env = env
            self.step_size = step_size  # Learning rate
            self.discount = discount  # Discount factor
            self.epsilon = epsilon  # Exploration-exploitation trade-off
            self.planning_steps = planning_steps  # Number of planning steps for model updates

            self.action_space_size = env.action_space.n

            # Initialize Q-table and model
            self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))
            self.model = self.reset_model()

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

    def reset_model(self):
        """
        Reset the model

        Returns:
        model (defaultdict): Model of the env with (state, action)
            as keys and (reward, next_state)
        """
        null_state = tuple(np.zeros(self.env.observation_space.shape, dtype=self.env.observation_space.dtype))
        model = defaultdict(lambda: [(0., null_state) for _ in range(self.env.action_space.n)])
        return model

    def model_update(self, state, action, reward, next_state):
        """
        Model update

        Input:
        prev_state (int): State the agent is currently in
        prev_action (int): Action the agent just took
        prev_reward (int): Reward the agent just got
        current_state (int): State the agent will be in

        """

        self.model[state][action] = (reward, next_state)

    def planning(self):
        """
        Plans 'planning_steps' ahead
        """

        for steps in range(self.planning_steps):
            rnd_state = random.choice(list(self.model.keys()))
            action = random.randint(0, self.action_space_size - 1)
            reward, current_state = self.model[rnd_state][action]
            self.q_table_update(rnd_state, action, reward, current_state) # , done

    def training(self, num_episodes=100):
        """
        Agent training loop

        Input:
        env: Custom Grid World environment
        num_episodes (int): Number of iterations for training

        Returns:
        total_rewards (list): Sum of all the rewards for each episode
        nb_steps_episodes (list): Number of steps for each episode
        """
        total_rewards = []
        nb_steps_episodes = []
        nb_episode = 0

        for episode in range(num_episodes):
            # Reset the environment
            state, _ = self.env.reset(seed=42)
            done = False

            total_reward_per_episode = 0.0
            nb_steps_per_episode = 0.0

            

            while not done:
                # Get action
                action = self.eps_greedy_policy(state)
                # Take step
                current_state, reward, terminated, truncated,_ = self.env.step(action)
                done = terminated or truncated
                # Update Q-table
                self.q_table_update(state, action, reward, current_state) # , done
                # Update model
                self.model_update(state, action, reward, current_state)
                # Planning
                self.planning()
                # Update state for next iteration
                state = current_state
                total_reward_per_episode += reward
                nb_steps_per_episode += 1.0
            
            nb_episode += 1
            if nb_episode == 3: # We want to plot the Q-values after the 2nd episode
                # print("q_values ", self.q_table, "\n")
                self.render_q_values_After10Episodes()

            total_rewards.append(total_reward_per_episode)
            nb_steps_episodes.append(nb_steps_per_episode)

        print("total_rewards ", total_rewards,"\n")
        print("nb_steps_episodes ", nb_steps_episodes, "\n")
        return total_rewards, nb_steps_episodes

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
                # Planning
                self.planning()
                # Update state for next iteration
                state = current_state
                total_reward_per_episode += reward
                nb_steps_per_episode += 1.0

            total_rewards.append(total_reward_per_episode)
            nb_steps_episodes.append(nb_steps_per_episode)

        # Save data
        file_path = (Path(__file__).parent.parent / "data/tabular_DynaQ.json").resolve()
        if not file_path.parent.exists():
            file_path.parent.mkdir()
        data = {"total_rewards":total_rewards, "nb_steps_episodes":nb_steps_episodes}
        with file_path.open("w") as json_file:
            json.dump(data, json_file, indent=4)

        return total_rewards, nb_steps_episodes

    def render_q_values(self):
        q_values = np.zeros(self.env.size)
        for state in self.env.get_states():
            x, y = state
            q_values[x, y] = self.q_table[state].max()

        viridis = get_cmap("viridis", 256)
        colors = viridis(np.linspace(0, 1, 256))
        colors[0] = np.array([0., 0., 0., 1.])
        cmap = ListedColormap(colors)

        plt.imshow(q_values.T, origin="lower", cmap=cmap, norm=LogNorm(clip=True))
        plt.colorbar()
        plt.title("Q-values across environment")
        plt.show()

    def render_q_values_After10Episodes(self):
        q_values = np.zeros(self.env.size)
        for state in self.env.get_states():
            x, y = state
            q_values[x, y] = self.q_table[state].max()

        print("q_values ", q_values, "\n")

        viridis = get_cmap("viridis", 256)
        colors = viridis(np.linspace(0, 1, 256))
        colors[0] = np.array([0., 0., 0., 1.])
        cmap = ListedColormap(colors)

        plt.imshow(q_values.T, origin="lower", cmap=cmap, norm=LogNorm(clip=True))
        plt.colorbar()
        plt.title("Q-values across environment after 3 episodes")
        plt.show()
