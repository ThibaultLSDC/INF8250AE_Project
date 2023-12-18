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
from tqdm import tqdm

from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.cm import get_cmap

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

    def training(self, num_steps, eval_step_interval: int = 500, eval: bool = False):
        """
        Agent training loop

        Input:
        env: Custom Grid World environment
        num_episodes (int): Number of iterations for training

        Returns:
        total_rewards (list): Sum of all the rewards for each episode
        nb_steps_episodes (list): Number of steps for each episode
        """

        steps_per_episode = []
        efficiencies = []
        done = True     # Makes it so that the environmenent is reset when the loop starts

        for step in tqdm(range(num_steps)):
            if done:
                state, _ = self.env.reset(seed=42)
                done = False

            # Get action
            action = self.eps_greedy_policy(state)
            # Take step
            current_state, reward, terminated, truncated,_ = self.env.step(action)
            done = terminated or truncated
            # Update Q-table
            self.q_table_update(state, action, reward, current_state) # , done
            # Update state for next iteration
            state = current_state

            if eval and (step == 1000):
                self.render_q_values(title="Q-values after 1000 steps")

            if eval and not (step % eval_step_interval):
                _, steps, efficiency = self.eval()
                steps_per_episode.append(np.mean(steps))
                efficiencies.append(np.mean(efficiency))
                done = True

        if eval:
            plt.subplot(1, 2, 1)
            plt.plot(eval_step_interval * np.arange(len(steps_per_episode)), steps_per_episode)
            plt.title(f"Number of steps per episode over {num_steps} training steps")
            plt.subplot(1, 2, 2)
            plt.plot(eval_step_interval * np.arange(len(steps_per_episode)), efficiencies)
            plt.title(f"Path performance relative to shortest path over {num_steps} steps")
            plt.show()

    def eval(self, num_episodes=25):
        total_rewards = []
        num_steps_per_episode = []
        efficiencies = []

        for _ in range(num_episodes):
            # Reset the environment
            state, _ = self.env.reset()
            start_state = state
            done = False

            total_reward_per_episode = 0.0
            nb_steps_per_episode = 0.0

            while not done:
                # Get action
                action = np.argmax(self.q_table[state])
                # Take step
                current_state, reward, terminated, truncated,_ = self.env.step(action)
                done = terminated or truncated
                # Update state for next iteration
                state = current_state
                total_reward_per_episode += reward
                nb_steps_per_episode += 1.0

            total_rewards.append(total_reward_per_episode)
            num_steps_per_episode.append(nb_steps_per_episode)
            efficiencies.append(self.env.distance_map[start_state[0], start_state[1]] / num_steps_per_episode[-1])

        # Save data
        file_path = (Path(__file__).parent.parent / "data/tabular_Q_learning.json").resolve()
        if not file_path.parent.exists():
            file_path.parent.mkdir()
        data = {
            "total_rewards": total_rewards,
            "num_steps_per_episode": num_steps_per_episode,
            "efficiencies": efficiencies,
        }
        with file_path.open("w") as json_file:
            json.dump(data, json_file, indent=4)

        return total_rewards, num_steps_per_episode, efficiencies

    def render_q_values(self, title: str = None):
        q_values = np.zeros(self.env.size)
        for state in self.env.get_states():
            x, y = state
            q_values[x, y] = self.q_table[state].max()

        viridis = get_cmap("viridis", 1024)
        colors = viridis(np.linspace(0, 1, 1024))
        colors[0] = np.array([0., 0., 0., 1.])
        cmap = ListedColormap(colors)

        if np.any(q_values):
            plt.imshow(q_values.T, origin="lower", cmap=cmap, norm=LogNorm(clip=True))
            plt.colorbar()
        else:
            plt.imshow(q_values.T, origin="lower", cmap=cmap)
        plt.title(title or "Q-values across environment")
        plt.show()
