# Import packages
import json
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm

from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.cm import get_cmap

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
    def __init__(self, env: gym.Env, model: TabularModel, step_size=0.1, discount=0.9):
            self.env = env
            self.model = model
            self.step_size = step_size  # Learning rate
            self.discount = discount  # Discount factor

            self.action_space_size = env.action_space.n

            # Initialize Q-table and model
            self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))
            np.random.seed(self.env.seed)

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
        num_steps (int): Number of iterations for training
        """
        steps_per_episode = []
        efficiencies = []
        self.env.reset()
        for step in tqdm(range(num_steps)):
            state = tuple(self.env.observation_space.sample())
            while self.env.obstacles[state[0], state[1]]:
                state = tuple(self.env.observation_space.sample())
            action = self.env.action_space.sample()
            reward, next_state = self.model.predict(state, action)

            self.q_table_update(state, action, reward, next_state)

            if eval and (step == 1000):
                self.render_q_values(title="Q-values after 1000 steps")

            if eval and not (step % eval_step_interval):
                _, steps, efficiency = self.eval()
                steps_per_episode.append(np.mean(steps))
                efficiencies.append(np.mean(efficiency))

        if eval:
            plt.subplot(1, 2, 1)
            plt.plot(eval_step_interval * np.arange(len(steps_per_episode)), steps_per_episode)
            plt.title(f"Number of steps per episode over {num_steps} training steps")
            plt.subplot(1, 2, 2)
            plt.plot(eval_step_interval * np.arange(len(steps_per_episode)), efficiencies)
            plt.title(f"Path performance relative to shortest path over {num_steps} steps")
            plt.show()

    def eval(self, num_episodes: int = 25):
        returns = []
        num_steps_per_episode = []
        efficiencies = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            start_state = state
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

            efficiencies.append(self.env.distance_map[start_state[0], start_state[1]] / num_steps_per_episode[-1])

        # Save data
        file_path = (Path(__file__).parent.parent / "data/tabular_DynaQ_plus.json").resolve()
        if not file_path.parent.exists():
            file_path.parent.mkdir()
        data = {
            "num_steps_per_episode": num_steps_per_episode,
            "efficiencies": efficiencies,
        }
        with file_path.open("w") as json_file:
            json.dump(data, json_file, indent=4)

        return returns, num_steps_per_episode, efficiencies

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
