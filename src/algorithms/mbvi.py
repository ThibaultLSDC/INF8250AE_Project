from copy import deepcopy
import time
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, LogNorm
import numpy as np
from typing import TYPE_CHECKING, Tuple

from tqdm import tqdm

from algorithms.world_models import WorldModel, StochasticWorldModel

if TYPE_CHECKING:
    import gymnasium as gym


def random_argmax(values: np.ndarray) -> int:
    return np.random.choice(np.flatnonzero(values == values.max()))


class MBValueIteration:

    def __init__(self,
                env: 'gym.Env',
                eval_env: 'gym.Env',
                gamma: float,
                epsilon: float,
                update_steps: int = 10,
                name="mbvi",
                folder="."
                ) -> None:
        self.name = name
        self.folder = folder
        self.env = env
        self.eval_env = eval_env
        self.env_size = env.size
        self.action_size = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_steps = update_steps

        self.model = WorldModel(self.env_size, self.action_size)

        self.values = {}
        self.policy = {}
    
    def act(self, state: int, greedy=False) -> int:
        if not greedy:
            if np.random.random() < self.epsilon:
                return np.random.randint(0, self.action_size)
        if state not in self.policy:
            return np.random.randint(0, self.action_size)
        return self.policy[state]
    
    def update(self):
        for state in self.model.model:
            if not any(self.model.model[state]):
                continue
            q = np.ones(self.action_size) * -np.inf
            for action in range(len(self.model.model[state])):
                if not self.model(state, action):
                    continue
                next_state, reward = self.model(state, action)
                q[action] = reward + self.gamma * self.values.get(next_state, 0)
            self.values[state] = np.max(q)
        self.update_policy()
    
    def update_policy(self):
        for state in self.model.model:
            q = np.ones(self.action_size) * -np.inf
            for action in range(len(self.model.model[state])):
                if not self.model(state, action):
                    continue
                next_state, reward = self.model(state, action)
                q[action] = reward + self.gamma * self.values.get(next_state, 0)
            self.policy[state] = random_argmax(q)
    
    def train(self, num_steps: int, eval_freq: int, eval_eps, render: bool=False, value_render_steps=[]) -> None:
        state, _ = self.env.reset()
        done = False
        lengths = []
        length_stds = []
        eff = []
        eff_stds = []
        for step in tqdm(range(num_steps)):
            action = self.act(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            if render: self.env.render()
            done = terminated or truncated
            self.model.update(state, action, next_state, reward)
            state = next_state
            for _ in range(self.update_steps):
                self.update()
            if done:
                state, _ = self.env.reset()
            else:
                state = next_state
    
            if step % eval_freq == 0:
                l, std, e, std_e = self.test(eval_eps, render=False)
                lengths.append(l)
                length_stds.append(std)
                eff.append(e)
                eff_stds.append(std_e)
            
            if step in value_render_steps:
                if self.name == "mbvi":
                    title = f"Model Based Value Iteration Values: {step} steps"
                else:
                    title = f"Stochastic Model Based Value Iteration Values: {step} steps"
                path = f"{self.folder}/values_{self.name}_{step}.png"
                self.render_values(title=title,
                                   save=True,
                                   filename=path)
        return lengths, length_stds, eff, eff_stds

    def test(self, num_episodes: int, render: bool=True) -> None:
        length = []
        efficiency = []
        for _ in range(num_episodes):
            state, _ = self.eval_env.reset()
            start_state = state
            done = False
            steps = 0
            while not done:
                steps += 1
                if render: self.eval_env.render()
                action = self.act(state, greedy=True)
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                state = next_state
                if done:
                    state, _ = self.eval_env.reset()
                else:
                    state = next_state
            length.append(steps)
            efficiency.append(self.eval_env.distance_map[start_state[0], start_state[1]] / steps)
        return np.array(length).mean(), np.array(length).std(), np.array(efficiency).mean(), np.array(efficiency).std()
    
    def render_values(self, title: str = None, save=False, filename="values.png"):
        if isinstance(self.values, dict):
            values = [self.values.get(state, 0) for state in range(self.env_size[0] * self.env_size[1])]
        else:
            values = self.values
        values = np.array(values, dtype=float).reshape(self.env_size)
        
        for i in range(self.env_size[0]):
            for j in range(self.env_size[1]):
                if self.env.obstacles[i, j]:
                    values[i, j] = 0.

        viridis = get_cmap("viridis", 1024)
        colors = viridis(np.linspace(0, 1, 1024))
        colors[0] = np.array([0., 0., 0., 1.])
        cmap = ListedColormap(colors)

        if np.any(values):
            plt.imshow(values.T, origin="lower", cmap=cmap)
            plt.colorbar()
        else:
            plt.imshow(values.T, origin="lower", cmap=cmap)
        plt.title(title or "Q-values across environment")

        if save:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()



class StochasticMBValueIteration(MBValueIteration):
    def __init__(self,
                 env: 'gym.Env',
                 eval_env: 'gym.Env',
                 gamma: float,
                 epsilon: float,
                 update_steps: int = 10,
                 name="stochastic_mbvi",
                 model_update_size: float = 0.1
                 ) -> None:
        super().__init__(env, eval_env, gamma, epsilon, update_steps, name)

        self.env_len = env.size[0] * env.size[1]

        self.model = StochasticWorldModel(env, update_size=model_update_size)
        # Setting last state to be terminal
        self.model.model[self.env_len-1][0] *= 0

        self.values = [0] * self.env_len
        self.policy = [np.random.randint(0, self.action_size) for _ in range(self.env_len)]

    def state_to_index(self, state: Tuple[int, int]) -> int:
        return self.model.state_to_index(state)
    
    def act(self, state: Tuple[int, int], greedy=False) -> int:
        if not greedy:
            if np.random.random() < self.epsilon:
                return np.random.randint(0, self.action_size)
        state = self.state_to_index(state)
        return self.policy[state]
    
    def update(self):
        for state in range(self.env_len):
            q = np.ones(self.action_size) * -np.inf
            for action in range(self.action_size):
                next_state_dist, exp_reward = self.model(state, action)
                q[action] = exp_reward + self.gamma * np.sum(self.values * next_state_dist)
            self.values[state] = np.max(q)
        self.update_policy()

    def update_policy(self):
        for state in range(self.env_len):
            q = np.ones(self.action_size) * -np.inf
            for action in range(self.action_size):
                next_state_dist, exp_reward = self.model(state, action)
                q[action] = exp_reward + self.gamma * np.sum(self.values * next_state_dist)
            self.policy[state] = random_argmax(q)
