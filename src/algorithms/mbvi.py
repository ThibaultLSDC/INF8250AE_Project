from copy import deepcopy
import time
import numpy as np
from typing import TYPE_CHECKING, Tuple

from tqdm import tqdm

from src.algorithms.world_models import WorldModel, StochasticWorldModel

if TYPE_CHECKING:
    import gymnasium as gym


def random_argmax(values: np.ndarray) -> int:
    return np.random.choice(np.flatnonzero(values == values.max()))


class MBValueIteration:
    def __init__(self,
                env_size: int,
                action_size: int,
                gamma: float,
                epsilon: float,
                update_steps: int = 10,
                ) -> None:
        self.env_size = env_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_steps = update_steps

        self.model = WorldModel(env_size, action_size)

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
    
    def train(self, env: 'gym.Env', num_steps: int, eval_freq: int, eval_eps, render: bool=False) -> None:
        state, _ = env.reset()
        done = False
        lengths = []
        stds = []
        for step in tqdm(range(num_steps)):
            action = self.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if render: env.render()
            done = terminated or truncated
            self.model.update(state, action, next_state, reward)
            state = next_state
            for _ in range(self.update_steps):
                self.update()
            if done:
                state, _ = env.reset()
            else:
                state = next_state
    
            if step % eval_freq == 0:
                l, std = self.test(deepcopy(env), eval_eps, render=False)
                lengths.append(l)
                stds.append(std)
        return lengths, stds

    def test(self, env: 'gym.Env', num_episodes: int, render: bool=True) -> None:
        length = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            steps = 0
            while not done:
                steps += 1
                if render: env.render()
                action = self.act(state, greedy=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                if done:
                    state, _ = env.reset()
                else:
                    state = next_state
            length.append(steps)
        return np.array(length).mean(), np.array(length).std()


class StochasticMBValueIteration(MBValueIteration):
    def __init__(self,
                 env: 'gym.Env',
                 gamma: float,
                 epsilon: float,
                 update_steps: int = 10,
                 ) -> None:
        super().__init__(env.size[0] * env.size[1], env.action_space.n, gamma, epsilon, update_steps)
        self.env_size = env.size
        self.env_len = env.size[0] * env.size[1]
        self.action_size = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_steps = update_steps

        self.model = StochasticWorldModel(env, update_size=0.1)

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
