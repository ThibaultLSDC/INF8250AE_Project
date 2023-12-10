import numpy as np
from typing import TYPE_CHECKING

from tqdm import tqdm

from src.algorithms.world_models import WorldModel

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

        self.values = np.zeros(env_size)
        self.policy = np.random.randint(0, action_size, env_size, dtype=np.int32)
    
    def act(self, state: int, greedy=False) -> int:
        if not greedy:
            if np.random.random() < self.epsilon:
                return np.random.randint(0, self.action_size)
        return self.policy[state]
    
    def update(self):
        model = self.model.get_model()
        for state in range(len(model)):
            if not any(model[state]):
                continue
            q = np.ones(self.action_size) * -np.inf
            for action in range(len(model[state])):
                if not model[state][action]:
                    continue
                next_state, reward = model[state][action]
                q[action] = reward + self.gamma * self.values[next_state]
            self.values[state] = np.max(q)
        self.update_policy()
    
    def update_policy(self):
        model = self.model.get_model()
        for state in range(len(model)):
            q = np.ones(self.action_size) * -np.inf
            for action in range(len(model[state])):
                if not model[state][action]:
                    continue
                next_state, reward = model[state][action]
                q[action] = reward + self.gamma * self.values[next_state]
            self.policy[state] = random_argmax(q)
    
    def train(self, env: 'gym.Env', num_steps: int) -> None:
        state, _ = env.reset()
        done = False
        for _ in tqdm(range(num_steps)):
            action = self.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.model.update(state, action, next_state, reward)
            state = next_state
            self.update()
            if done:
                state, _ = env.reset()
            else:
                state = next_state
    
    def test(self, env: 'gym.Env', num_steps: int) -> None:
        state, _ = env.reset()
        done = False
        for _ in tqdm(range(num_steps)):
            action = self.act(state, greedy=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            if done:
                state, _ = env.reset()
            else:
                state = next_state