from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym
    from src.envs.discrete_gridworld import StochasticDiscreteGridWorld


class WorldModel:
    def __init__(self, env_size: int, action_size: int) -> None:
        self.env_size = env_size
        self.action_size = action_size

        self.model = {}
    
    def update(self, state: int, action: int, next_state: int, reward: float) -> None:
        if state not in self.model:
            self.model[state] = [None] * self.action_size
        self.model[state][action] = (next_state, reward)
    
    def __call__(self, state, action):
        return self.model[state][action]


class StochasticWorldModel:
    def __init__(self,
                 env: 'StochasticDiscreteGridWorld',
                 update_size: float=0.1,
                 ) -> None:
        self.env_size = env.size
        self.env_len = env.size[0] * env.size[1]

        self.update_size = update_size

        self.model = {}
        for state in range(self.env_len):
            self.model[state] = [np.ones((env.action_space.n, self.env_len)) / self.env_len, np.zeros(env.action_space.n)]
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        return state[0] * self.env_size[0] + state[1]
    
    def one_hot(self, index: int) -> np.ndarray:
        one_hot = np.zeros(self.env_len)
        one_hot[index] = 1
        return one_hot
    
    def update(self, state: Tuple[int, int], action: int, next_state: Tuple[int, int], reward: float) -> None:
        state = self.state_to_index(state)
        next_state = self.state_to_index(next_state)
        self.model[state][0][action] += self.update_size * (self.one_hot(next_state) - self.model[state][0][action])
        self.model[state][1][action] += self.update_size * (reward - self.model[state][1][action])
    
    def __call__(self, state, action):
        return self.model[state][0][action], self.model[state][1][action]


class TorchWorldModel:
    def __init__(self, env_size: int, action_size: int) -> None:
        self.model = self.build_model(env_size, action_size)
        self.env_size = env_size
        self.action_size = action_size

        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-4)
    
    def build_model(self, env_size: int, action_size: int):
        model = nn.Sequential(
            nn.Linear(env_size + action_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, env_size + 1),
        )
        return model
    
    def pred(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, float]:
        state = torch.tensor(state, dtype=torch.float32)
        # actions to one-hot
        action_onehot = torch.zeros((action.shape[0], self.action_size))
        action_onehot[torch.arange(action.shape[0]), action] = 1
        input = torch.cat((state, action_onehot), -1)
        with torch.no_grad():
            output = self.model(input)
        next_state = output[:, :self.env_size].detach().numpy()
        reward = torch.sigmoid(output[:, -1:].detach()).numpy()
        return next_state, reward
    
    def update(self, states: np.ndarray, action: np.ndarray, rewards: np.ndarray, next_states: np.ndarray) -> dict:
        states = torch.tensor(states, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        actions = torch.zeros((action.shape[0], self.action_size))
        actions[torch.arange(action.shape[0]), action] = 1
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        input = torch.cat((states, actions), dim=1)
        output = self.model(input)
        next_state_pred = output[:, :self.env_size]
        # Sigmoid for reward
        reward_pred = torch.sigmoid(output[:, -1])
        state_loss = nn.MSELoss()(next_state_pred, next_states)
        reward_loss = nn.BCELoss()(reward_pred, rewards.squeeze())
        loss = state_loss + reward_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {
            'state_loss': state_loss.item(),
            'reward_loss': reward_loss.item()}