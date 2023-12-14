from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

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


class TorchWorldModel:
    def __init__(self, env_size: int, action_size: int) -> None:
        self.model = self.build_model(env_size, action_size)
        self.env_size = env_size
        self.action_size = action_size

        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.0001)
    
    def build_model(self, env_size: int, action_size: int):
        model = nn.Sequential(
            nn.Linear(env_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, env_size + 1)
        )
        return model
    
    def pred(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        input = torch.cat((state, action), -1)
        with torch.no_grad():
            output = self.model(input)
        next_state = output[:, :self.env_size].detach().numpy()
        reward = torch.sigmoid(output[:, -1:].detach()).numpy()
        return next_state, reward
    
    def update(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray) -> None:
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
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