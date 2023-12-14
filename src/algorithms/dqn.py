from copy import deepcopy
import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from tqdm import tqdm


class DQN():
    def __init__(self, env, epsilon=.1, discount=.995, buffer_capacity=50000) -> None:
        
        self.env = env
        
        self.epsilon = epsilon
        self.discount = discount
        self.buffer = deque(maxlen=buffer_capacity)

        self.q_network = self.build_network()
        self.update_target()
        self.opt = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    def build_network(self):
        input_dimension = self.env.reset().shape[0]
        return nn.Sequential(
            nn.Linear(input_dimension, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.env.action_space.n)
        )
    
    def update_target(self):
        if not hasattr(self, 'target_network'):
            self.target_network = deepcopy(self.q_network)
            return
        if not hasattr(self, 'target_counter'):
            self.target_counter = 0
        self.target_counter += 1
        if self.target_counter % 1000 == 0:
            self.target_network = deepcopy(self.q_network)
    
    def process_state(self, state):
        return 2 * state / self.env.observation_space.high - 1

    def update_buffer(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def get_batch(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return np.array(state), np.array(action)[:, None], np.array(reward)[:, None], np.array(next_state)

    def act(self, state, greedy=False):
        if not greedy:
            if np.random.random() < self.epsilon:
                return np.random.randint(0, self.env.action_space.n)
        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.q_network(state)
        return int(torch.argmax(q_values).cpu().detach().numpy())

    def learn_q(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = 1 - reward

        value = self.q_network(state).gather(1, action)
        with torch.no_grad():
            next_value = self.target_network(next_state)
            max_next_value = torch.max(next_value, dim=1)[0][:, None]
            target = reward + (1-done) * max_next_value * self.discount
        loss = nn.MSELoss()(value, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def train(self, steps=10000):
        state = self.env.reset()
        state = self.process_state(state)
        done = False
        counter = tqdm(range(steps))
        for step in counter:
            if step < 10000:
                action = self.env.action_space.sample()
            else:
                action = self.act(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = self.process_state(next_state)
            done = terminated or truncated
            self.update_buffer(state, action, reward, next_state)
            state = next_state
            if done:
                state = self.env.reset()
                state = self.process_state(state)

            if len(self.buffer) > 10000:
                # self.env.render()
                batch = self.get_batch()
                loss_q = self.learn_q(*batch)
                self.update_target()
                counter.set_description(f"Q loss: {loss_q:.4f}")
    
    def test(self, steps=10000):
        state = self.env.reset()
        state = self.process_state(state)
        done = False
        for _ in tqdm(range(steps)):
            action = self.act(state, greedy=False)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = self.process_state(next_state)
            self.env.render()
            done = terminated or truncated
            state = next_state
            if done:
                state = self.env.reset()
                state = self.process_state(state)