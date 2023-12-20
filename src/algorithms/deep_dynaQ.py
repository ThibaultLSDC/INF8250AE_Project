from copy import deepcopy
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from tqdm import tqdm

from algorithms.world_models import TorchWorldModel


class DeepDynaQ():
    def __init__(self, env, planning_steps=10, epsilon=.1, discount=.995,
                 buffer_capacity=20000, name='dynaQ', folder='.') -> None:
        self.name = name
        self.folder = folder

        self.env = env
        
        self.planning_steps = planning_steps
        self.epsilon = epsilon
        self.discount = discount
        self.buffer = deque(maxlen=buffer_capacity)

        self.q_network = self.build_network()
        self.q_network[-1].weight.data.zero_()
        self.q_network[-1].bias.data.zero_()
        self.update_target()
        self.opt = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)

        self.world_model = TorchWorldModel(env.observation_space.shape[0], env.action_space.n)

    def build_network(self):
        input_dimension = self.env.reset().shape[0]
        return nn.Sequential(
            nn.Linear(input_dimension, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.env.action_space.n)
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
    
    def get_batch(self, batch_size=256):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return np.array(state), np.array(action)[:, None], np.array(reward)[:, None], np.array(next_state)

    def act(self, state, greedy=False):
        self.q_network.eval()
        if not greedy:
            if np.random.random() < self.epsilon:
                return np.random.randint(0, self.env.action_space.n)
        state = torch.tensor(state, dtype=torch.float32)[None, :]
        q_values = self.q_network(state)
        self.q_network.train()
        return int(torch.argmax(q_values).cpu().detach().numpy())
    
    def learn_q(self, state, action=None, reward=None, next_state=None):
        if action is None:
            action = self.q_network(torch.tensor(state, dtype=torch.float32)).argmax(dim=1, keepdims=True)
            next_state, reward = self.world_model.pred(state, action)
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = reward

        value = self.q_network(state).gather(1, action)
        with torch.no_grad():
            next_value = self.target_network(next_state)
            max_next_value = next_value.max(dim=1, keepdims=True).values
            target = reward + (1-done) * max_next_value * self.discount
        loss = nn.MSELoss()(value, target.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def train(self, steps=10000, eval_freq=100, eval_eps=10, value_render_steps=[]):
        state = self.env.reset()
        state = self.process_state(state)
        done = False
        counter = tqdm(range(steps))

        best_model = None
        best_length = np.inf

        metrics = {
            'loss_q': [],
            'loss_state': [],
            'loss_reward': [],
            'length': [],
            'std': []
        }

        for step in counter:
            if step < 2000:
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

            if step > 2000:

                # eval
                if step % eval_freq == 0 or len(metrics['length']) == 0:
                    length, std = self.test(episodes=eval_eps, render=False)
                    metrics['length'].append(length)
                    metrics['std'].append(std)
                    if length < best_length:
                        best_model = (deepcopy(self.q_network), deepcopy(self.world_model.model))
                        best_length = length

                batch = self.get_batch()
                metrics['loss_q'].append(self.learn_q(*batch))
                self.update_target()
                model_loss = self.world_model.update(*batch)
                metrics['loss_state'].append(model_loss['state_loss'])
                metrics['loss_reward'].append(model_loss['reward_loss'])
                desc = f"Loss Q: {np.mean(metrics['loss_q'][-100:]):.5f} | Loss Model: {np.mean(metrics['loss_state'][-100:]):.5f} | Length: {np.mean(metrics['length'][-10:]):.2f}"
                counter.set_description(desc)
                for _ in range(self.planning_steps):
                    batch = self.get_batch()
                    self.world_model.update(*batch)
                    self.learn_q(batch[0])
                    self.update_target()
            if step in value_render_steps:
                title = f"{self.name} values after {step} steps"
                path = f"{self.folder}/values_{self.name}_{step}.png"
                self.render_values(title=title, save=True, filename=path)
        return metrics, best_model
    
    def test(self, episodes=1, render=False):
        length = []
        for _ in range(episodes):
            state = self.env.reset()
            state = self.process_state(state)
            done = False
            l = 0
            while not done:
                l += 1
                if render: self.env.render()
                action = self.act(state, greedy=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.process_state(next_state)
                done = terminated or truncated
                state = next_state
                state = next_state
            length.append(l)
        return np.array(length).mean(), np.array(length).std()
    
    def render_values(self, title=None, save=False, filename=None):
        value = np.zeros((100, 100))
        self.q_network.eval()
        for i, x in enumerate(np.linspace(-1, 1, 100)):
            for j, y in enumerate(np.linspace(-1, 1, 100)):
                state = torch.tensor([x, y])[None].float()
                a = self.q_network(state).squeeze().max()
                value[i, j] = a.item()
        self.q_network.train()

        value = np.log(value)

        value[60:100, 60:100] = np.nan

        plt.title(title)
        plt.imshow((np.flipud(value)))
        plt.colorbar()

        if save:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()