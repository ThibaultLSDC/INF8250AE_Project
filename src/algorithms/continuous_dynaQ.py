# Import packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F 
import gymnasium as gym
import random


# Continuous dyna-Q class
class Continuous_DynaQ():
  """
  Tabular dyna-Q steps
  1. Take step in env
  2. Direct RL
  3. Model-learning
  4. Planning
  """

  def __init__(self, env:gym.Env, step_size=0.1, discount=0.9, epsilon=0.1, planning_steps=5, 
        q_learning_rate=1e-3, model_learning_rate=1e-3, memory_capacity=10000, batch_size=32):
        
        self.env = env
        self.step_size = step_size  # Learning rate
        self.discount = discount  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.planning_steps = planning_steps  # Number of planning steps for model updates

        self.state_space_size = env.observation_space.n
        self.action_space_size = env.action_space.n

        # Initialize Q-value neural network
        self.q_learning_rate = q_learning_rate
        self.q_network = torch.nn.Sequential(
          torch.nn.Linear(self.state_space_size[0], 256),
          torch.nn.ReLU(),
          torch.nn.Linear(256, 256),
          torch.nn.ReLU(),
          torch.nn.Linear(256, self.action_space_size[0])
        )
        self.q_opt = torch.optim.Adam(self.q_network.parameters(), lr=self.q_learning_rate)
        
        # Initialize model neural network
        self.model_learning_rate = model_learning_rate
        self.model_network = torch.nn.Sequential(
          torch.nn.Linear(self.state_space_size[0]+self.action_space_size[0], 256),
          torch.nn.ReLU(),
          torch.nn.Linear(256, 256),
          torch.nn.ReLU(),
          torch.nn.Linear(256, self.state_space_size[0])
        )
        self.model_opt = torch.optim.Adam(self.model_network.parameters(), lr=self.model_learning_rate)
        
        self.memory = []

        self.memory_capacity = memory_capacity
        self.batch_size = batch_size

  def build_network(self, state_space_size, action_space_size):
    """
    Builds a neural network that maps observations to Q-values for each action.
    
    Input:
      state_space_size (matrix): Dimensions of the state space
      action_space_size (int): Size of the action space
    """
    input_dimension = state_space_size.shape[0]
    output_dimension = action_space_size.n
    return torch.nn.Sequential(
        torch.nn.Linear(input_dimension, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, output_dimension),
    )

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
      step_action = random.choice((0,self.action_space_size))
    else: # Exploit optimal action with probability 1-p
      state_tensor = torch.tensor(current_state, dtype=torch.float32)
      q_values = self.q_network(state_tensor).item()
      step_action = int(torch.argmax(q_values).cpu().detach().numpy())
    # --------------------------------
    return step_action
  

  def q_network_update(self, prev_state, prev_action, prev_reward, current_state): # , done # Only if we need to treat terminal states
    """
    Q-network update
   
    Input:
      prev_state (int): State the agent was previously in
      prev_action (int): Action the agent previously took
      prev_reward (int): Reward the agent previously obtained
      current_state (int): State the agent is currently in
      done (bool): Says if the env has terminated or truncated # Only if we need to treat terminal states
      
      env: Custom Grid World environment
      num_episodes (int): Number of iterations for training
    """

    # update_action = np.argmax(self.q_table[current_state,:])

    # self.q_table[prev_state, prev_action] = self.q_table[prev_state, prev_action] + self.step_size*(prev_reward+self.discount*self.q_table[current_state, update_action]-self.q_table[prev_state, prev_action])

    prev_state_tensor = torch.tensor(prev_state, dtype=torch.float32)
    current_state_tensor = torch.tensor(current_state, dtype=torch.float32)
    targets = prev_reward + self.discount*(torch.max(self.q_network(current_state_tensor), dim=1)[0])
    loss = nn.MSELoss()(self.q_network_update(prev_state_tensor), targets)
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()

    # TBD if we need to treat terminal states
    # targets = prev_reward + self.discount*(torch.max(self.q_network, dim=1)[0])*(1-terminated)

  def model_update(self, state, action, reward, next_state):
    """
    Model update
   
    Input:
      prev_state (int): State the agent is currently in
      prev_action (int): Action the agent just took
      prev_reward (int): Reward the agent just got
      current_state (int): State the agent will be in

    """
    if len(self.model) < self.model_capacity:
      self.model.append((state, action, reward, next_state))
    else: # Replace an element of the model
      self.model[random.choice(range(len(self.model)))] = (state, action, reward, next_state)

  def planning(self):
    """
    Plans 'planning_steps' ahead
    """

    for steps in range(self.planning_steps): # Regression gradient descent for self.model
      rnd_sample = random.choice(list(self.model.keys())) 
      state, action = rnd_sample
      current_state, reward = self.model[rnd_sample]
      self.q_network_update(state, action, reward, current_state) # , done

  def training(self, env, num_episodes):
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

    for episode in range(num_episodes):

      # Reset the environment
      state, _ = env.reset(seed=0)
      done = False

      total_reward_per_episode = 0.0
      nb_steps_per_episode = 0.0

      while not done:
        # Get action
        action = self.eps_greedy_policy(state)
        # Take step
        current_state, reward, terminated, truncated,_ = env.step(action)
        done = terminated or truncated
        # Update Q-table
        self.q_network_update(state, action, reward, current_state) # , done
        # Update model
        self.model_update(state, action, reward, current_state)
        # Planning
        self.planning()
        # Update state for next iteration
        state = current_state
        total_reward_per_episode += reward
        nb_steps_per_episode += 1.0

      total_rewards.append(total_reward_per_episode)
      nb_steps_episodes.append(nb_steps_per_episode)
    return total_rewards, nb_steps_episodes
  

# Create test gym environment
# Example usage with CartPole environment
env = gym.make("Pendulum-v0", render_mode="rgb_array",max_episode_steps=1) # , render_mode="rgb_array", max_episode_steps=200
dyna_q_agent = Continuous_DynaQ(env)
dyna_q_agent.training(env, num_episodes=100)


