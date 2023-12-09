# Import packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F 
import gymnasium as gym
import random

# Tabular dyna-Q class
class Tabular_DynaQ():
  """
  Tabular dyna-Q steps
  1. Take step in env
  2. Direct RL
  3. Model-learning
  4. Planning
  """

  def __init__(self, env, step_size=0.1, discount=0.9, epsilon=0.1, planning_steps=5):
        self.env = env
        self.step_size = step_size  # Learning rate
        self.discount = discount  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.planning_steps = planning_steps  # Number of planning steps for model updates

        self.state_space_size = env.observation_space.n
        self.action_space_size = env.action_space.n

        # Initialize Q-table and model
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))
        self.model = self.reset_model()

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
      step_action =random.choice((0,self.action_space_size))
    else: # Exploit optimal action with probability 1-p
      step_action = np.argmax(self.q_table[current_state,:])
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

    update_action = np.argmax(self.q_table[current_state,:])

    self.q_table[prev_state, prev_action] = self.q_table[prev_state, prev_action] + self.step_size*(prev_reward+self.discount*self.q_table[current_state, update_action]-self.q_table[prev_state, prev_action])

    # TBD if we need to treat terminal states
    # if done == True:
    #   self.q_table[prev_state, prev_action] = self.q_table[prev_state, prev_action] + self.step_size*(prev_reward+self.discount*0-self.q_table[prev_state, prev_action])
    # else: # Essentially if done == False
    #   self.q_table[prev_state, prev_action] = self.q_table[prev_state, prev_action] + self.step_size*(prev_reward+self.discount*self.q_table[current_state, update_action]-self.q_table[prev_state, prev_action])

  def reset_model(self):
    """
    Reset the model

    Returns:
      model (dict): Model of the env with (state, action) 
        as keys and (reward, next_state)
    """

    model = dict()
    for state in range(self.state_space_size):
      for action in range(self.action_space_size):
        model[state, action] = (0,0)
    return model

  def model_update(self, state, action, reward, next_state):
    """
    Model update
   
    Input:
      prev_state (int): State the agent is currently in
      prev_action (int): Action the agent just took
      prev_reward (int): Reward the agent just got
      current_state (int): State the agent will be in

    """

    self.model[state, action] = (reward, next_state)

  def planning(self):
    """
    Plans 'planning_steps' ahead
    """

    for steps in range(self.planning_steps):
      rnd_sample = random.choice(list(self.model.keys()))
      state, action = rnd_sample
      current_state, reward = self.model[rnd_sample]
      self.q_table_update(state, action, reward, current_state) # , done

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
        # terminated.append(terminated)
        # truncated.append(truncated)
        # Update Q-table
        self.q_table_update(state, action, reward, current_state) # , done
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
env = gym.make("CliffWalking-v0", render_mode="rgb_array",max_episode_steps=1) # , render_mode="rgb_array", max_episode_steps=200
dyna_q_agent = Tabular_DynaQ(env)
dyna_q_agent.training(env, num_episodes=100)


