# INF8250AE_Project

This repository contains our code for PolyMTL's INF8250AE's course project.
In our project, we aim at describing and understanding in depth the basics of model based reinforcement learning.
We implemented the following algorithms:
- Tabular Q-Learning (as a baseline)
- Q-Planning
- Dyna-Q
- Dyna-Q+
- Model Based Value Iteration
- Deep Dyna-Q (with a DQN as baseline)

## Project Structure

```
.
├── src/
│ ├── algorithms/
│ │ ├── deep_dynaQ.py
│ │ ├── dqn.py
│ │ ├── mbvi
│ │ ├── tabular_dynaQ_plus.py
│ │ ├── tabular_dynaQ.py
│ │ ├── tabular_Q_learning.py
│ │ ├── tabular_Q_planning.py
│ │ ├── world_models.py
│ ├── envs/
│ │ ├── continuous_gridworld.py
│ │ ├── discrete_gridworld.py
│ ├── data/
├── README.md
├── project.ipynb
```

## Usage

The project is meant to be ran from project.ipynb using Google Colaboratory.

## Environments description

For the sake of flexibility, we implemented our own version of grid world. This allows us to do several visualization of the values and agents behaviors.

To complete the environment, the agent must reach the goal situated in the upper right corner of the grid. The agent receives a single reward of 1 when reaching this goal.

![discrete gridworld](assets/gridworld.gif?raw=True)

For the sake of showing the limits of our model based methods in a continuous setting, we use an analog "continuous gridworld".

![continuous gridworld](assets/continuous_gridworld.gif?raw=True)

The task is essentially the same, with actions and states being continous. This simple detail requires the models to use function approximation, which brings up several challenges.

## Algorithms overview

#### 1. Tabular Q Planning

Leverages a sample model to learn Q-values offline using planning

#### 2. DynaQ

Learns a model online, and learns the Q-values both online and offline using planning with the learned model

#### 3. DynaQ+

Modification to original DynaQ to improve the learned model

#### 4. Model Based Value Iteration

Learns and leverages a distribution model to use the value iteration algorithm

#### 5. Deep DynaQ

A simple adaptation of DynaQ to continuous setting, using function approximation for the world model and the Q-network

Further details on these algorithms can be found in the ```project.ipynb``` notebook.