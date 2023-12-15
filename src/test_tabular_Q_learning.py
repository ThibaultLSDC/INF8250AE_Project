import numpy as np

from algorithms.tabular_Q_learning import Tabular_Q_learning
from envs.discrete_gridworld import DiscreteGridWorld

env = DiscreteGridWorld()
env.reset()
env.render()

Q_learning = Tabular_Q_learning(env)

Q_learning.training(50)
total_rewards, nb_steps_episodes = Q_learning.eval()

print(total_rewards)
print(np.mean(total_rewards))
print(nb_steps_episodes)

Q_learning.render_q_values()
