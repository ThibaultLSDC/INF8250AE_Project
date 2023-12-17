import numpy as np
import matplotlib.pyplot as plt

from algorithms.tabular_Q_learning import Tabular_Q_learning
from envs.discrete_gridworld import DiscreteGridWorld
from envs.discrete_gridworld import StochasticDiscreteGridWorld

# env = DiscreteGridWorld(size=(10,10))
# env.reset(seed=42)
# env.render()

# Q_learning = Tabular_Q_learning(env)

# Q_learning.training(50)
# total_rewards, nb_steps_episodes = Q_learning.eval()

# print(total_rewards)
# print(np.mean(total_rewards))
# print(nb_steps_episodes)

# # Final graph
# Q_learning.render_q_values()


env = StochasticDiscreteGridWorld(size=(10, 10))
env.reset()
env.render()

Q_learning = Tabular_Q_learning(env)

Q_learning.training(50)
total_rewards, nb_steps_episodes = Q_learning.eval()

print(total_rewards)
print(np.mean(total_rewards))
print(nb_steps_episodes)

# Final graph
Q_learning.render_q_values()


plt.plot(nb_steps_episodes)
plt.xlabel("Number of episodes")
plt.ylabel("Number of steps per episode")
plt.show()