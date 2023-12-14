import numpy as np

from algorithms.tabular_dynaQ import Tabular_DynaQ
from envs.discrete_gridworld import DiscreteGridWorld

env = DiscreteGridWorld()
env.reset()
env.render()

dynaQ = Tabular_DynaQ(env)

dynaQ.training(50)
total_rewards, nb_steps_episodes = dynaQ.eval()

print(total_rewards)
print(np.mean(total_rewards))
print(nb_steps_episodes)

