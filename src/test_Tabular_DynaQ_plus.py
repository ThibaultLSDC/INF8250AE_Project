import numpy as np
import matplotlib.pyplot as plt

from algorithms.tabular_dynaQ_plus import Tabular_DynaQ_plus
from envs.discrete_gridworld import DiscreteGridWorld

env = DiscreteGridWorld(size=(10, 10), seed=42)
env.reset()
env.render()

dynaQ_plus = Tabular_DynaQ_plus(env)

dynaQ_plus.training(50)
total_rewards, nb_steps_episodes = dynaQ_plus.eval()

print(total_rewards)
print(np.mean(total_rewards))
print(nb_steps_episodes)

# Final graph
dynaQ_plus.render_q_values()

plt.plot(nb_steps_episodes)
plt.xlabel("Number of episodes")
plt.ylabel("Number of steps per episode")
plt.show()

