import numpy as np
import matplotlib.pyplot as plt

from algorithms.tabular_dynaQ import Tabular_DynaQ
from envs.discrete_gridworld import DiscreteGridWorld
from envs.discrete_gridworld import StochasticDiscreteGridWorld

N_TRAINING_STEPS = 50_000
EVAL_STEP_INTERVAL = N_TRAINING_STEPS // 100

### Deterministic environment
env = DiscreteGridWorld(size=(10, 10), seed=42)
env.reset()
env.render()

dynaQ = Tabular_DynaQ(env)

dynaQ.training(N_TRAINING_STEPS, eval_step_interval=EVAL_STEP_INTERVAL, eval=True)
total_rewards, nb_steps_episodes, efficiencies = dynaQ.eval(25)
print(f"Results after {N_TRAINING_STEPS} training steps : ")
print(f"    - Mean return     : {np.mean(total_rewards)}")
print(f"    - Mean efficiency : {np.mean(efficiencies)}")

# Final graph
dynaQ.render_q_values()


### Stochatsic environment
env = StochasticDiscreteGridWorld(size=(10, 10))
env.reset()
env.render()

dynaQ = Tabular_DynaQ(env)

dynaQ.training(N_TRAINING_STEPS, eval_step_interval=EVAL_STEP_INTERVAL, eval=True)
total_rewards, nb_steps_episodes, efficiencies = dynaQ.eval(25)
print(f"Results after {N_TRAINING_STEPS} training steps : ")
print(f"    - Mean return     : {np.mean(total_rewards)}")
print(f"    - Mean efficiency : {np.mean(efficiencies)}")

# Final graph
dynaQ.render_q_values()