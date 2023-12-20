import numpy as np

from algorithms.tabular_dynaQ import Tabular_DynaQ
from envs.discrete_gridworld import DiscreteGridWorld, StochasticDiscreteGridWorld
from algorithms.tabular_Q_planning import TabularModel, Q_Planner

N_TRAINING_STEPS = 20_000
EVAL_STEP_INTERVAL = N_TRAINING_STEPS // 100

# Deterministic environment
env = DiscreteGridWorld(size=(10, 10), seed=42)
env.reset()
env.render()

dynaQ_model = Tabular_DynaQ(env)
dynaQ_model.training(50000)
dynaQ_model.render_q_values(title="DynaQ Model Q-values")
model = TabularModel(dynaQ_model.model)

q_planner = Q_Planner(env, model)
q_planner.training(N_TRAINING_STEPS, eval_step_interval=EVAL_STEP_INTERVAL, eval=True)
total_rewards, nb_steps_episodes, efficiencies = q_planner.eval()
print(f"Results after {N_TRAINING_STEPS} training steps : ")
print(f"    - Mean return     : {np.mean(total_rewards)}")
print(f"    - Mean efficiency : {np.mean(efficiencies)}")

q_planner.render_q_values()


# Stochastic environment
env = StochasticDiscreteGridWorld(size=(10 ,10), seed=42)
env.reset()
env.render()

dynaQ_model = Tabular_DynaQ(env)
dynaQ_model.training(50000)
dynaQ_model.render_q_values(title="DynaQ Model Q-values")
model = TabularModel(dynaQ_model.model)

q_planner = Q_Planner(env, model)
q_planner.training(N_TRAINING_STEPS, eval_step_interval=EVAL_STEP_INTERVAL, eval=True)
total_rewards, nb_steps_episodes, efficiencies = q_planner.eval()
print(f"Results after {N_TRAINING_STEPS} training steps : ")
print(f"    - Mean return     : {np.mean(total_rewards)}")
print(f"    - Mean efficiency : {np.mean(efficiencies)}")

q_planner.render_q_values()
