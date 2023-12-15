import numpy as np

from algorithms.tabular_dynaQ import Tabular_DynaQ
from envs.discrete_gridworld import DiscreteGridWorld
from algorithms.tabular_Q_planning import TabularModel, Q_Planner


env = DiscreteGridWorld(size=(10, 10))
env.reset()
env.render()

dynaQ_model = Tabular_DynaQ(env)

dynaQ_model.training(50)
model = TabularModel(dynaQ_model.model)
dynaQ_model.render_q_values()

q_planner = Q_Planner(env, model)
q_planner.training(10000)
ret, n = q_planner.eval()

print(ret)
print(np.mean(ret))
print(n)

q_planner.render_q_values()