import numpy as np
import matplotlib.pyplot as plt

from algorithms.mbvi import StochasticMBValueIteration
from envs.discrete_gridworld import StochasticDiscreteGridWorld

np.random.seed(42)

N_TRAINING_STEPS = 10000
EVAL_STEP_INTERVAL = N_TRAINING_STEPS // 100

# Deterministic environment
env = StochasticDiscreteGridWorld(size=(10, 10), seed=42, stochasticity=0.)
eval_env = StochasticDiscreteGridWorld(size=(10, 10), seed=42, stochasticity=0.)

mbvi_model = StochasticMBValueIteration(env,
                                        eval_env,
                                        gamma=0.9,
                                        epsilon=0.05,
                                        update_steps=2,
                                        name="mbvi",
                                        model_update_size=1.,
                                        folder="./graphs"
                                        )
lengths, stds, efficiency, eff_stds = mbvi_model.train(N_TRAINING_STEPS,
                                                       eval_freq=EVAL_STEP_INTERVAL,
                                                       eval_eps=20,
                                                       render=False,
                                                       value_render_steps=[0, 500, 1000, 2500, 4000, 6000, 8000, 10000],)

l, s = np.array(efficiency), np.array(eff_stds)
plt.plot(np.arange(len(l))*EVAL_STEP_INTERVAL, efficiency)
plt.fill_between(np.arange(len(l))*EVAL_STEP_INTERVAL, l - s, l + s, alpha=.5)
plt.title('Efficiency over training: Deterministic environment')
plt.xlabel('Steps')
plt.ylabel('Efficiency')
plt.savefig('mbvi_efficiency.png')
plt.clf()

np.random.seed(42)

# Stochastic environment
env = StochasticDiscreteGridWorld(size=(10, 10), seed=42, stochasticity=.5)
eval_env = StochasticDiscreteGridWorld(size=(10, 10), seed=42, stochasticity=.5)

mbvi_model = StochasticMBValueIteration(env, eval_env, gamma=0.9, epsilon=0.05, update_steps=2,
                                        model_update_size=.1, folder="./graphs")
lengths, stds, efficiency, eff_stds = mbvi_model.train(N_TRAINING_STEPS,
                                                       eval_freq=EVAL_STEP_INTERVAL,
                                                       eval_eps=20,
                                                       render=False,
                                                       value_render_steps=[0, 4000, 4500, 5000, 9999])

l, s = np.array(efficiency), np.array(eff_stds)
plt.plot(np.arange(len(l))*EVAL_STEP_INTERVAL, efficiency)
plt.fill_between(np.arange(len(l))*EVAL_STEP_INTERVAL, l - s, l + s, alpha=.5)
plt.title('Efficiency over training: Stochastic environment')
plt.xlabel('Steps')
plt.ylabel('Efficiency')
plt.savefig('stochastic_mbvi_efficiency.png')
plt.clf()
l, s = np.array(lengths), np.array(stds)
plt.plot(np.arange(len(l))*EVAL_STEP_INTERVAL, l)
plt.fill_between(np.arange(len(l))*EVAL_STEP_INTERVAL, l - s, l + s, alpha=.5)
plt.title('Length over training: Stochastic environment')
plt.xlabel('Steps')
plt.ylabel('Length')
plt.savefig('stochastic_mbvi_lengths.png')
plt.clf()
