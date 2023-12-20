import json
import numpy as np
import matplotlib.pyplot as plt

from algorithms.deep_dynaQ import DeepDynaQ
from envs.continuous_gridworld import DiscreteContinuousGridWorld


N_TRAINING_STEPS = 100000
EVAL_STEP_INTERVAL = N_TRAINING_STEPS // 100

# Deep Dyna-Q
env = DiscreteContinuousGridWorld(render_mode='human', size=(3, 3), wall_positions=((((1.5, 1), (1.5, 3)),)))

# only 1 planning steps or too unstable
model = DeepDynaQ(env, 5, epsilon=.1, discount=.99, buffer_capacity=100000, name='DeepDynaQ5')

metrics, best = model.train(N_TRAINING_STEPS, eval_freq=EVAL_STEP_INTERVAL, eval_eps=20,
                            value_render_steps=[10000*i for i in range(11)])

# save metrics dict
with open('data/deepDynaQ5_metrics.json', 'w') as f:
    json.dump(metrics, f)

model.q_network = best[0]
model.render_values(title='Best Deep DynaQ 5 Values', save=True, filename='values_deepdynaQ5_best.png')

l, s = np.array(metrics['length']), np.array(metrics['std'])
plt.plot(np.arange(len(l))*EVAL_STEP_INTERVAL, l)
plt.fill_between(np.arange(len(l))*EVAL_STEP_INTERVAL, l - s, l + s, alpha=.5)
plt.title('Length over training: Deep DynaQ, 5 planning step')
plt.xlabel('Steps')
plt.ylabel('Length')
plt.savefig('deepdynaQ5_length.png')
plt.clf()


# DQN
env = DiscreteContinuousGridWorld(render_mode='human', size=(3, 3), wall_positions=((((1.5, 1), (1.5, 3)),)))

# 0 planning steps means DQN
model = DeepDynaQ(env, 0, epsilon=.1, discount=.99, buffer_capacity=100000, name='DQN', folder='./graphs')

metrics, best = model.train(N_TRAINING_STEPS, eval_freq=EVAL_STEP_INTERVAL, eval_eps=20,
                            value_render_steps=[10000*i for i in range(11)])

# save metrics dict
with open('data/dqn_metrics.json', 'w') as f:
    json.dump(metrics, f)

l, s = np.array(metrics['length']), np.array(metrics['std'])
plt.plot(np.arange(len(l))*EVAL_STEP_INTERVAL, l)
plt.fill_between(np.arange(len(l))*EVAL_STEP_INTERVAL, l - s, l + s, alpha=.5)
plt.title('Length over training: DQN')
plt.xlabel('Steps')
plt.ylabel('Length')
plt.savefig('dqn_length.png')
plt.clf()

# Deep Dyna-Q
env = DiscreteContinuousGridWorld(render_mode='human', size=(3, 3), wall_positions=((((1.5, 1), (1.5, 3)),)))

# only 1 planning steps or too unstable
model = DeepDynaQ(env, 1, epsilon=.1, discount=.99, buffer_capacity=100000, name='DeepDynaQ', folder='./graphs')

metrics, best = model.train(N_TRAINING_STEPS, eval_freq=EVAL_STEP_INTERVAL, eval_eps=20,
                            value_render_steps=[10000*i for i in range(11)])

# save metrics dict
with open('data/deepDynaQ_metrics.json', 'w') as f:
    json.dump(metrics, f)

model.q_network = best[0]
model.render_values(title='Best Deep DynaQ Values', save=True, filename='values_deepdynaQ_best.png')

l, s = np.array(metrics['length']), np.array(metrics['std'])
plt.plot(np.arange(len(l))*EVAL_STEP_INTERVAL, l)
plt.fill_between(np.arange(len(l))*EVAL_STEP_INTERVAL, l - s, l + s, alpha=.5)
plt.title('Length over training: Deep DynaQ, 1 planning step')
plt.xlabel('Steps')
plt.ylabel('Length')
plt.savefig('deepdynaQ_length.png')
plt.clf()
