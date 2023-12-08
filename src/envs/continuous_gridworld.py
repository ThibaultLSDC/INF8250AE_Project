import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame as pg


class ContinuousGridWorld(gym.Env):
    def __init__(self,
                 size=(10, 10),
                 start=(0, 0),
                 goal=(9, 9),
                 max_steps=1000,
                 render_mode='human',
                 ):

        self.size = np.array(size)
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.max_steps = max_steps

        # Rendering
        self.render_mode = render_mode
        self.window_size = (size[0] * 50, size[1] * 50)
        self.window = None
        self.clock = None

        # Observation space
        self.observation_space = spaces.Box(low=np.zeros(2), high=self.size, shape=(2,), dtype=np.float32)

        # Action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        assert self.observation_space.contains(self.start), "Invalid start position"
        assert self.observation_space.contains(self.goal), "Invalid goal position"

    def reset(self):
        self.position = self.observation_space.sample().astype(np.float32).clip(0, self.goal)
        self.steps = 0
        return self._get_obs()

    def _step(self, action):
        assert hasattr(self, 'position'), "Environment not initialized"
        assert self.action_space.contains(action), "Invalid action"

        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        action = self._update_action(action)
        self.position += action
        self.steps += 1

        truncated = self.steps >= self.max_steps
        terminated = self._has_won()

        reward = float(terminated)

        return self._get_obs(), reward, terminated, truncated, {}
    
    def step(self, action):
        return self._step(action)
    
    def render(self):
        return self._render_frame()
    
    def close(self):
        if self.window is not None:
            pg.display.quit()
            pg.quit()

    def _get_obs(self):
        assert hasattr(self, 'position'), "Environment not initialized"
        return self.position

    def _has_won(self):
        return np.all(self.position >= self.goal)
    
    def _update_action(self, action):
        action = action / 10
        n_x, n_y = self.position + action
        if n_x < 0:
            action[0] = -self.position[0]
        elif n_x > self.size[0]:
            action[0] = self.size[0] - self.position[0]
        if n_y < 0:
            action[1] = -self.position[1]
        elif n_y > self.size[1]:
            action[1] = self.size[1] - self.position[1]
        return action

    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pg.init()
            pg.display.init()
            self.window = pg.display.set_mode((self.window_size[1], self.window_size[0]))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pg.time.Clock()
        canvas = pg.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        # Goal
        pg.draw.rect(canvas, (0, 0, 255), pg.Rect((int(self.goal[0] * 50), int(self.goal[1] * 50)), (50, 50)))
        # Agent
        pg.draw.circle(canvas, (255, 0, 0), (self.position*50).astype(int), 10)
        canvas = pg.transform.rotate(canvas, 90)

        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pg.event.pump()
            pg.display.update()
            self.clock.tick(20)
        return np.transpose(np.array(pg.surfarray.pixels3d(canvas)), axes=(1, 0, 2))


class DiscreetContinuousGridWorld(ContinuousGridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Discrete(9)
        

    def step(self, action):
        return self._step(self.discretize_action(action))