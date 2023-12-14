import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame as pg


class ContinuousGridWorld(gym.Env):
    def __init__(self,
                 size=(5, 5),
                 max_steps=200,
                 render_mode='human',
                 wall_positions=(((3., 1), (3., 5)), ((2, 0), (2, 4))),
                 ):

        self.size = np.array(size)
        self.goal = np.array(size, dtype=np.float32) - 1
        self.max_steps = max_steps

        self.wall_positions = np.array(wall_positions)

        # Rendering
        self.render_mode = render_mode
        self.window_size = (size[0] * 50, size[1] * 50)
        self.window = None
        self.clock = None

        # Observation space
        self.observation_space = spaces.Box(low=np.zeros(2), high=self.size, shape=(2,), dtype=np.float32)

        # Action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        assert self.observation_space.contains(self.goal), "Invalid goal position"

    def reset(self):
        self.position = self.observation_space.sample().astype(np.float32)
        while self.position[0] > self.goal[0] - .9 and self.position[1] > self.goal[1] - .9:
            self.position = self.observation_space.sample().astype(np.float32)
        self.steps = 0
        return self._get_obs()

    def _step(self, action):
        assert hasattr(self, 'position'), "Environment not initialized"
        assert np.all(np.abs(action) <= 1), "Invalid action (action should be in [-1, 1])"

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
        action = action / 5
        n_x, n_y = self.position + action
        if n_x < 0:
            action[0] = -self.position[0]
        elif n_x > self.size[0]:
            action[0] = self.size[0] - self.position[0]
        if n_y < 0:
            action[1] = -self.position[1]
        elif n_y > self.size[1]:
            action[1] = self.size[1] - self.position[1]

        # Wall collision
        if len(self.wall_positions) > 0:
            for wall in self.wall_positions:
                A = wall[0]
                B = wall[1]
                C = self.position
                D = self.position + action

                det = (A[0] - B[0]) * (D[1] - C[1]) - (A[1] - B[1]) * (D[0] - C[0])
                if det != 0:
                    root1 = ((D[1] - C[1]) * (D[0] - B[0]) - (D[0] - C[0]) * (D[1] - B[1])) / det
                    root2 = ((A[0] - B[0]) * (D[1] - B[1]) - (A[1] - B[1]) * (D[0] - B[0])) / det
                    if 0 <= root1 <= 1 and 0 <= root2 <= 1:
                        if self.position[0] < wall[0][0]:
                            action[0] = wall[0][0] - self.position[0] - 6/50
                        else:
                            action[0] = wall[0][0] - self.position[0] + 6/50
                        action[0] = 0
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
        # Wall
        if len(self.wall_positions) > 0:
            for wall in self.wall_positions:
                x = int(wall[0][0] * 50 -2)
                y = int(wall[0][1] * 50)
                height = int(wall[1][1] - wall[0][1]) * 50
                width = 4
                pg.draw.rect(canvas, (0, 0, 0), pg.Rect((x, y), (width, height)))
        # Agent
        pg.draw.circle(canvas, (255, 0, 0), (self.position*50).astype(int), 4)
        canvas = pg.transform.rotate(canvas, 90)

        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pg.event.pump()
            pg.display.update()
            self.clock.tick(20)
        return np.transpose(np.array(pg.surfarray.pixels3d(canvas)), axes=(1, 0, 2))


class DiscreteContinuousGridWorld(ContinuousGridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Discrete(9)
        
        self._discreet_to_continuous = {i: np.array((np.cos(i * np.pi / 4), np.sin(i * np.pi / 4))) for i in range(8)}
        self._discreet_to_continuous[8] = np.array((0., 0.))

    def step(self, action):
        return self._step(self._discretize_action(action))
    
    def _discretize_action(self, action):
        action = self._discreet_to_continuous[action]
        return action