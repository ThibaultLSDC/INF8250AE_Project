import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame as pg


class DiscreteGridWorld(gym.Env):
    action_id_to_dir = {
        0: (0, 0),
        1: (1, 0),
        2: (0, 1),
        3: (-1, 0),
        4: (0, -1),
    }


    def __init__(self,
                 size=(5, 5),
                 max_steps=1000,
                 render_mode='human',
                 obstacles: np.ndarray = None,
                 ):

        self.size = np.array(size)
        self.goal = np.array(size, dtype=np.int32) - 1
        self.max_steps = max_steps

        if obstacles:
            assert (obstacles.shape[0] == self.size[0]) and (obstacles.shape[1] == self.size[1]), f"Obstacle map size ({obstacles.shape}) doesn't match environment shape ({self.size})"
            self.obstacles = obstacles

        else:
            self.obstacles = (np.random.rand(*self.size) <= 0.1)    # This doesn't guarantee that the goal is reachable

        self.obstacles[self.goal[0], self.goal[1]] = 0

        # Rendering
        self.render_mode = render_mode
        self.window_size = (size[0] * 50, size[1] * 50)
        self.window = None
        self.clock = None

        # Observation space
        self.observation_space = spaces.MultiDiscrete(self.size)

        # Action space
        self.action_space = spaces.Discrete(n=5)

        assert self.observation_space.contains(self.goal), "Invalid goal position"

    def reset(self, **kwargs):
        self.position = self.observation_space.sample().astype(np.int32).clip(0, self.goal)
        while self.obstacles[self.position[0], self.position[1]]:
            self.position = self.observation_space.sample().astype(np.int32).clip(0, self.goal)
        self.steps = 0
        return self._get_obs(), {}

    def _step(self, action):
        assert hasattr(self, 'position'), "Environment not initialized"
        assert np.all((0 <= action) & (action < 5)), "Action must be an integer between 0 and 4 (included)"

        if isinstance(action, np.ndarray):
            action = int(action[0])

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
        return tuple(self.position)

    def _has_won(self):
        return np.all(self.position >= self.goal)

    def _update_action(self, action):
        action = np.array(self.action_id_to_dir[action])
        n_x, n_y = self.position + action

        if not (0 <= n_x < self.size[0]):
            action[0] = 0
            return action   # Works because only one component of each action isn't null.
                            # In this case (out of bound new position), the action is nullified then returned, ignoring other checks

        if not (0 <= n_y < self.size[1]):
            action[1] = 0
            return action

        if self.obstacles[n_x, n_y]:
            action = np.array([0, 0])

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
        # Obstacles
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                if self.obstacles[x, y]:
                    pg.draw.rect(canvas, (0, 0, 0), pg.Rect(50 * x, 50 * y, 50, 50), width=0)
                pg.draw.rect(canvas, (127, 127, 127), pg.Rect(50 * x, 50 * y, 50, 50), width=1)
        # Agent
        pg.draw.circle(canvas, (255, 0, 0), (self.position*50 + 25).astype(int), 10)
        canvas = pg.transform.rotate(canvas, 90)

        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pg.event.pump()
            pg.display.update()
            self.clock.tick(20)
        return np.transpose(np.array(pg.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
