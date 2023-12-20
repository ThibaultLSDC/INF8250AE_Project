import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame as pg
import skimage.measure as skimg
from queue import Queue


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
                 goal_region_min_size: float = 0.25,
                 seed: int = 42,
                 ):

        self.size = np.array(size)
        self.goal = np.array(size, dtype=np.int32) - 1
        self.max_steps = max_steps
        self.goal_region_min_size = goal_region_min_size
        self.seed = seed
        np.random.seed(self.seed)

        if obstacles:
            assert (obstacles.shape[0] == self.size[0]) and (obstacles.shape[1] == self.size[1]), f"Obstacle map size ({obstacles.shape}) doesn't match environment shape ({self.size})"
            self.obstacles = obstacles
            self.region_labels = skimg.label(self.obstacles, connectivity=1, background=1)

        else:
            self._generate_obstacles()

        self.fill_holes()
        self.distance_map = self.generate_distance_map()

        # Rendering
        self.render_mode = render_mode
        self.window_size = (size[0] * 50, size[1] * 50)
        self.window = None
        self.clock = None

        # Observation space
        self.observation_space = spaces.MultiDiscrete(self.size, seed=self.seed)

        # Action space
        self.action_space = spaces.Discrete(n=5, seed=self.seed)

        assert self.observation_space.contains(self.goal), "Invalid goal position"

    def _generate_obstacles(self):  # Generates a map where the obstacle-free region containing the goal contains at least self.goal_region_min_size pixels
        done = False
        while not done:
            self.obstacles = (np.random.rand(*self.size) <= 0.25)
            self.obstacles[self.goal[0], self.goal[1]] = 0
            self.region_labels = skimg.label(self.obstacles, connectivity=1, background=1)

            done = (self.region_labels == self.region_labels[self.goal[0], self.goal[1]]).mean() >= self.goal_region_min_size

    def fill_holes(self):
        self.obstacles[self.region_labels != self.region_labels[self.goal[0], self.goal[1]]] = True

    def generate_distance_map(self):
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        distance_map = np.inf * np.ones_like(self.obstacles)
        distance_map[self.goal[0], self.goal[1]] = 0
        exploration_queue = Queue()
        exploration_queue.put(self.goal)

        while not exploration_queue.empty():
            state = exploration_queue.get()
            for direction in directions:
                nx, ny = state[0] + direction[0], state[1] + direction[1]
                if (0 <= nx < self.size[0]) and (0 <= ny < self.size[1]) and not self.obstacles[nx, ny]:
                    new_distance = min(distance_map[nx, ny], distance_map[state[0], state[1]] + 1)
                    if new_distance < distance_map[nx, ny]:
                        distance_map[nx, ny] = new_distance
                        exploration_queue.put((nx, ny))

        return distance_map

    def reset(self, **kwargs):
        self.position = self.observation_space.sample().astype(np.int32).clip(0, self.goal)
        while self.region_labels[self.position[0], self.position[1]] != self.region_labels[self.goal[0], self.goal[1]]:
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
            self.window = pg.display.set_mode((self.window_size[0], self.window_size[1]))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pg.time.Clock()
        canvas = pg.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        INVERT_Y = self.window_size[1] - 50
        # Goal
        pg.draw.rect(canvas, (0, 0, 255), pg.Rect((int(self.goal[0] * 50), INVERT_Y - int(self.goal[1] * 50)), (50, 50)))
        # Obstacles
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                if self.obstacles[x, y]:
                    pg.draw.rect(canvas, (0, 0, 0), pg.Rect(50 * x, INVERT_Y - 50 * y, 50, 50), width=0)
                pg.draw.rect(canvas, (127, 127, 127), pg.Rect(50 * x, INVERT_Y - 50 * y, 50, 50), width=1)
        # Agent
        pg.draw.circle(canvas, (255, 0, 0), np.array([self.position[0]*50 + 25, INVERT_Y - (self.position[1]*50) + 25]).astype(np.int32), 10)

        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pg.event.pump()
            pg.display.update()
            self.clock.tick(20)
        return np.array(pg.surfarray.pixels3d(canvas))

    def get_states(self):
        states = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if not self.obstacles[i, j]:
                    states.append((i, j))

        return states


class StochasticDiscreteGridWorld(DiscreteGridWorld):
    def __init__(self,
                 size=(5, 5),
                 max_steps=1000,
                 render_mode='human',
                 obstacles: np.ndarray = None,
                 goal_region_min_size: float = 0.25,
                 seed: int = 42,
                 stochasticity=0.5,
                 ):

        super().__init__(size, max_steps, render_mode, obstacles, goal_region_min_size, seed)
        self.stochasticity = stochasticity

    def _update_action(self, action):
        if np.random.random() < self.stochasticity:
            action = np.random.randint(0, 5)
        return super()._update_action(action)