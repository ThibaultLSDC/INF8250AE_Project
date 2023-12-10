import numpy as np


class WorldModel:
    def __init__(self, env_size: int, action_size: int) -> None:
        self.env_size = env_size
        self.action_size = action_size

        self.model = [[None] * action_size for _ in range(env_size)]
    
    def update(self, state: int, action: int, next_state: int, reward: float) -> None:
        self.model[state][action] = (next_state, reward)
    
    def get_model(self):
        return self.model