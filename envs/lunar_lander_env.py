import gym
import numpy as np

class LunarLanderEnv:
    def __init__(self):
        self.env = gym.make("LunarLanderContinuous-v2")
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

    def reset(self):
        state = self.env.reset()
        return np.array(state, dtype=np.float32)

    def step(self, action):
        # Clip action to valid range
        action = np.clip(action, self.action_low, self.action_high)
        next_state, reward, done, info = self.env.step(action)
        return np.array(next_state, dtype=np.float32), reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
