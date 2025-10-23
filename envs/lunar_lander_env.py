import gym

class LunarLanderEnvWrapper:
    def __init__(self):
        self.env = gym.make('LunarLanderContinuous-v2')
        self.env._max_episode_steps = 1000

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space
