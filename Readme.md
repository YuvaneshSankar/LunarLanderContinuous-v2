LunarLanderContinuous-v2
A classic continuous control environment from OpenAI Gym's Box2D suite. The goal is to land a lunar module safely on a landing pad using continuous engine controls.

Environment Overview
Observation Space: 8-dimensional continuous vector (position, velocity, angle, angular velocity, and leg contact flags)

Action Space: 2-dimensional continuous vector (Box(-1, +1, (2,), dtype=np.float32))

action[0]: Main engine throttle (vertical)

action[1]: Lateral engine throttle (horizontal)

Reward: Positive for landing, negative for crashing or moving away from the pad. Bonus for leg contact, penalty for fuel usage.

Episode End: When the lander crashes, comes to rest, or after 1000 steps.

Getting Started
1. Install Dependencies
bash
pip install gym[box2d]
pip install numpy
2. Running the Environment
python
import gym

env = gym.make("LunarLanderContinuous-v2")
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Replace with your policy
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
3. Using with TD3 or Other RL Algorithms
This environment is compatible with continuous-action RL algorithms like TD3, DDPG, and SAC.

Plug in your agent to select actions instead of env.action_space.sample().

Environment Details
State Vector: [x, y, x_dot, y_dot, theta, theta_dot, leg1_contact, leg2_contact]

Action Vector: [main_engine, side_engine] (both in range [-1, 1])

Main Engine: Only works for action[0] >= 0 (scales from 50% to 100% power for 0 <= action[0] <= 1)

Side Engines: Only fire for action[1] <= -0.5 (left) or action[1] >= 0.5 (right)

Tips
The environment is stochastic; results may vary between runs.

For best results, normalize observations and clip actions to valid ranges.

Use a replay buffer and target networks for stable RL training.

References
[OpenAI Gym LunarLanderContinuous-v2 Documentation]​

Box2D Physics Engine

OpenAI Gym