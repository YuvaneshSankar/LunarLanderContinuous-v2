LunarLanderContinuous-v2
A classic continuous control environment from OpenAI Gym's Box2D suite. The goal is to land a lunar module safely on a landing pad using continuous engine controls.

Environment Overview
Observation Space: 8-dimensional continuous vector (position, velocity, angle, angular velocity, and leg contact flags)

Action Space: 2-dimensional continuous vector (Box(-1, +1, (2,), dtype=np.float32))

# LunarLanderContinuous-v2

A classic continuous-control environment from the OpenAI Gym Box2D suite. The objective is to pilot a lunar lander to a designated landing pad using continuous thrust controls.

---

## Features

- Continuous 2D control (main + lateral engines)
- 8-dimensional observation space (position, velocity, angle, angular velocity, leg contacts)
- Compatible with continuous-action RL algorithms (TD3, DDPG, SAC, PPO with squashing, etc.)

---

## Quick overview

- Observation space: 8-d vector
    - [x, y, x_dot, y_dot, theta, theta_dot, leg1_contact, leg2_contact]
- Action space: 2-d continuous vector in [-1, 1]
    - action[0]: main engine throttle (vertical)
    - action[1]: side engine (lateral)
- Reward: Bonuses for successful landing and leg contact; penalties for crashing, fuel usage, and moving away from the pad
- Episode termination: crash, come to rest, or after 1000 timesteps

---

## Requirements

- Python 3.7+
- gym with Box2D (OpenAI Gym)
- numpy

You can install the common dependencies with pip:

```bash
pip install gym[box2d] numpy
```

Note: Some systems require additional system packages for Box2D. If installation fails, refer to the Box2D/gym installation docs for your platform.

---

## Quick start

Run the environment interactively with the following minimal script:

```python
import gym

env = gym.make("LunarLanderContinuous-v2")
obs = env.reset()
done = False

while not done:
        action = env.action_space.sample()  # Replace with your policy
        obs, reward, done, info = env.step(action)
        env.render()

env.close()
```

To use this environment with an RL agent, replace the random action sampling with your agent's action output.

---

## Tips for RL training

- Normalize observations (zero mean, unit variance) for faster learning.
- Clip or squash actions to the valid range [-1, 1].
- Use a replay buffer, target networks, and a stable optimizer when using off-policy methods (TD3, DDPG, SAC).
- Seed the environment and your RNGs for reproducible experiments.
- Train multiple seeds and average results to account for environment stochasticity.

---

## Environment details

- State vector (8): [x, y, x_dot, y_dot, theta, theta_dot, leg1_contact, leg2_contact]
- Action vector (2): [main_engine, side_engine]
    - main_engine (action[0]) is effective for values >= 0 and typically scales thrust for 0..1.
    - side_engine (action[1]) controls lateral thrust; left/right thresholds depend on the implementation (commonly fires when |action[1]| exceeds a small threshold).

Reward shaping, termination conditions and exact engine models are defined by the Gym environment implementation; consult Gym's documentation/source for precise formulas.

---

## References

- OpenAI Gym LunarLander Continuous environment
- Box2D physics engine

---

## License & contact

This repository is a local README for the Gym environment. Check the upstream Gym repository for license details.

If you need changes or have suggestions, open an issue or contact me.
