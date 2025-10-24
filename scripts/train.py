import torch
import numpy as np
from envs.lunar_lander_env import LunarLanderEnv
from agents.td3_agent import TD3Agent
import yaml
import os

def train(config_path):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env = LunarLanderEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = TD3Agent(env.state_dim, env.action_dim, env.action_high[0], device, config['train'])

    max_episodes = config['train']['max_episodes']
    max_steps = config['env']['max_episode_steps']
    batch_size = config['train']['batch_size']
    total_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        agent.reset_noise()

        for step in range(max_steps):
            action = agent.select_action(state)
            action = agent.add_noise(action)
            next_state, reward, done, _ = env.step(action)
            agent.add_experience(state, action, reward, next_state, done)

            agent.train(batch_size)

            state = next_state
            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)

        if (episode+1) % config['logging']['log_interval'] == 0:
            avg_reward = np.mean(total_rewards[-config['logging']['log_interval']:])
            print(f"Episode: {episode+1}, Avg Reward: {avg_reward:.2f}")

    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/td3_config.yaml", help="Path to config file")
    args = parser.parse_args()
    train(args.config)
