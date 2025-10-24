import torch
from envs.lunar_lander_env import LunarLanderEnv
from agents.td3_agent import TD3Agent
import yaml
import numpy as np

def evaluate(config_path, model_path, episodes=10):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env = LunarLanderEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = TD3Agent(env.state_dim, env.action_dim, env.action_high[0], device, config['train'])
    agent.actor.load_state_dict(torch.load(model_path, map_location=device))
    agent.actor.eval()

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            env.render()
            state = next_state
            total_reward += reward

        print(f"Episode {episode+1}: Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/td3_config.yaml", help="Path to config file")
    parser.add_argument("--model", required=True, help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    args = parser.parse_args()
    evaluate(args.config, args.model, args.episodes)
