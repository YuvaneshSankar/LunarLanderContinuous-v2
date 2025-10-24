import torch
import torch.nn.functional as F
import numpy as np
from networks.actor import Actor
from networks.critic import Critic
from utils.ou_noise import OUNoise
from replay_buffer.replay_buffer import ReplayBuffer

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, device, config):
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Twin critics and their targets
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.replay_buffer = ReplayBuffer(config['replay_buffer_size'], state_dim, action_dim, device)
        self.noise = OUNoise(action_dim)

        self.gamma = config['gamma']
        self.tau = config['tau']
        self.policy_noise = config['policy_noise']
        self.noise_clip = config['noise_clip']
        self.policy_freq = config['policy_freq']
        self.max_action = max_action

        self.total_it = 0

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()
        return action

    def train(self, batch_size):
        if self.replay_buffer.size < batch_size:
            return

        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            # We clip the action noise here
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-value
            target_Q1 = self.critic_target_1(next_state, next_action)
            target_Q2 = self.critic_target_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.gamma * target_Q)

        # Get current Q estimates
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)


        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()


            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Soft update target networks
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic_1, self.critic_target_1)
            self.soft_update(self.critic_2, self.critic_target_2)

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def reset_noise(self):
        self.noise.reset()

    def add_noise(self, action):
        noise = self.noise.noise()
        return np.clip(action + noise, -self.max_action, self.max_action)
