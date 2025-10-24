import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, device):
        self.max_size = max_size
        self.device = device

        self.ptr = 0
        self.size = 0

        self.state_buf = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state_buf = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done_buf = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_state_buf[self.ptr] = next_state
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)

        states = torch.tensor(self.state_buf[idxs], device=self.device)
        actions = torch.tensor(self.action_buf[idxs], device=self.device)
        rewards = torch.tensor(self.reward_buf[idxs], device=self.device)
        next_states = torch.tensor(self.next_state_buf[idxs], device=self.device)
        dones = torch.tensor(self.done_buf[idxs], device=self.device)

        return states, actions, rewards, next_states, dones
