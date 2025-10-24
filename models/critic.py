import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400, lr=1e-3):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def weight_updates(self, loss):

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
