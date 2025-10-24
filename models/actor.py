import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, max_action, hidden_dim=400, lr=1e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.max_action = max_action


        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        # Forward pass
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.max_action * x

    def weight_updates(self, loss):

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
