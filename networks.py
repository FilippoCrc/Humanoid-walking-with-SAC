import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    """
    Q-Network for SAC that predicts Q-values given state-action pairs.
    We use two Q-networks to mitigate overestimation bias.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(QNetwork, self).__init__()
        
        # First layer processes both state and action together
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)  # Added extra layer
        self.fc4 = nn.Linear(hidden_dim // 2, 1) # Output single Q-value
        
        # Initialize weights with small values for stability
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Forward pass through the network with activation functions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        return q_value


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, device="cuda", action_bounds=None):
        super(GaussianPolicy, self).__init__()
        
        self.device = device
        
        # Neural network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        self.mean = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std = nn.Linear(hidden_dim // 2, action_dim)
        
        # Store action bounds for scaling
        if action_bounds is None:
            action_bounds = (-0.4, 0.4)  # Default action bounds for humanoid
        self.action_scale = (action_bounds[1] - action_bounds[0]) / 2
        self.action_bias = (action_bounds[1] + action_bounds[0]) / 2
        
        # Initialize weights
        self.apply(self._init_weights)

        self.to(device)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):

            # Xavier initialization
            #torch.nn.init.xavier_uniform_(m.weight)
            #torch.nn.init.constant_(m.bias, 0)

            # Orthogonal initialization 
            torch.nn.init.orthogonal_(m.weight, gain=1.0)
            torch.nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        # Forward pass with layer normalization
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        
        # Clamp log_std for stable exploration
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Use reparameterization trick for backpropagation
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Sample from the normal distribution
        
        # Apply tanh squashing to bound actions
        y_t = torch.tanh(x_t)
        
        # Scale and shift the tanh output to match the desired action range
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log probability, accounting for the action transformation
        log_prob = normal.log_prob(x_t)
        
        # Account for tanh squashing in log probability using the change of variables formula
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

""" class ValueNetwork(nn.Module):
    
    #Optional Value Network for SAC. Some implementations use this instead of double Q-networks.
    #We include it for completeness, though modern implementations often skip it.
    
    def __init__(self, state_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value """