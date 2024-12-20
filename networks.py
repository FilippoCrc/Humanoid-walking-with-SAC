import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    """
    Q-Network for SAC that predicts Q-values given state-action pairs.
    We use two Q-networks to mitigate overestimation bias.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        # First layer processes both state and action together
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output single Q-value
        
        # Initialize weights with small values for stability
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class GaussianPolicy(nn.Module):
    """
    Policy Network for SAC that outputs a Gaussian distribution over actions.
    Uses the reparameterization trick for backpropagation through the sampling process.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_bounds=None):
        super(GaussianPolicy, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Mean and log_std are separate output layers
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Store action bounds for scaling
        if action_bounds is None:
            action_bounds = (-0.4, 0.4)  # Bounds for humanoid robot
        self.action_scale = (action_bounds[1] - action_bounds[0]) / 2
        self.action_bias = (action_bounds[1] + action_bounds[0]) / 2
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Get mean and log_std of action distribution
        mean = self.mean(x)
        log_std = self.log_std(x)
        
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def sample(self, state):
        """
        Sample actions from the Gaussian policy using the reparameterization trick.
        Returns the sampled action and its log probability.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Use reparameterization trick
        
        # Apply tanh squashing and scale to action bounds
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log probability, accounting for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class ValueNetwork(nn.Module):
    """
    Optional Value Network for SAC. Some implementations use this instead of double Q-networks.
    We include it for completeness, though modern implementations often skip it.
    """
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
        return value