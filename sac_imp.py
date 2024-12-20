import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from networks import QNetwork, GaussianPolicy
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class SAC:
    """Soft Actor-Critic implementation for continuous action spaces"""
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=512,  # Increased for Humanoid
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        alpha=0.2,
        automatic_entropy_tuning=True,
        use_per=False,   # New parameter to toggle PER
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        
        self.device = device
        
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft update parameter
        self.alpha = alpha  # Temperature parameter
        self.device = device
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Initialize the networks
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim, device=device).to(device)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Create target networks (initialized as copies of the main networks)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Initialize optimizers with different learning rates
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # Automatic entropy tuning
        if automatic_entropy_tuning:
            # Target entropy is -|A|
            self.target_entropy = -action_dim
            # Log alpha will be optimized
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Initialize replay buffer based on use_per flag
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=1000000)
        else:
            self.replay_buffer = ReplayBuffer(capacity=1000000)
    
        self.use_per = use_per

    def select_action(self, state, evaluate=False):
        """
        Select an action from the policy.
        Args:
            state: Can be either a numpy array or a torch tensor
            evaluate: Boolean indicating whether to use deterministic action selection
        """
         # First check if state is already a tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
    
        # Make sure state has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
    
        # Ensure state is on the correct device (although it should be already)
        state = state.to(self.device)

        
        if evaluate:
            # During evaluation, use the mean action without sampling
            with torch.no_grad():
                mean, _ = self.policy(state)
                return torch.tanh(mean).cpu().numpy()[0]
        else:
        # During training, sample from the policy
            with torch.no_grad():
                action, _ = self.policy.sample(state)
                return action.cpu().numpy()[0]
            
            
    def update_parameters(self, batch_size=256):
        """Update the networks using a batch of experiences"""

        # Sample a batch from replay buffer
        if self.use_per:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices, weights = self.replay_buffer.sample(batch_size)
            # Convert weights to tensor
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(batch_size)
            weights = 1.0

        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # Sample next actions and their log probs for next states
            next_actions, next_log_probs = self.policy.sample(next_state_batch)
            
            # Calculate target Q-values
            q1_next = self.q1_target(next_state_batch, next_actions)
            q2_next = self.q2_target(next_state_batch, next_actions)
            q_next = torch.min(q1_next, q2_next)
            
            # Calculate target including entropy term
            value_target = q_next - self.alpha * next_log_probs
            q_target = reward_batch + (1 - done_batch) * self.gamma * value_target

        # Calculate TD errors for priority updates
        if self.use_per:
            with torch.no_grad():
                td_error = torch.abs(self.q1(state_batch, action_batch) - q_target)
                self.replay_buffer.update_priorities(indices, td_error.cpu().numpy())

        # Update Q-Networks
        q1_pred = self.q1(state_batch, action_batch)
        q2_pred = self.q2(state_batch, action_batch)
        
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update Policy
        new_actions, log_probs = self.policy.sample(state_batch)
        q1_new = self.q1(state_batch, new_actions)
        q2_new = self.q2(state_batch, new_actions)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update temperature parameter if automatic entropy tuning is enabled
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # Soft update target networks
        self._soft_update_target_networks()

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item()
        }

    def _soft_update_target_networks(self):
        """Soft update the target networks using the tau parameter"""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_( target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_( target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, path):
        """Save the model parameters"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'alpha': self.alpha
        }, path)

    def load(self, path):
        """Load the model parameters"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        self.alpha = checkpoint['alpha']