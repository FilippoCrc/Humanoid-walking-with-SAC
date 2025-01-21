import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from networks import QNetwork, GaussianPolicy

class ReplayBuffer:
    """Stores experience tuples for off-policy training"""
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # Random sampling with replacement
        transitions = random.sample(self.buffer, batch_size)
        # Transpose the batch for easier access
        state, action, reward, next_state, done = zip(*transitions)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)


class SAC:
    """Soft Actor-Critic implementation for continuous action spaces"""
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        alpha=0.2,
        automatic_entropy_tuning=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft update parameter
        self.alpha = alpha  # Temperature parameter
        self.device = device
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Initialize the networks
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Create target networks (initialized as copies of the main networks)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Initialize optimizers
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

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, evaluate=False):
        """Select an action from the input state"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            # During evaluation, use the mean action without sampling
            with torch.no_grad():
                mean, _ = self.policy(state)
                action = torch.tanh(mean)  # Apply tanh
                # scale the actions
                action = action * self.policy.action_scale + self.policy.action_bias
                #print("action choosen during eval",action.cpu().numpy()[0])
                return action.cpu().numpy()[0]
        else:
            # During training, sample from the policy
            with torch.no_grad():
                action, _ = self.policy.sample(state)
                #print("action choosen during training",action.cpu().numpy()[0])
                return action.cpu().numpy()[0]

    def update_parameters(self, batch_size=256):
        """Update the networks using a batch of experiences"""
        # Sample from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(batch_size)

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
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
            
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

    """------------------------- NEW CODE ADDED FOR THE CHECKPOINT FUNCTIONALITY -DATE:22/12/2024 --------------------------"""
    
    def save_checkpoint(self, path, episode, total_steps, replay_buffer=True):
        """Enhanced save method that includes training state"""
        checkpoint = {
            'episode': episode,
            'total_steps': total_steps,
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'alpha': self.alpha
        }
    
        # Save automatic entropy tuning parameters if enabled
        if self.automatic_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
    
        # Optionally save replay buffer
        if replay_buffer:
            checkpoint['replay_buffer'] = self.replay_buffer.buffer
    
            torch.save(checkpoint, path)

    def load_checkpoint(self, path, load_replay_buffer=True):
        """Enhanced load method that restores training state"""
        checkpoint = torch.load(path)
    
        # Load network states
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
    
        # Load optimizer states if they exist
        if 'policy_optimizer_state_dict' in checkpoint:
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        if 'q1_optimizer_state_dict' in checkpoint:
            self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        if 'q2_optimizer_state_dict' in checkpoint:
            self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
    
        # Load alpha and entropy tuning parameters
        self.alpha = checkpoint['alpha']
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
        if 'alpha_optimizer_state_dict' in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
    
        #    Optionally load replay buffer
        if load_replay_buffer and 'replay_buffer' in checkpoint:
            self.replay_buffer.buffer = checkpoint['replay_buffer']
    
        # Return episode and total steps if they exist, otherwise return 0
        return checkpoint.get('episode', 0), checkpoint.get('total_steps', 0)
    