import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
from collections import deque
import random

# 1. Policy Network Architecture
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and log_std heads for the Gaussian policy
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Action rescaling
        self.action_scale = torch.tensor(1.0)
        self.action_bias = torch.tensor(0.0)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Squash the actions to be in [-1, 1]
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Calculate log probability, accounting for the tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

# 2. Q-Network Architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)

# 3. SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2  # Temperature parameter
        
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim)
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_network_target = QNetwork(state_dim, action_dim, hidden_dim)
        
        # Copy parameters to target network
        for target_param, param in zip(self.q_network_target.parameters(), 
                                     self.q_network.parameters()):
            target_param.data.copy_(param.data)
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=3e-4)
        
        self.memory = deque(maxlen=1000000)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if evaluate:
            action, _ = self.policy.sample(state)
            return action.detach().cpu().numpy()[0]
        else:
            action, _ = self.policy.sample(state)
            return action.detach().cpu().numpy()[0]

    def train_step(self, batch_size=256):
        if len(self.memory) < batch_size:
            return
        
        # Sample from replay buffer
        batch = random.sample(self.memory, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1)

        # Update Q-networks
        with torch.no_grad():
            next_action, next_log_pi = self.policy.sample(next_state_batch)
            q1_target_next, q2_target_next = self.q_network_target(next_state_batch, next_action)
            min_q_target_next = torch.min(q1_target_next, q2_target_next)
            next_q_value = reward_batch + self.gamma * (1 - done_batch) * \
                          (min_q_target_next - self.alpha * next_log_pi)

        q1, q2 = self.q_network(state_batch, action_batch)
        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy
        new_actions, log_pi = self.policy.sample(state_batch)
        q1_new, q2_new = self.q_network(state_batch, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_pi - min_q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.q_network_target.parameters(), 
                                     self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + 
                                  (1 - self.tau) * target_param.data)

# 4. Training Loop
def train_bipedal_walker():
    env = gym.make('BipedalWalker-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SACAgent(state_dim, action_dim)
    
    max_episodes = 1000
    max_steps = 1600
    batch_size = 256
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            agent.memory.append((state, action, reward, next_state, float(done)))
            
            if len(agent.memory) > batch_size:
                agent.train_step(batch_size)
            
            state = next_state
            episode_reward += reward
            
            if done or truncated:
                break
        
        print(f"Episode {episode}, Reward: {episode_reward}")
        
        # Evaluation every 10 episodes
        if episode % 10 == 0:
            evaluate(env, agent)

# 5. Evaluation Function
def evaluate(env, agent, eval_episodes=5):
    avg_reward = 0
    for _ in range(eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if truncated:
                break
            
        avg_reward += episode_reward
    
    avg_reward /= eval_episodes
    print(f"Average Evaluation Reward: {avg_reward}")
    return avg_reward