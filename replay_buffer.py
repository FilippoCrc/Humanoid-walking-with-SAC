import numpy as np
from collections import deque
import random

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

#-----------------------------Prioritized Experience Replay Buffer--------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha        # How much prioritization to use (0 = none, 1 = full)
        self.beta_start = beta_start  # Starting value of beta for importance sampling
        self.beta_frames = beta_frames
        self.frame = 1  # Counter for beta annealing
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer"""
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions with priorities"""
        n_samples = min(batch_size, len(self.buffer))
        
        # Calculate current beta for importance sampling
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        # Get priorities for current buffer
        priorities = self.priorities[:len(self.buffer)]
        
        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), n_samples, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights
        
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Return numpy arrays without normalization
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),  # Changed to float32 for continuous actions
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights
        )

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority.item() + 1e-6  # Add small constant for stability

    def __len__(self):
        return len(self.buffer)