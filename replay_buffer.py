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
#This is the implementation of the Prioritized Experience Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity   # buffer capacity
        self.alpha = alpha    # alpha value for prioritized experience replay, control how much prioritization
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    #this is the function which storage the expereince in the buffer
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        #control if the buffer has reached the maximum capacity
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        #New experience get maximum priority
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

# sampling function for the buffer
    def sample(self, batch_size, beta=0.4):
        
        n_samples = min(batch_size, len(self.buffer))

        
        #rank based sampling

        # Get priorities of all experiences in buffer
        priorities = self.priorities[:len(self.buffer)]
        
        # Get ranks (ascending order, so highest priority = rank 1)
        ranks = len(priorities) - np.argsort(np.argsort(priorities))
        
        # Calculate probabilities based on rank
        # P(i) = 1 / (rank(i))^α
        probs = (1 / ranks) ** self.alpha
        probs /= probs.sum()
        
        """
        # Random sampling
        # Converts priorities to probabilities
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum() 
        
        """
        
        # Sample experiences indices based on probs (before computed)
        indices = np.random.choice(len(self.buffer), n_samples, p=probs)

        # Calculate importance sampling weights
        # reduces the bias introduced by the non-uniform probabilities
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Normalize the experience
        states = np.stack(states).astype(np.float32) / 255.0
        next_states = np.stack(next_states).astype(np.float32) / 255.0
        
        actions = np.array(actions, dtype=np.int64)  
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return (states, actions, rewards, next_states, dones, indices, weights)

    #After the training_step, this function updates the priorities based on the new TD errors
    def update_priorities(self, indices, priorities):
        priorities = priorities.flatten()
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority) + 1e-6

    def __len__(self):
        return len(self.buffer)