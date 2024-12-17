import gymnasium as gym
import numpy as np
import torch
from sac_imp import SAC
import time
from collections import deque
import json
import os

class SACTrainer:
    """Handles the training process for SAC algorithm"""
    def __init__(
        self,
        env_name='BipedalWalker-v3',
        max_episodes=1000,
        max_steps=1000,
        batch_size=256,
        eval_interval=10,
        updates_per_step=1,
        start_steps=10000,
        eval_episodes=10,
    ):
        # Initialize training parameters
        self.env_name = env_name
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.eval_interval = eval_interval
        self.updates_per_step = updates_per_step
        self.start_steps = start_steps
        self.eval_episodes = eval_episodes
        
        # Create environments
        self.env = gym.make(env_name)
        self.eval_env = gym.make(env_name)
        
        # Get environment dimensions
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Initialize SAC agent
        self.agent = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            gamma=0.99,
            tau=0.005,
            lr=3e-4,
            alpha=0.2,
            automatic_entropy_tuning=True
        )
        
        # Initialize logging
        self.rewards_history = []
        self.eval_rewards_history = []
        self.episode_length_history = []
        self.loss_history = []
        
        # Create directory for saving results
        self.save_dir = f"results/sac_{env_name}_{int(time.time())}"
        os.makedirs(self.save_dir, exist_ok=True)

    def evaluate_policy(self):
        """Runs evaluation episodes without training"""
        eval_rewards = []
        for _ in range(self.eval_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action deterministically for evaluation
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
                
            eval_rewards.append(episode_reward)
            
        return np.mean(eval_rewards), np.std(eval_rewards)

    def train(self):
        """Main training loop"""
        # Initialize tracking variables
        total_steps = 0
        best_eval_reward = float('-inf')
        early_stop_patience = 50
        no_improvement_count = 0
        
        # Initialize rolling window for reward tracking
        rolling_reward = deque(maxlen=100)
        
        print(f"Starting training on {self.env_name}")
        print(f"State dim: {self.env.observation_space.shape[0]}")
        print(f"Action dim: {self.env.action_space.shape[0]}")
        
        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_losses = []
            done = False
            
            while not done:
                # Sample action from policy or randomly for exploration
                if total_steps < self.start_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state)
                
                # Take action in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_steps += 1
                total_steps += 1
                episode_reward += reward
                
                # Store transition in replay buffer
                self.agent.replay_buffer.push(
                    state, action, reward, next_state, done
                )
                
                # Update networks if enough samples are gathered
                if len(self.agent.replay_buffer) > self.batch_size:
                    for _ in range(self.updates_per_step):
                        losses = self.agent.update_parameters(self.batch_size)
                        episode_losses.append(losses)
                
                state = next_state
                
                # End episode if max steps reached
                if episode_steps >= self.max_steps:
                    done = True
            
            # Record episode information
            self.rewards_history.append(episode_reward)
            self.episode_length_history.append(episode_steps)
            rolling_reward.append(episode_reward)
            
            if episode_losses:
                avg_losses = {
                    k: np.mean([l[k] for l in episode_losses])
                    for k in episode_losses[0].keys()
                }
                self.loss_history.append(avg_losses)
            
            # Evaluate the policy periodically
            if episode % self.eval_interval == 0:
                eval_reward, eval_std = self.evaluate_policy()
                self.eval_rewards_history.append(eval_reward)
                
                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.agent.save(f"{self.save_dir}/best_model.pt")
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Print progress
                print(f"Episode {episode}")
                print(f"Total steps: {total_steps}")
                print(f"Episode reward: {episode_reward:.2f}")
                print(f"Evaluation reward: {eval_reward:.2f} ± {eval_std:.2f}")
                print(f"100 episode average: {np.mean(rolling_reward):.2f}")
                print("-" * 50)
                
                # Save training progress
                self.save_training_history()
                
                # Early stopping check
                if no_improvement_count >= early_stop_patience:
                    print("No improvement for a while. Stopping training.")
                    break
            
            # Success criterion for BipedalWalker
            if np.mean(rolling_reward) > 300:
                print("Environment solved! Stopping training.")
                break
    
    def save_training_history(self):
        """Saves training metrics to disk"""
        history = {
            'rewards': self.rewards_history,
            'eval_rewards': self.eval_rewards_history,
            'episode_lengths': self.episode_length_history,
            'losses': self.loss_history
        }
        
        with open(f"{self.save_dir}/training_history.json", 'w') as f:
            json.dump(history, f)

def main():
    # Create trainer with default parameters
    trainer = SACTrainer(
        env_name='BipedalWalker-v3',
        max_episodes=1000,
        max_steps=1000,
        batch_size=256,
        eval_interval=10,
        updates_per_step=1,
        start_steps=10000
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()