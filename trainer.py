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
        debug_config = None
    ):
                # Default debugging configuration
        self.debug_config = {
            'print_episode_progress': True,    # Basic episode information
            'print_detailed_loss': True,      # Detailed loss components
            'print_action_values': False,      # Action statistics
            'print_network_updates': True,    # Network update information
            'print_buffer_stats': False,       # Replay buffer statistics
            'print_eval_details': True         # Detailed evaluation information
        }
        # Update with user-provided config if any
        if debug_config is not None:
            self.debug_config.update(debug_config)
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

    def debug_print(self, category, message):
        """Utility function for conditional debugging output"""
        if self.debug_config.get(f'print_{category}', False):
            print(f"[DEBUG-{category}] {message}")

    def evaluate_policy(self):
        """Evaluation with debug information"""
        eval_rewards = []
        eval_steps = []
        
        for eval_ep in range(self.eval_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, evaluate=True)
                
                self.debug_print('action_values', 
                    f"Eval Episode {eval_ep} - Step {episode_steps} - Action: {action}")
                
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1
                state = next_state
            
            self.debug_print('eval_details',
                f"Eval Episode {eval_ep} complete - Reward: {episode_reward:.2f} - Steps: {episode_steps}")
            
            eval_rewards.append(episode_reward)
            eval_steps.append(episode_steps)
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        self.debug_print('eval_details',
            f"\nEvaluation Summary:\n"
            f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}\n"
            f"Mean Steps: {np.mean(eval_steps):.1f}\n"
            f"Success Rate: {sum(r > 300 for r in eval_rewards)/len(eval_rewards):.2%}")
        
        return mean_reward, std_reward

    def train(self):
        """Training loop with comprehensive debugging"""
        total_steps = 0
        best_eval_reward = float('-inf')
        early_stop_patience = 50
        no_improvement_count = 0
        rolling_reward = deque(maxlen=100)
        
        self.debug_print('episode_progress',
            f"\nStarting training on {self.env_name}\n"
            f"State dim: {self.env.observation_space.shape[0]}\n"
            f"Action dim: {self.env.action_space.shape[0]}\n"
            f"Training parameters:\n"
            f"  Max episodes: {self.max_episodes}\n"
            f"  Batch size: {self.batch_size}\n"
            f"  Updates per step: {self.updates_per_step}")
        
        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_losses = []
            action_values = []  # Track actions for debugging
            done = False
            
            while not done:
                if total_steps < self.start_steps:
                    action = self.env.action_space.sample()
                    self.debug_print('action_values', 
                        f"Random action at step {total_steps}: {action}")
                else:
                    action = self.agent.select_action(state)
                    self.debug_print('action_values', 
                        f"Policy action at step {total_steps}: {action}")
                
                action_values.append(action)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_steps += 1
                total_steps += 1
                episode_reward += reward
                
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                
                self.debug_print('buffer_stats',
                    f"Buffer size: {len(self.agent.replay_buffer)} - "
                    f"Episode step: {episode_steps}")
                
                if len(self.agent.replay_buffer) > self.batch_size:
                    for update_idx in range(self.updates_per_step):
                        losses = self.agent.update_parameters(self.batch_size)
                        episode_losses.append(losses)
                        
                        self.debug_print('network_updates',
                            f"Update {update_idx} at step {total_steps}:\n"
                            f"  Q1 Loss: {losses['q1_loss']:.4f}\n"
                            f"  Q2 Loss: {losses['q2_loss']:.4f}\n"
                            f"  Policy Loss: {losses['policy_loss']:.4f}")
                
                state = next_state
                
                if episode_steps >= self.max_steps:
                    done = True
            
            # Episode completion statistics
            self.rewards_history.append(episode_reward)
            self.episode_length_history.append(episode_steps)
            rolling_reward.append(episode_reward)
            
            if episode_losses:
                avg_losses = {k: np.mean([l[k] for l in episode_losses])
                            for k in episode_losses[0].keys()}
                self.loss_history.append(avg_losses)
                
                self.debug_print('detailed_loss',
                    f"\nEpisode {episode} Average Losses:\n"
                    f"  Q1 Loss: {avg_losses['q1_loss']:.4f}\n"
                    f"  Q2 Loss: {avg_losses['q2_loss']:.4f}\n"
                    f"  Policy Loss: {avg_losses['policy_loss']:.4f}")
            
            # Action statistics for the episode
            if self.debug_config['print_action_values']:
                action_array = np.array(action_values)
                self.debug_print('action_values',
                    f"\nEpisode {episode} Action Statistics:\n"
                    f"  Mean: {action_array.mean(axis=0)}\n"
                    f"  Std: {action_array.std(axis=0)}\n"
                    f"  Min: {action_array.min(axis=0)}\n"
                    f"  Max: {action_array.max(axis=0)}")
            
            # Episode summary
            self.debug_print('episode_progress',
                f"\nEpisode {episode} Complete:\n"
                f"Total steps: {total_steps}\n"
                f"Episode reward: {episode_reward:.2f}\n"
                f"Episode length: {episode_steps}\n"
                f"Rolling avg reward: {np.mean(rolling_reward):.2f}")
            
            # Evaluation phase
            if episode % self.eval_interval == 0:
                eval_reward, eval_std = self.evaluate_policy()
                self.eval_rewards_history.append(eval_reward)
                
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.agent.save(f"{self.save_dir}/best_model.pt")
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                self.save_training_history()
            
            # Early stopping check
            if no_improvement_count >= early_stop_patience:
                self.debug_print('episode_progress',
                    "No improvement for a while. Stopping training.")
                break
            
            # Success criterion check
            if np.mean(rolling_reward) > 300:
                self.debug_print('episode_progress',
                    "Environment solved! Stopping training.")
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