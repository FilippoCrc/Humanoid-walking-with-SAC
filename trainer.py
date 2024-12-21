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
        env_name='Walker2d-v4',
        max_episodes=1000,
        max_steps=1000,
        batch_size=256,
        eval_interval=10,
        updates_per_step=1,
        start_steps=25000,
        eval_episodes=20,
        debug_config = None
    ):
        
        # First, determine the device at the beginning of the trainer initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # This will help us confirm which device is being used

        # Default debugging configuration with more focused outputs
        self.debug_config = {
            'print_episode_summary': True,     # Summary of each episode
            'print_eval_summary': True,        # Summary of evaluation runs
            'print_periodic_stats': True       # Periodic training statistics
        }
        if debug_config is not None:
            self.debug_config.update(debug_config)

        # Storage for episode statistics
        self.episode_stats = {
            'q1_losses': [],
            'q2_losses': [],
            'policy_losses': [],
            'action_means': [],
            'action_stds': []
        }
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
            hidden_dim=512,      # Increased network size for more complex environment
            gamma=0.99,
            tau=0.005,
            lr=3e-4,
            alpha=0.2,
            automatic_entropy_tuning=True,
            use_per=True,       # PER is disabled by default
            device=self.device
        )
        
        # Initialize logging
        self.rewards_history = []
        self.eval_rewards_history = []
        self.episode_length_history = []
        self.loss_history = []
    
    def print_episode_summary(self, episode, total_steps, episode_reward, episode_length, rolling_reward):
        """Prints a concise summary of the episode"""
        if not self.debug_config['print_episode_summary']:
            return

        # Calculate mean losses for the episode
        mean_losses = {
            'q1': np.mean(self.episode_stats['q1_losses']),
            'q2': np.mean(self.episode_stats['q2_losses']),
            'policy': np.mean(self.episode_stats['policy_losses'])
        }
        
        # Calculate action statistics
        action_means = np.mean(self.episode_stats['action_means'], axis=0)
        action_stds = np.mean(self.episode_stats['action_stds'], axis=0)

        print(f"\n{'='*50}")
        print(f"Episode {episode} Summary:")
        print(f"Steps: {total_steps} | Length: {episode_length}")
        print(f"Reward: {episode_reward:.2f} | Rolling Avg: {rolling_reward:.2f}")
        print(f"Mean Losses - Q1: {mean_losses['q1']:.4f}, Q2: {mean_losses['q2']:.4f}, Policy: {mean_losses['policy']:.4f}")
        print(f"Action Means: {action_means}")
        print(f"Action StDevs: {action_stds}")
        print(f"{'='*50}")

        # Clear episode statistics for next episode
        for key in self.episode_stats:
            self.episode_stats[key] = []

    def update_episode_stats(self, losses, actions):
        """Updates episode statistics during training"""
        if losses:
            self.episode_stats['q1_losses'].append(losses['q1_loss'])
            self.episode_stats['q2_losses'].append(losses['q2_loss'])
            self.episode_stats['policy_losses'].append(losses['policy_loss'])
        
        if actions is not None:
            self.episode_stats['action_means'].append(np.mean(actions, axis=0))
            self.episode_stats['action_stds'].append(np.std(actions, axis=0))

    def debug_print(self, category, message):
        """Utility function for conditional debugging output"""
        if self.debug_config.get(f'print_{category}', False):
            print(f"[DEBUG-{category}] {message}")

    def evaluate_policy(self):
        """Evaluation with concise output"""
        eval_rewards = []
        eval_lengths = []
        
        for eval_ep in range(self.eval_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).to(self.device)
                action = self.agent.select_action(state_tensor, evaluate=True)
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_steps)

        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)

        if self.debug_config['print_eval_summary']:
            print(f"\n{'='*50}")
            print(f"Evaluation Summary:")
            print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"Mean Episode Length: {np.mean(eval_lengths):.1f}")
            print(f"Success Rate: {sum(r > 5000 for r in eval_rewards)/len(eval_rewards):.2%}")
            print(f"{'='*50}")

        return mean_reward, std_reward

    def train(self):
        """
        Main training loop for SAC algorithm.
        Handles episode management, statistics collection, and model evaluation.
        """
        # Initialize training variables
        total_steps = 0
        best_eval_reward = float('-inf')
        early_stop_patience = 1000
        no_improvement_count = 0
        rolling_reward = deque(maxlen=100)
        episode_reward = 0
        episode_steps = 0

        # Print initial training information
        print(f"\nStarting training on {self.env_name}")
        print(f"State dim: {self.env.observation_space.shape[0]}")
        print(f"Action dim: {self.env.action_space.shape[0]}")
        print(f"Training parameters:")
        print(f"  Max episodes: {self.max_episodes}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Updates per step: {self.updates_per_step}")
        print("\n" + "="*50)

        # Create directory for saving results
        self.save_dir = f"results/sac_{self.env_name}_{int(time.time())}"
        os.makedirs(self.save_dir, exist_ok=True)

        for episode in range(self.max_episodes):
            # Reset environment and episode variables
            state, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_actions = []
            done = False
            
            # Reset episode statistics
            for key in self.episode_stats:
                self.episode_stats[key] = []

            while not done:

                state_tensor = torch.FloatTensor(state).to(self.device)

                # Select action: random for exploration or from policy
                if total_steps < self.start_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state_tensor)
                
                # Store action for statistics
                episode_actions.append(action)
                
                # Take action in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_steps += 1
                total_steps += 1
                episode_reward += reward
                
                # Store transition in replay buffer
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update networks if enough samples are gathered
                if len(self.agent.replay_buffer) > self.batch_size:
                    for _ in range(self.updates_per_step):
                        losses = self.agent.update_parameters(self.batch_size)
                        self.update_episode_stats(losses, None)  # Update loss statistics
                
                state = next_state
                
                # End episode if max steps reached
                if episode_steps >= self.max_steps:
                    done = True

            # Update action statistics at episode end
            self.update_episode_stats(None, episode_actions)
            
            # Store episode information
            self.rewards_history.append(episode_reward)
            self.episode_length_history.append(episode_steps)
            rolling_reward.append(episode_reward)
            
            # Print episode summary
            self.print_episode_summary(
                episode, total_steps, episode_reward,
                episode_steps, np.mean(rolling_reward)
            )
            
            # Evaluate policy periodically
            if episode % self.eval_interval == 0 and episode > 0:
                eval_reward, eval_std = self.evaluate_policy()
                self.eval_rewards_history.append(eval_reward)
                
                # Save if best performance
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.agent.save(f"{self.save_dir}/best_model.pt")
                    no_improvement_count = 0
                    print(f"New best model saved with reward: {best_eval_reward:.2f}")
                else:
                    no_improvement_count += 1
                
                # Save training history
                self.save_training_history()
                
            # Early stopping check
            if no_improvement_count >= early_stop_patience:
                print("\nNo improvement for a while. Stopping training.")
                break
            
            # Success criterion check for BipedalWalker
            if np.mean(rolling_reward) > 50000:
                print("\nEnvironment solved! Stopping training.")
                break

        # Final save and summary
        self.agent.save(f"{self.save_dir}/final_model.pt")
        print("\nTraining completed!")
        print(f"Total steps: {total_steps}")
        print(f"Best evaluation reward: {best_eval_reward:.2f}")
        print(f"Final average reward: {np.mean(rolling_reward):.2f}")
    
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

