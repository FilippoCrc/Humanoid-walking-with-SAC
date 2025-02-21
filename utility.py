import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
import re

# taken from https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/record_video/
def capped_cubic_video_schedule(episode_id: int) -> bool:
    """Video recording schedule for evaluation"""
    return episode_id % 10 == 0  

class TrainingVisualizer:
    
    def __init__(self, json_path: str):
        # Set up the visual style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Load and process the training data
        self.data = self.load_training_data(json_path)
        self.statistics = self.calculate_statistics()
        
        # Calculate moving averages for smoothing
        self.moving_avg = self.calculate_moving_average(self.data['rewards'], window=100)
        
    def load_training_data(self, json_path: str) -> Dict:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def calculate_statistics(self) -> Dict:
        return {
            'total_episodes': len(self.data['rewards']),
            'best_reward': max(self.data['rewards']),
            'best_eval_reward': max(self.data['eval_rewards']),
            'avg_last_100': np.mean(self.data['rewards'][-100:]),
            'initial_avg': np.mean(self.data['rewards'][:100]),
            'final_avg': np.mean(self.data['rewards'][-100:])
        }
    
    def calculate_moving_average(self, values: List[float], window: int) -> np.ndarray:
        """Calculate moving average with specified window size."""
        weights = np.ones(window) / window
        return np.convolve(values, weights, mode='valid')
    
    def create_training_progress_plot(self, save_path: str = None):
        plt.figure(figsize=(12, 6))
        
        # Plot episode rewards with low opacity
        plt.plot(self.data['rewards'], alpha=0.3, label='Episode Reward', color='#3b82f6')
        
        # Plot moving average
        ma_x = np.arange(len(self.moving_avg)) + (100 - 1)
        plt.plot(ma_x, self.moving_avg, label='100-Episode Moving Avg', 
                color='#ef4444', linewidth=2)
        
        # Plot evaluation rewards
        eval_x = np.arange(0, len(self.data['rewards']), 50)[:len(self.data['eval_rewards'])]
        plt.plot(eval_x, self.data['eval_rewards'], 'go', label='Evaluation Reward', color='#22c55e', markersize=8)
        plt.title('Humanoid SAC Training Progress', fontsize=14, pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_episode_length_plot(self, save_path: str = None):
        """ plot showing episode lengths over time."""
        plt.figure(figsize=(12, 4))
        
        # Create filled area plot for episode lengths
        plt.fill_between(range(len(self.data['episode_lengths'])), self.data['episode_lengths'], alpha=0.3, color='#8b5cf6')
        plt.plot(self.data['episode_lengths'], color='#8b5cf6', alpha=0.7)
        plt.title('Episode Lengths Over Training', fontsize=14, pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Steps', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_statistics_summary(self):
        """Print a summary of training statistics."""
        print("\n=== Humanoid SAC Training Summary ===")
        print(f"Total Episodes: {self.statistics['total_episodes']}")
        print(f"Best Training Reward: {self.statistics['best_reward']:.1f}")
        print(f"Best Evaluation Reward: {self.statistics['best_eval_reward']:.1f}")
        print(f"Final 100-Episode Average: {self.statistics['avg_last_100']:.1f}")
        print("\nLearning Progress:")
        print(f"Initial Average (first 100 ep): {self.statistics['initial_avg']:.1f}")
        print(f"Final Average (last 100 ep): {self.statistics['final_avg']:.1f}")
        print(f"Improvement Factor: {self.statistics['final_avg'] / self.statistics['initial_avg']:.1f}x")
        
    def create_full_visualization(self, save_dir: str = None):
        """Create and save all visualizations."""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
            
            self.create_training_progress_plot(save_dir / 'training_progress.png')
            self.create_episode_length_plot(save_dir / 'episode_lengths.png')
        else:
            self.create_training_progress_plot()
            self.create_episode_length_plot()
            
        self.create_statistics_summary()

def parse_evaluation_data(text):
    episodes = []
    rewards = []
    distances = []
    
    # Split the text into episodes
    episode_blocks = text.split('Episode')
    
    for block in episode_blocks[1:]:  # Skip the first empty split
        # Extract episode number
        episode_num = int(block.split(':')[0])
        
        # Extract metrics using regex
        reward_match = re.search(r'Reward: ([-\d.]+)', block)
        distance_match = re.search(r'Distance: ([-\d.]+)', block)
        
        if reward_match and distance_match:
            reward = float(reward_match.group(1))
            distance = float(distance_match.group(1))
            
            episodes.append(episode_num)
            rewards.append(reward)
            distances.append(distance)
    
    return episodes, rewards, distances

def plot_metrics(episodes, rewards, distances):
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])
    fig.suptitle('NAO Walking Metrics Over Episodes', fontsize=14, y=0.95)
    
    # Plot total reward
    ax1.plot(episodes, rewards, color='blue', linewidth=1.5, alpha=0.8)
    ax1.set_title('Total Reward', fontsize=12, pad=10)
    ax1.set_xlabel('Episode', fontsize=10)
    ax1.set_ylabel('Reward', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Plot distance
    ax2.plot(episodes, distances, color='green', linewidth=1.5, alpha=0.8)
    ax2.set_title('Distance', fontsize=12, pad=10)
    ax2.set_xlabel('Episode', fontsize=10)
    ax2.set_ylabel('Distance', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Add horizontal line at y=0 for distance plot to show direction
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    return plt

# Read the file content
with open('results/NAO_WALK_700k/evaluation_metrics.txt', 'r') as file:
    text_content = file.read()

# Parse the data
episodes, rewards, distances = parse_evaluation_data(text_content)

# Create and display the plots
plt = plot_metrics(episodes, rewards, distances)
plt.show()