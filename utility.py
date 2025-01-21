import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd
from pathlib import Path

def capped_cubic_video_schedule(episode_id: int) -> bool:
    """Video recording schedule for evaluation"""
    return episode_id % 10 == 0  

class TrainingVisualizer:
    """
    A class to create comprehensive visualizations of SAC training results.
    This includes reward plots, episode lengths, and training statistics.
    """
    
    def __init__(self, json_path: str):
        """
        Initialize the visualizer with training data.
        
        Args:
            json_path: Path to the training history JSON file
        """
        # Set up the visual style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Load and process the training data
        self.data = self._load_training_data(json_path)
        self.statistics = self._calculate_statistics()
        
        # Calculate moving averages for smoothing
        self.moving_avg = self._calculate_moving_average(self.data['rewards'], window=100)
        
    def _load_training_data(self, json_path: str) -> Dict:
        """Load training data from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def _calculate_statistics(self) -> Dict:
        """Calculate key training statistics."""
        return {
            'total_episodes': len(self.data['rewards']),
            'best_reward': max(self.data['rewards']),
            'best_eval_reward': max(self.data['eval_rewards']),
            'avg_last_100': np.mean(self.data['rewards'][-100:]),
            'initial_avg': np.mean(self.data['rewards'][:100]),
            'final_avg': np.mean(self.data['rewards'][-100:])
        }
    
    def _calculate_moving_average(self, values: List[float], window: int) -> np.ndarray:
        """Calculate moving average with specified window size."""
        weights = np.ones(window) / window
        return np.convolve(values, weights, mode='valid')
    
    def create_training_progress_plot(self, save_path: str = None):
        """
        Create a comprehensive plot showing training progress over time.
        Includes episode rewards, evaluation rewards, and moving average.
        """
        plt.figure(figsize=(12, 6))
        
        # Plot episode rewards with low opacity
        plt.plot(self.data['rewards'], alpha=0.3, label='Episode Reward', color='#3b82f6')
        
        # Plot moving average
        ma_x = np.arange(len(self.moving_avg)) + (100 - 1)
        plt.plot(ma_x, self.moving_avg, label='100-Episode Moving Avg', 
                color='#ef4444', linewidth=2)
        
        # Plot evaluation rewards
        eval_x = np.arange(0, len(self.data['rewards']), 50)[:len(self.data['eval_rewards'])]
        plt.plot(eval_x, self.data['eval_rewards'], 'go', label='Evaluation Reward',
                color='#22c55e', markersize=8)
        
        plt.title('Humanoid SAC Training Progress', fontsize=14, pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_episode_length_plot(self, save_path: str = None):
        """Create a plot showing episode lengths over time."""
        plt.figure(figsize=(12, 4))
        
        # Create filled area plot for episode lengths
        plt.fill_between(range(len(self.data['episode_lengths'])), 
                        self.data['episode_lengths'],
                        alpha=0.3, color='#8b5cf6')
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

