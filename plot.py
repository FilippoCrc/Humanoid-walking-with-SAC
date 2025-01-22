import matplotlib.pyplot as plt
import numpy as np
import json

def load_training_history(filepath):
    """
    Loads and returns training history from a JSON file.
    Each JSON file should contain 'rewards' and 'eval_rewards' lists.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_training_metrics(histories, titles, window_size=50, figsize=(15, 10)):
    """
    Creates two subplots: one for training rewards and one for evaluation rewards.
    Applies moving average smoothing to make trends more visible.
    
    Parameters:
    histories -- List of training history dictionaries
    titles -- List of titles for each history (e.g., ['Bipedal Walker', 'Humanoid'])
    window_size -- Size of the moving average window for smoothing
    figsize -- Size of the overall figure
    """
    # Create a figure with two subplots, stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Colors for different environments
    colors = ['blue', 'red', 'green', 'purple']
    
    # Plot training rewards
    for idx, (history, title) in enumerate(zip(histories, titles)):
        episodes = range(len(history['rewards']))
        
        # Plot raw training data with low opacity
        ax1.plot(episodes, history['rewards'], alpha=0.2, color=colors[idx])
        
        # Calculate and plot smoothed training data
        if len(history['rewards']) >= window_size:
            smooth_data = np.convolve(history['rewards'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            smooth_episodes = episodes[window_size-1:]
            ax1.plot(smooth_episodes, smooth_data, 
                    label=f'{title} (MA-{window_size})', 
                    color=colors[idx])
    
    ax1.set_title('Training Rewards Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Training Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot evaluation rewards
    for idx, (history, title) in enumerate(zip(histories, titles)):
        episodes = range(len(history['eval_rewards']))
        
        # Plot raw evaluation data with low opacity
        ax2.plot(episodes, history['eval_rewards'], alpha=0.2, color=colors[idx])
        
        # Calculate and plot smoothed evaluation data
        if len(history['eval_rewards']) >= window_size:
            smooth_data = np.convolve(history['eval_rewards'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            smooth_episodes = episodes[window_size-1:]
            ax2.plot(smooth_episodes, smooth_data, 
                    label=f'{title} (MA-{window_size})', 
                    color=colors[idx])
    
    ax2.set_title('Evaluation Rewards Over Time')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Evaluation Reward')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Load your training histories
try:
    bipedal_history = load_training_history('training_history_bipedal_walker_second model.json')
    humanoid_history = load_training_history('training_history_humanoidv4_second_model.json')
    
    # Create lists of histories and their titles
    histories = [bipedal_history, humanoid_history]
    titles = ['Bipedal Walker', 'Humanoid']
    
    # Create and display the plots
    fig = plot_training_metrics(histories, titles)
    
    # Add a main title to the figure
    plt.suptitle('Training and Evaluation Metrics Comparison', size=16, y=1.02)
    
    # Display the plot window
    plt.show()
    
    # Optionally save the figure to a file
    # fig.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    
except FileNotFoundError as e:
    print(f"Error: Could not find one or more JSON files. Please check the file paths.")
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON format in one of the files. Error details: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")