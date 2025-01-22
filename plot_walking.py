import matplotlib.pyplot as plt
import re

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