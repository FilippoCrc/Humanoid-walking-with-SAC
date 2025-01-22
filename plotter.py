import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open('results/training_history_humanoidv4_second_model.json', 'r') as f:
    data = json.load(f)

# Extract rewards and eval_rewards
training_rewards = data['rewards']
eval_rewards = data['eval_rewards']

# Create x-axes
training_episodes = np.arange(len(training_rewards))
eval_episodes = np.linspace(0, len(training_rewards), len(eval_rewards))

# Create the plot
plt.figure(figsize=(12, 6))

# Plot training rewards
plt.plot(training_episodes, training_rewards, 'b-', alpha=0.5, label='Training Rewards')

# Plot evaluation rewards
plt.plot(eval_episodes, eval_rewards, 'r-', label='Evaluation Rewards')

# Add labels and title
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Bipedal Walker Training Progress')
plt.legend()

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Optional: Add a horizontal line at y=0 to show the baseline
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()

# Print some statistics
print("\nTraining Statistics:")
print(f"Average training reward: {np.mean(training_rewards):.2f}")
print(f"Max training reward: {np.max(training_rewards):.2f}")
print(f"Min training reward: {np.min(training_rewards):.2f}")
print("\nEvaluation Statistics:")
print(f"Average evaluation reward: {np.mean(eval_rewards):.2f}")
print(f"Max evaluation reward: {np.max(eval_rewards):.2f}")
print(f"Min evaluation reward: {np.min(eval_rewards):.2f}")