import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from sac_fc import SACAgent, evaluate
import os

def create_training_loop():
    # Create the environment
    env = gym.make('BipedalWalker-v3')
    
    # Get dimensions of state and action spaces
    state_dim = env.observation_space.shape[0]  # 24 for BipedalWalker
    action_dim = env.action_space.shape[0]      # 4 for BipedalWalker
    
    # Initialize the agent
    agent = SACAgent(state_dim, action_dim)
    
    # Training parameters
    max_episodes = 1000
    max_steps = 1600
    batch_size = 256
    eval_frequency = 10  # Evaluate every 10 episodes
    
    # Lists to store metrics
    training_rewards = []
    eval_rewards = []
    
    # Create directory for saving results
    results_dir = f"sac_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("Starting training...")
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition in replay buffer
            agent.memory.append((state, action, reward, next_state, float(done)))
            
            # Train agent if enough samples are available
            if len(agent.memory) > batch_size:
                agent.train_step(batch_size)
            
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        # Store episode reward
        training_rewards.append(episode_reward)
        
        # Print episode information
        print(f"Episode {episode + 1}/{max_episodes}")
        print(f"Episode Reward: {episode_reward:.2f}")
        print(f"Buffer Size: {len(agent.memory)}")
        print("-" * 50)
        
        # Evaluate agent
        if (episode + 1) % eval_frequency == 0:
            eval_reward = evaluate(env, agent)
            eval_rewards.append(eval_reward)
            
            # Save model checkpoint
            torch.save({
                'episode': episode,
                'policy_state_dict': agent.policy.state_dict(),
                'q_network_state_dict': agent.q_network.state_dict(),
                'policy_optimizer_state_dict': agent.policy_optimizer.state_dict(),
                'q_optimizer_state_dict': agent.q_optimizer.state_dict(),
            }, f'{results_dir}/checkpoint_episode_{episode}.pt')
            
            # Plot and save training curves
            plot_training_curves(training_rewards, eval_rewards, results_dir)

def plot_training_curves(training_rewards, eval_rewards, results_dir):
    plt.figure(figsize=(12, 4))
    
    # Plot training rewards
    plt.subplot(1, 2, 1)
    plt.plot(training_rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot evaluation rewards
    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(eval_rewards) * 10, 10), eval_rewards)
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/learning_curves.png')
    plt.close()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Start training
    create_training_loop()