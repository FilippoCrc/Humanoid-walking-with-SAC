import argparse
import random
import numpy as np
from sac import Policy
import gymnasium as gym

def evaluate(env=None, n_episodes=1, render=False):
    """
    Evaluates the trained policy on the BipedalWalker environment.
    The environment provides a 24-dimensional state space and expects
    a 4-dimensional continuous action space for the joint motors.
    """
    agent = Policy()
    agent.load()  # Load the trained SAC policy

    # Create environment with appropriate render mode
    if render:
        env = gym.make('BipedalWalker-v3', render_mode='human')
    else:
        env = gym.make('BipedalWalker-v3')
        
    rewards = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        s, _ = env.reset()  # Reset returns initial state and info dict
        
        while not done:
            # Get continuous actions for the 4 joint motors
            action = agent.act(s)
            
            # Step through environment - BipedalWalker gives rich reward signal
            s, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        rewards.append(total_reward)
        
    print('Mean Reward:', np.mean(rewards))
    print('Std Reward:', np.std(rewards))  # Added to track consistency


def train():
    """
    Trains the SAC agent on the BipedalWalker environment.
    The training loop and hyperparameter updates are handled in the Policy class.
    """
    agent = Policy()
    agent.train()  # This will implement the SAC training loop
    agent.save()   # Save the trained policy networks


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate SAC on BipedalWalker')
    parser.add_argument('--render', action='store_true', 
                      help='Render the environment during evaluation')
    parser.add_argument('-t', '--train', action='store_true',
                      help='Train the SAC agent')
    parser.add_argument('-e', '--evaluate', action='store_true',
                      help='Evaluate the trained agent')
    parser.add_argument('--episodes', type=int, default=1,
                      help='Number of evaluation episodes')
    args = parser.parse_args()

    if args.train:
        train()

    if args.evaluate:
        evaluate(n_episodes=args.episodes, render=args.render)

    
if __name__ == '__main__':
    main()