import argparse
import numpy as np
import gymnasium as gym
from trainer import SACTrainer
import torch
import os
from utility import TrainingVisualizer
import json
from datetime import datetime
import time

def create_results_directory(env_name):
    """
    Creates a unique results directory for the current training run.
    Returns the path to the created directory.
    """
    # Create base results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Create unique directory name using timestamp
    timestamp = int(time.time())
    dir_name = f"sac_{env_name}_{timestamp}"
    full_path = os.path.join('results', dir_name)
    
    # Create the directory
    os.makedirs(full_path)
    return full_path

def save_training_history(trainer, save_dir):
    """
    Saves training metrics to disk and generates visualization plots.
    
    Args:
        trainer (SACTrainer): The trainer instance containing training history
        save_dir (str): Directory where the history and plots should be saved
    """
    # Prepare training history dictionary
    history = {
        'rewards': trainer.rewards_history,
        'eval_rewards': trainer.eval_rewards_history,
        'episode_lengths': trainer.episode_length_history,
        'losses': trainer.loss_history,
        'training_params': {
            'env_name': trainer.env_name,
            'max_episodes': trainer.max_episodes,
            'batch_size': trainer.batch_size,
            'updates_per_step': trainer.updates_per_step,
            'start_steps': trainer.start_steps
        }
    }
    
    # Save training history to JSON
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    # Generate and save visualization plots
    visualizer = TrainingVisualizer(history_path)
    visualizer.create_full_visualization(save_dir)
    
    print(f"\nTraining history and plots saved to: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate SAC on specified environment')
    
    # Add command line arguments
    parser.add_argument('--render', action='store_true', 
                       help='Render the environment during evaluation')
    parser.add_argument('--train', action='store_true',
                       help='Train the SAC agent')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the trained agent')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--env', type=str, default='BipedalWalker-v3',
                       help='Gymnasium environment name')
    
    args = parser.parse_args()
    
    # Create results directory for this run
    results_dir = create_results_directory(args.env)
    
    # Create the trainer with default parameters
    trainer = SACTrainer(
        env_name=args.env,
        max_episodes=600,
        max_steps=1000,
        batch_size=256,
        eval_interval=20,
        updates_per_step=1,
        start_steps=1000,
        eval_episodes=args.episodes
    )

    # Training phase
    if args.train:
        print("\nStarting training phase...")
        print(f"Results will be saved to: {results_dir}")
        
        # Train the agent
        trainer.train()
        
        # Save the trained model
        model_path = os.path.join(results_dir, "best_model.pt")
        trainer.agent.save(model_path)
        
        # Save training history and generate plots
        save_training_history(trainer, results_dir)
        
        print("Training completed!")

    # Evaluation phase
    if args.evaluate:
        print("\nStarting evaluation phase...")
        model_path = os.path.join(results_dir, "best_model.pt")
        print(f"Using model path: {model_path}")
        
        # Set up environment with rendering if specified
        if args.render:
            trainer.eval_env = gym.make(args.env, render_mode='human')
        
        try:
            # Verify model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            # Load the trained model
            trainer.agent.load(model_path)
            print("Model loaded successfully!")
            
            # Perform evaluation
            rewards = []
            steps = []
            
            for episode in range(args.episodes):
                state, _ = trainer.eval_env.reset()
                episode_reward = 0
                episode_steps = 0
                done = False
                
                while not done:
                    action = trainer.agent.select_action(state, evaluate=True)
                    next_state, reward, terminated, truncated, _ = trainer.eval_env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    episode_steps += 1
                    state = next_state
                
                rewards.append(episode_reward)
                steps.append(episode_steps)
                
                print(f"Episode {episode + 1}/{args.episodes} - "
                      f"Reward: {episode_reward:.2f} - "
                      f"Steps: {episode_steps}")
            
            # Print evaluation summary
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            mean_steps = np.mean(steps)
            success_rate = sum(r > 300 for r in rewards) / len(rewards)
            
            print("\nEvaluation Summary:")
            print(f"Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"Average Steps: {mean_steps:.1f}")
            print(f"Success Rate: {success_rate:.2%}")
            
            # Save evaluation results
            eval_results = {
                'rewards': rewards,
                'steps': steps,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'mean_steps': float(mean_steps),
                'success_rate': float(success_rate)
            }
            
            eval_path = os.path.join(results_dir, 'evaluation_results.json')
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f)
            
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
            print("Please make sure you have trained the model first and "
                  "the model path is correct.")
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
        
        finally:
            trainer.eval_env.close()

if __name__ == '__main__':
    main()