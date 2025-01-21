import argparse
import numpy as np
import gymnasium as gym
from trainer import SACTrainer
import torch
import os
from utility import TrainingVisualizer

json_path = "results\sac_Humanoid-v5_1734629000\training_history.json"  # Update this path to your JSON file
save_dir = "training_plots"  # Directory where plots will be saved
    

# Define the path to the model
MODEL_DIR = "results\sac_Humanoid-v5_1734629000"  # Change this to your model directory
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
ENV_NAME = "Humanoid-v4"
def main():
    parser = argparse.ArgumentParser(description='Train and evaluate SAC on BipedalWalker')
    
    # Simplified command line arguments
    parser.add_argument('--render', action='store_true', 
                       help='Render the environment during evaluation')
    parser.add_argument('--train', action='store_true',
                       help='Train the SAC agent')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the trained agent')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    # Create the trainer with default parameters
    trainer = SACTrainer(
        env_name=ENV_NAME,
        max_episodes=20000,
        max_steps=1000,
        batch_size=256,
        eval_interval=20,
        updates_per_step=1,
        start_steps=15000,
        eval_episodes=args.episodes
    )

    # Training phase
    if args.train:
        print("\nStarting training phase...")
        trainer.train()
        print("Training completed!")

    # Evaluation phase
    if args.evaluate:
        print("\nStarting evaluation phase...")
        print(f"Using model path: {MODEL_PATH}")
        
        # Set up environment with rendering if specified
        if args.render:
            trainer.eval_env = gym.make(ENV_NAME, render_mode='human')
        
        try:
            # Verify model file exists
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
            
            # Load the trained model
            trainer.agent.load(MODEL_PATH)
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
            
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
            print("Please make sure you have trained the model first and "
                  "the model path is correct.")
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
        
        finally:
            trainer.eval_env.close()


    # Create visualizer and generate all plots
    visualizer = TrainingVisualizer(json_path)
    visualizer.create_full_visualization(save_dir)


if __name__ == '__main__':
    main()