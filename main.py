import argparse
import numpy as np
import gymnasium as gym
from trainer import SACTrainer

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate SAC on BipedalWalker')
    parser.add_argument('--render', action='store_true', 
                       help='Render the environment during evaluation')
    parser.add_argument('-t', '--train', action='store_true',
                       help='Train the SAC agent')
    parser.add_argument('-e', '--evaluate', action='store_true',
                       help='Evaluate the trained agent')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    args = parser.parse_args()

    # Create the trainer with default parameters
    trainer = SACTrainer(
        env_name='BipedalWalker-v3',
        max_episodes=1000,
        max_steps=1000,
        batch_size=256,
        eval_interval=10,
        updates_per_step=1,
        start_steps=10000,
        eval_episodes=args.episodes
    )

    # When evaluating, we need to recreate the evaluation environment with render mode if specified
    if args.evaluate and args.render:
        trainer.eval_env = gym.make('BipedalWalker-v3', render_mode='human')

    # Handle both training and evaluation based on command line arguments
    if args.train:
        trainer.train()  # This will save the model after training
    
    if args.evaluate:
        trainer.evaluate_policy()  # This will evaluate using the saved model

if __name__ == '__main__':
    main()