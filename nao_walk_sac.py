import os
import gymnasium as gym
import torch
import numpy as np
from trainer import SACTrainer
from gymnasium.wrappers import RecordVideo
import argparse
from utility import capped_cubic_video_schedule

""" GLOBAL VARIABLES"""
MODEL_PATH_EVAL = "results\sac_nao_walk\best_model.pt"
MODEL_PATH = "results/sac_nao_walk"
ENV_NAME = "NaoWalk-v1"

# Configure MuJoCo with EGL backend if GPU is available
if torch.cuda.is_available():
     os.environ["MUJOCO_GL"] = "glfw"

# Define the NAO Walk SAC Trainer super-class
class NaoWalkSACTrainer(SACTrainer):
    def __init__(
        self,
        max_episodes=1000,
        max_steps=1000,
        batch_size=256,
        eval_interval=50,
        updates_per_step=1,
        start_steps=10000,
        eval_episodes=50,
        render_eval=True,
        model_dir=MODEL_PATH,  # Changed default directory
        #checkpoint_interval=50  # Change when to save checkpoints
    ):
        # Set up fixed directories
        self.save_dir = model_dir
        self.model_dir = model_dir
        self.best_model_path = os.path.join(self.save_dir, "best_model.pt")
        self.video_dir = os.path.join(self.save_dir, "videos")
        
        # Create directories if they don't exist, but don't clean them
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        # Print directory information
        print("\nTraining Directory Setup:")
        print(f"Main Directory: {self.save_dir}")
        print(f"Checkpoint Location: {os.path.join(self.save_dir, 'checkpoint_latest.pt')}")
        print(f"Video Directory: {self.video_dir}")
        
        # Register the NAO environment
        gym.register(
            id=ENV_NAME,
            entry_point="spqr_rl_mujoco.envs.walk_env:NaoWalk",
            max_episode_steps=1000,
        )

        # Initialize parent class
        super().__init__(
            env_name=ENV_NAME,
            max_episodes=max_episodes,
            max_steps=max_steps,
            batch_size=batch_size,
            eval_interval=eval_interval,
            updates_per_step=updates_per_step,
            start_steps=start_steps,
            eval_episodes=eval_episodes
        )

        #self.checkpoint_interval = checkpoint_interval

        # Create evaluation environments
        self.eval_env = gym.make(ENV_NAME)
        
        if render_eval:
            self.video_env = gym.make(ENV_NAME, render_mode="rgb_array")
            self.video_env = RecordVideo(
                self.video_env,
                video_folder=self.video_dir,
                episode_trigger=capped_cubic_video_schedule,
                name_prefix="eval"
            )
        else:
            self.video_env = None

        # Initialize metrics tracking
        self.episode_metrics = []
        self.best_eval_reward = float('-inf')

        # Print training configuration
        print("\nInitializing NAO Walk Training:")
        print(f"Save Directory: {self.save_dir}")
        print(f"Video Directory: {self.video_dir}")
        print(f"Max Episodes: {max_episodes}")
        #print(f"Checkpoint Interval: {checkpoint_interval}")
        print(f"Evaluation Interval: {eval_interval}")
        print("=" * 50)

    def save_checkpoint(self, episode, total_steps):
        """
        Saves a complete checkpoint with verification.
        """
        print(f"\nSaving checkpoint for episode {episode}...")
        
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_latest.pt")
        metrics_path = os.path.join(self.save_dir, "latest_metrics.pt")
        
        # Save detailed training state
        training_state = {
            'current_episode': episode,
            'total_steps': total_steps,
            'rewards_history': self.rewards_history,
            'eval_rewards_history': self.eval_rewards_history,
            'episode_length_history': self.episode_length_history,
            'loss_history': self.loss_history,
            'best_eval_reward': self.best_eval_reward,
            'episode_metrics': self.episode_metrics
        }
        
        # Save agent state
        self.agent.save_checkpoint(checkpoint_path, episode, total_steps)
        
        # Save training metrics
        torch.save(training_state, metrics_path)
        
        print(f"Checkpoint saved:")
        print(f"- Episode: {episode}")
        print(f"- Total steps: {total_steps}")
        #print(f"- Checkpoint path: {checkpoint_path}")
        #print(f"- Metrics path: {metrics_path}")

    def load_checkpoint(self, checkpoint_path=None):
        """
        Loads the most recent checkpoint with detailed verification printing.
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.save_dir, "checkpoint_latest.pt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        print("\nLoading checkpoint from:", checkpoint_path)
        
        # Load training metrics first
        metrics_path = os.path.join(os.path.dirname(checkpoint_path), "latest_metrics.pt")
        if os.path.exists(metrics_path):
            print("Found metrics file:", metrics_path)
            training_state = torch.load(metrics_path)
            
            # Load and verify all training state
            self.rewards_history = training_state['rewards_history']
            self.eval_rewards_history = training_state['eval_rewards_history']
            self.episode_length_history = training_state['episode_length_history']
            self.loss_history = training_state['loss_history']
            self.best_eval_reward = training_state['best_eval_reward']
            self.episode_metrics = training_state['episode_metrics']
            
            current_episode = training_state['current_episode']
            total_steps = training_state['total_steps']
            
            print(f"Loaded training state:")
            print(f"- Current episode: {current_episode}")
            print(f"- Total steps: {total_steps}")
            print(f"- History length: {len(self.rewards_history)} episodes")
            print(f"- Best reward so far: {self.best_eval_reward:.2f}")
        else:
            print("Warning: No metrics file found!")
            return 0, 0
        
        # Now load the agent's state
        self.agent.load_checkpoint(checkpoint_path)
        print("Agent state loaded successfully")
        
        return current_episode, total_steps

    def evaluate_policy(self):
        """
        Evaluates the current policy by running multiple episodes and collecting metrics.
        Returns mean and standard deviation of evaluation rewards.
        """
        eval_rewards = []
        eval_lengths = []
        forward_distances = []
        avg_velocities = []
        
        for eval_ep in range(self.eval_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            episode_steps = 0
            start_x = None
            done = False
            
            while not done:
                # Get action from the policy and take step in environment
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                # Track initial position for distance calculation
                if start_x is None and 'x_position' in info:
                    start_x = info['x_position']
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
            
            # Calculate performance metrics
            total_distance = info['x_position'] - start_x if start_x is not None else 0
            avg_velocity = total_distance / episode_steps if episode_steps > 0 else 0
            
            # Store metrics
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_steps)
            forward_distances.append(total_distance)
            avg_velocities.append(avg_velocity)

            # Record detailed metrics for this evaluation episode
            self.episode_metrics.append({
                'episode': len(self.episode_metrics),
                'reward': episode_reward,
                'length': episode_steps,
                'distance': total_distance,
                'avg_velocity': avg_velocity,
                'forward_reward': info.get('reward_forward', 0),
                'ctrl_cost': info.get('reward_ctrl', 0),
                'contact_cost': info.get('reward_contact', 0),
                'alive_bonus': info.get('reward_alive', 0)
            })

            # Record evaluation video if enabled
            if self.video_env is not None:
                self._record_evaluation_video(episode_steps)

        # Calculate summary statistics
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        mean_distance = np.mean(forward_distances)
        mean_velocity = np.mean(avg_velocities)

        # Print evaluation summary if enabled
        if self.debug_config['print_eval_summary']:
            print(f"\n{'='*50}")
            print(f"Training Evaluation Summary:")
            print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"Mean Episode Length: {np.mean(eval_lengths):.1f}")
            print(f"Mean Distance: {mean_distance:.3f}")
            print(f"Mean Velocity: {mean_velocity:.3f}")
            print(f"Success Rate: {sum(d > 1.0 for d in forward_distances)/len(forward_distances):.2%}")
            print(f"{'='*50}")

        self._save_metrics()
        return mean_reward, std_reward
    def evaluate_with_render(self, episodes=10):
        """
        Evaluates the policy with human-viewable rendering for visual inspection.
        
        Args:
            episodes (int): Number of episodes to evaluate
        """
        print("\nStarting human-rendered evaluation...")
        render_env = gym.make(ENV_NAME, render_mode='human')
        
        rewards = []
        steps = []
        distances = []
        velocities = []
        
        try:
            for episode in range(episodes):
                state, _ = render_env.reset()
                episode_reward = 0
                episode_steps = 0
                start_x = None
                done = False
                
                while not done:
                    action = self.agent.select_action(state, evaluate=True)
                    next_state, reward, terminated, truncated, info = render_env.step(action)
                    done = terminated or truncated
                    
                    if start_x is None and 'x_position' in info:
                        start_x = info['x_position']
                    
                    episode_reward += reward
                    episode_steps += 1
                    state = next_state
                
                total_distance = info['x_position'] - start_x if start_x is not None else 0
                avg_velocity = total_distance / episode_steps if episode_steps > 0 else 0
                
                rewards.append(episode_reward)
                steps.append(episode_steps)
                distances.append(total_distance)
                velocities.append(avg_velocity)
                
                print(f"Episode {episode + 1}/{episodes} - "
                      f"Reward: {episode_reward:.2f} - "
                      f"Steps: {episode_steps} - "
                      f"Distance: {total_distance:.3f} - "
                      f"Velocity: {avg_velocity:.3f}")
            
            # Print evaluation summary
            print("\nHuman Render Evaluation Summary:")
            print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"Average Steps: {np.mean(steps):.1f}")
            print(f"Average Distance: {np.mean(distances):.3f}")
            print(f"Average Velocity: {np.mean(velocities):.3f}")
            print(f"Success Rate: {sum(d > 1.0 for d in distances)/len(distances):.2%}")
            
        finally:
            render_env.close()

    def _save_metrics(self):
        """Saves detailed evaluation metrics to a file."""
        metrics_path = os.path.join(self.save_dir, "evaluation_metrics.txt")
        with open(metrics_path, 'w') as f:
            for metric in self.episode_metrics:
                f.write(f"Episode {metric['episode']}:\n")
                f.write(f"  Reward: {metric['reward']:.2f}\n")
                f.write(f"  Length: {metric['length']}\n")
                f.write(f"  Distance: {metric['distance']:.3f}\n")
                f.write(f"  Average Velocity: {metric['avg_velocity']:.3f}\n")
                f.write(f"  Forward Reward: {metric['forward_reward']:.3f}\n")
                f.write(f"  Control Cost: {metric['ctrl_cost']:.3f}\n")
                f.write(f"  Contact Cost: {metric['contact_cost']:.3f}\n")
                f.write(f"  Alive Bonus: {metric['alive_bonus']:.3f}\n")
                f.write("-" * 30 + "\n")

    def _record_evaluation_video(self, max_steps):
        """Records a video of an evaluation episode."""
        try:
            state, _ = self.video_env.reset()
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                action = self.agent.select_action(state, evaluate=True)
                state, _, terminated, truncated, _ = self.video_env.step(action)
                done = terminated or truncated
                steps += 1
        except Exception as e:
            print(f"Warning: Failed to record video: {e}")

    def save_best_model(self):
        """
        Saves the best performing model during training.
        This method is called whenever the agent achieves a new best evaluation reward.
        The saved model can be used later for deployment or continued training.
        """
        # Create the models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
    
        # Save the model
        self.agent.save(self.best_model_path)
    
        # Log the save event
        print(f"\nNew best model saved to: {self.best_model_path}")
    
        """ # Optionally save additional metrics about this best model
        best_model_metrics_path = os.path.join(self.save_dir, "best_model_metrics.txt")
        with open(best_model_metrics_path, 'w') as f:
            f.write(f"Best Model Metrics:\n")
            f.write(f"Evaluation Reward: {self.best_eval_reward:.2f}\n")
            f.write(f"Saved at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n") """




def main():
    parser = argparse.ArgumentParser(description='Train and evaluate SAC on NAO Walking')
    parser.add_argument('--train', action='store_true', help='Train the SAC agent')
    parser.add_argument('--render', action='store_true', help='Render the environment during evaluation')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the trained agent')
    parser.add_argument('--checkpoint-path', type=str, help='Path to checkpoint file to resume training')
    args = parser.parse_args()

    #actual class callback
    trainer = NaoWalkSACTrainer(
        max_episodes=20000,
        max_steps=1000,
        batch_size=256,
        eval_interval=50,
        updates_per_step=1,
        start_steps=0,
        eval_episodes=5,
        render_eval=True
    )

    if args.train:
        if args.checkpoint_path:
            print("\nAttempting to resume training from checkpoint...")
            print(f"Checkpoint path: {args.checkpoint_path}")
            
            # Load checkpoint and get the starting episode
            start_episode, total_steps = trainer.load_checkpoint(args.checkpoint_path)
            
            # Verify loaded state
            print("\nTraining will resume from:")
            print(f"Episode: {start_episode}")
            print(f"Total Steps: {total_steps}")
            input("Press Enter to continue or Ctrl+C to abort...")
            
            # Pass the correct parameters to train
            trainer.train(start_episode=start_episode, total_steps=total_steps)
        else:
            print("\nStarting new training session...")
            trainer.train()

    if args.evaluate:
        if not os.path.exists(trainer.best_model_path):
            print(f"Error: No model found at {trainer.best_model_path}")
            return
        
        print("\nLoading best model for evaluation...")
        trainer.load_checkpoint(trainer.best_model_path)
        trainer.evaluate_with_render(episodes=10)

if __name__ == "__main__":
    main()