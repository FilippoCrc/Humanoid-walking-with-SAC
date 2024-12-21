import os
import gymnasium as gym
import torch
import numpy as np
from trainer import SACTrainer
from gymnasium.wrappers import RecordVideo
import time
import argparse

# Configure MuJoCo with EGL backend if GPU is available
if torch.cuda.is_available():
    NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
        with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
            f.write("""{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}""")
    os.environ["MUJOCO_GL"] = "egl"

def capped_cubic_video_schedule(episode_id: int) -> bool:
    """Video recording schedule for evaluation"""
    return episode_id % 10 == 0  
MODEL_PATH_EVAL = "results\sac_NaoWalk-v1_1734781102/best_model.pt"
class NaoWalkSACTrainer(SACTrainer):
    def __init__(
        self,
        max_episodes=1000,
        max_steps=1000,
        batch_size=256,
        eval_interval=20,
        updates_per_step=1,
        start_steps=10000,
        eval_episodes=5,
        render_eval=True,
        model_dir="results/sac_nao_walk"
    ):
        # Register the environment
        gym.register(
            id="NaoWalk-v1",
            entry_point="spqr_rl_mujoco.envs.walk_env:NaoWalk",
            max_episode_steps=1000,
        )

        self.model_dir = model_dir
        self.best_model_path = os.path.join(model_dir, "best_model.pt")
        
        # Initialize parent class with Nao environment
        super().__init__(
            env_name="NaoWalk-v1",
            max_episodes=max_episodes,
            max_steps=max_steps,
            batch_size=batch_size,
            eval_interval=eval_interval,
            updates_per_step=updates_per_step,
            start_steps=start_steps,
            eval_episodes=eval_episodes
        )

        # Create video directory
        self.video_dir = f"videos/sac_nao_walk_{int(time.time())}"
        os.makedirs(self.video_dir, exist_ok=True)

        # Create two separate evaluation environments
        self.eval_env = gym.make("NaoWalk-v1")  # Clean env for evaluation
        
        # Optional separate environment for video recording
        if render_eval:
            self.video_env = gym.make("NaoWalk-v1", render_mode="rgb_array")
            self.video_env = RecordVideo(
                self.video_env,
                video_folder=self.video_dir,
                episode_trigger=capped_cubic_video_schedule,
                name_prefix="eval"
            )

        # Set up metrics tracking
        self.episode_metrics = []

    def evaluate_policy(self):
        """Evaluation method used during training"""
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
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                if start_x is None and 'x_position' in info:
                    start_x = info['x_position']
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
            
            total_distance = info['x_position'] - start_x if start_x is not None else 0
            avg_velocity = total_distance / episode_steps if episode_steps > 0 else 0
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_steps)
            forward_distances.append(total_distance)
            avg_velocities.append(avg_velocity)

            # Store metrics
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

            # Record video if configured
            if self.video_env is not None:
                try:
                    video_state, _ = self.video_env.reset()
                    video_done = False
                    video_steps = 0
                    
                    while not video_done and video_steps < episode_steps:
                        video_action = self.agent.select_action(video_state, evaluate=True)
                        video_state, _, video_terminated, video_truncated, _ = self.video_env.step(video_action)
                        video_done = video_terminated or video_truncated
                        video_steps += 1
                except Exception as e:
                    print(f"Warning: Failed to record video: {e}")

        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        mean_distance = np.mean(forward_distances)
        mean_velocity = np.mean(avg_velocities)

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
        """Separate evaluation method with human rendering for visual inspection"""
        print("\nStarting human-rendered evaluation...")
        
        # Create a new environment with human rendering
        render_env = gym.make("NaoWalk-v1", render_mode='human')
        
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
        """Save evaluation metrics to a file"""
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

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate SAC on NAO Walking')
    parser.add_argument('--render', action='store_true', 
                       help='Render the environment during evaluation')
    parser.add_argument('--train', action='store_true',
                       help='Train the SAC agent')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the trained agent')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to load a specific model for evaluation')
    
    args = parser.parse_args()

    # Initialize the trainer
    trainer = NaoWalkSACTrainer(
        max_episodes=20000,
        max_steps=1000,
        batch_size=256,
        eval_interval=20,
        updates_per_step=1,
        start_steps=0,
        eval_episodes=5,
        render_eval=True
    )

    if args.train:
        print("Starting SAC training for Nao Walking...")
        trainer.train()
        print(f"Training completed. Final model saved to {trainer.best_model_path}")

    if args.evaluate:
        print("\nStarting evaluation phase...")
        model_path = MODEL_PATH_EVAL
        
        if os.path.exists(model_path):
            trainer.agent.load(model_path)
            print(f"Model loaded successfully from {model_path}!")
            if args.render:
                trainer.evaluate_with_render(episodes=args.episodes)
            else:
                trainer.evaluate_policy()
        else:
            print(f"Error: Model file not found at: {model_path}")
            print("Please make sure you have trained the model first or provide a valid model path.")

    # Close environments
    trainer.env.close()
    trainer.eval_env.close()

if __name__ == "__main__":
    main()