import os
import gymnasium as gym
import torch
import numpy as np
from trainer import SACTrainer
from gymnasium.wrappers import RecordVideo
import time

# Configure MuJoCo with EGL backend if GPU is available
if torch.cuda.is_available():
    NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
        with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
            f.write("""{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}""")
    os.environ["MUJOCO_GL"] = "egl"

def capped_cubic_video_schedule(episode_id: int) -> bool:
    """Video recording schedule for evaluation"""
    return episode_id % 40 == 0  # Record every 40 episodes

class NaoWalkSACTrainer(SACTrainer):
    """Extended SACTrainer specifically for Nao Walking environment"""
    def __init__(
        self,
        max_episodes=1000,
        max_steps=1000,
        batch_size=256,
        eval_interval=10,
        updates_per_step=1,
        start_steps=10000,
        eval_episodes=5,
        render_eval=False
    ):
        # Register the environment
        gym.register(
            id="NaoWalk-v1",
            entry_point="spqr_rl_mujoco.envs.walk_env:NaoWalk",
            max_episode_steps=1000,
        )

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
        self.render_eval = render_eval

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
        """Override evaluation to include walking-specific metrics"""
        eval_rewards = []
        eval_lengths = []
        forward_distances = []  # Track distance traveled
        avg_velocities = []    # Track average velocity
        
        for eval_ep in range(self.eval_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            episode_steps = 0
            start_x = None
            done = False
            
            while not done and episode_steps < self.max_steps:
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                # Track position and velocity
                if start_x is None and 'x_position' in info:
                    start_x = info['x_position']
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
            
            # Calculate distance and velocity metrics
            total_distance = info['x_position'] - start_x if start_x is not None else 0
            avg_velocity = total_distance / episode_steps if episode_steps > 0 else 0
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_steps)
            forward_distances.append(total_distance)
            avg_velocities.append(avg_velocity)

            # Store episode metrics
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

            # Record successful episodes if video environment exists
            if self.render_eval and episode_reward > 300:  # Threshold for "good" episodes
                try:
                    # Run the same episode in the video environment
                    video_state, _ = self.video_env.reset()
                    video_done = False
                    video_steps = 0
                    
                    while not video_done and video_steps < episode_steps:
                        video_action = self.agent.select_action(video_state, evaluate=True)
                        video_state, _, video_terminated, video_truncated, _ = self.video_env.step(video_action)
                        video_done = video_terminated or video_truncated
                        video_steps += 1
                except Exception as e:
                    print(f"Warning: Failed to record video for successful episode: {e}")

        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        mean_distance = np.mean(forward_distances)
        mean_velocity = np.mean(avg_velocities)

        if self.debug_config['print_eval_summary']:
            print(f"\n{'='*50}")
            print(f"Evaluation Summary:")
            print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"Mean Episode Length: {np.mean(eval_lengths):.1f}")
            print(f"Mean Distance Traveled: {mean_distance:.3f}")
            print(f"Mean Velocity: {mean_velocity:.3f}")
            print(f"Success Rate: {sum(d > 1.0 for d in forward_distances)/len(forward_distances):.2%}")
            print(f"{'='*50}")

        # Save metrics to file
        self._save_metrics()

        return mean_reward, std_reward

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
    # Initialize the trainer with custom parameters
    trainer = NaoWalkSACTrainer(
        max_episodes=20000,        # Training episodes
        max_steps=1000,           # Steps per episode
        batch_size=256,
        eval_interval=10,         # Evaluate every 10 episodes
        updates_per_step=1,
        start_steps=10000,        # Initial random actions for exploration
        eval_episodes=5,          # Number of episodes for evaluation
        render_eval=True          # Enable video recording
    )

    # Train the agent
    print("Starting SAC training for Nao Walking...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(trainer.save_dir, "final_model.pt")
    trainer.agent.save(final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")

    # Close environments
    trainer.env.close()
    trainer.eval_env.close()

if __name__ == "__main__":
    main()