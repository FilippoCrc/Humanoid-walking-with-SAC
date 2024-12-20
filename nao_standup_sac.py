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
    if episode_id < 10000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 10000 == 0

class NaoStandupSACTrainer(SACTrainer):
    """Extended SACTrainer specifically for Nao Standup environment"""
    def __init__(
        self,
        max_episodes=1000,
        max_steps=1000,  # Matches the max_episode_steps from environment registration
        batch_size=256,
        eval_interval=10,
        updates_per_step=1,
        start_steps=10000,
        eval_episodes=5,
        render_eval=True
    ):
        # Register the environment
        gym.register(
            id="NaoStandup-v1",
            entry_point="spqr_rl_mujoco.envs.getup_env:NaoStandup",
            max_episode_steps=2500,
        )

        # Initialize parent class with Nao environment
        super().__init__(
            env_name="NaoStandup-v1",
            max_episodes=max_episodes,
            max_steps=max_steps,
            batch_size=batch_size,
            eval_interval=eval_interval,
            updates_per_step=updates_per_step,
            start_steps=start_steps,
            eval_episodes=eval_episodes
        )

        # Create video directory
        self.video_dir = f"videos/sac_nao_{int(time.time())}"
        os.makedirs(self.video_dir, exist_ok=True)

        # Set up video recording for evaluation with proper render mode
        if render_eval:
            # Recreate eval environment with render mode
            self.eval_env = gym.make("NaoStandup-v1", render_mode="rgb_array")
            self.eval_env = RecordVideo(
                self.eval_env,
                video_folder=self.video_dir,
                episode_trigger=capped_cubic_video_schedule,
                name_prefix="eval"
            )

        # Set up metrics tracking
        self.episode_metrics = []

    def evaluate_policy(self):
        """Override evaluation to include additional Nao-specific metrics"""
        eval_rewards = []
        eval_lengths = []
        head_heights = []  # Track the maximum height achieved by the head
        
        for eval_ep in range(self.eval_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            episode_steps = 0
            max_head_height = float('-inf')
            done = False
            
            while not done:
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                # print("Reward components:", info)  # This will show uph_cost, quad_ctrl_cost, and quad_impact_cost
                done = terminated or truncated
                
                # Track head height (available in the reward calculation)
                if 'reward_linup' in info:
                    current_head_height = info['reward_linup']
                    max_head_height = max(max_head_height, current_head_height)
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_steps)
            head_heights.append(max_head_height)

            # Store episode metrics
            self.episode_metrics.append({
                'episode': len(self.episode_metrics),
                'reward': episode_reward,
                'length': episode_steps,
                'max_head_height': max_head_height,
                'impact_cost': info.get('reward_impact', 0),
                'ctrl_cost': info.get('reward_quadctrl', 0)
            })

        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        mean_height = np.mean(head_heights)

        if self.debug_config['print_eval_summary']:
            print(f"\n{'='*50}")
            print(f"Evaluation Summary:")
            print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"Mean Episode Length: {np.mean(eval_lengths):.1f}")
            print(f"Mean Max Head Height: {mean_height:.3f}")
            print(f"Success Rate: {sum(r > 300 for r in eval_rewards)/len(eval_rewards):.2%}")
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
                f.write(f"  Max Head Height: {metric['max_head_height']:.3f}\n")
                f.write(f"  Impact Cost: {metric['impact_cost']:.3f}\n")
                f.write(f"  Control Cost: {metric['ctrl_cost']:.3f}\n")
                f.write("-" * 30 + "\n")

def main():
    # Initialize the trainer with custom parameters
    trainer = NaoStandupSACTrainer(
        max_episodes=2000,          # Increased episodes for complex task
        max_steps=1000,            
        batch_size=256,
        eval_interval=10,          # Evaluate every 20 episodes
        updates_per_step=1,
        start_steps=10000,         # Initial random actions for exploration
        eval_episodes=5,           # Number of episodes for evaluation
        render_eval=True           # Enable video recording
    )

    # Train the agent
    print("Starting SAC training for Nao Standup...")
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