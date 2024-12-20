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
    return episode_id % 20 == 0  # Record every 20 episodes

class NaoStandupSACTrainer(SACTrainer):
    """Extended SACTrainer specifically for Nao Standup environment"""
    def __init__(
        self,
        max_episodes=1000,
        max_steps=1000,
        batch_size=256,
        eval_interval=10,
        updates_per_step=1,
        start_steps=10000,
        eval_episodes=5,
        render_eval=True,
        model_dir="results/sac_nao"
    ):
        # Register the environment
        gym.register(
            id="NaoStandup-v1",
            entry_point="spqr_rl_mujoco.envs.getup_env:NaoStandup",
            max_episode_steps=2500,
        )

        self.model_dir = model_dir
        self.best_model_path = os.path.join(model_dir, "best_model.pt")

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
        head_heights = []
        
        for eval_ep in range(self.eval_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            episode_steps = 0
            max_head_height = float('-inf')
            done = False
            
            while not done:
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                if 'reward_linup' in info:
                    current_head_height = info['reward_linup']
                    max_head_height = max(max_head_height, current_head_height)
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_steps)
            head_heights.append(max_head_height)

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

        self._save_metrics()
        return mean_reward, std_reward

    def evaluate_with_render(self, episodes=10):
        """Evaluate the agent with human rendering"""
        print("\nStarting human-rendered evaluation...")
        
        # Create a new environment with human rendering
        render_env = gym.make("NaoStandup-v1", render_mode='human')
        
        rewards = []
        steps = []
        head_heights = []
        
        try:
            for episode in range(episodes):
                state, _ = render_env.reset()
                episode_reward = 0
                episode_steps = 0
                max_head_height = float('-inf')
                done = False
                
                while not done:
                    action = self.agent.select_action(state, evaluate=True)
                    next_state, reward, terminated, truncated, info = render_env.step(action)
                    done = terminated or truncated
                    
                    if 'reward_linup' in info:
                        current_head_height = info['reward_linup']
                        max_head_height = max(max_head_height, current_head_height)
                    
                    episode_reward += reward
                    episode_steps += 1
                    state = next_state
                
                rewards.append(episode_reward)
                steps.append(episode_steps)
                head_heights.append(max_head_height)
                
                print(f"Episode {episode + 1}/{episodes} - "
                      f"Reward: {episode_reward:.2f} - "
                      f"Steps: {episode_steps} - "
                      f"Max Height: {max_head_height:.3f}")
            
            # Print evaluation summary
            print("\nEvaluation Summary:")
            print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"Average Steps: {np.mean(steps):.1f}")
            print(f"Average Max Height: {np.mean(head_heights):.3f}")
            print(f"Success Rate: {sum(r > 300 for r in rewards)/len(rewards):.2%}")
            
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
                f.write(f"  Max Head Height: {metric['max_head_height']:.3f}\n")
                f.write(f"  Impact Cost: {metric['impact_cost']:.3f}\n")
                f.write(f"  Control Cost: {metric['ctrl_cost']:.3f}\n")
                f.write("-" * 30 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate SAC on NAO Standup')
    parser.add_argument('--render', action='store_true', 
                       help='Render the environment during evaluation')
    parser.add_argument('--train', action='store_true',
                       help='Train the SAC agent')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the trained agent')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()

    # Initialize the trainer
    trainer = NaoStandupSACTrainer(
        max_episodes=2000,
        max_steps=1000,
        batch_size=256,
        eval_interval=10,
        updates_per_step=1,
        start_steps=10000,
        eval_episodes=5,
        render_eval=True
    )

    if args.train:
        print("Starting SAC training for Nao Standup...")
        trainer.train()
        print(f"Training completed. Final model saved to {trainer.best_model_path}")

    if args.evaluate:
        print("\nStarting evaluation phase...")
        if os.path.exists(trainer.best_model_path):
            trainer.agent.load(trainer.best_model_path)
            print("Model loaded successfully!")
            if args.render:
                trainer.evaluate_with_render(episodes=args.episodes)
            else:
                trainer.evaluate_policy()
        else:
            print(f"Error: Model file not found at: {trainer.best_model_path}")
            print("Please make sure you have trained the model first.")

    # Close environments
    trainer.env.close()
    trainer.eval_env.close()

if __name__ == "__main__":
    main()