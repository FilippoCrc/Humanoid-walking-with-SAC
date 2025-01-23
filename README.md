# Soft Actor-Critic (SAC) Implementation for Robotic Control

## Overview
This project implements the Soft Actor-Critic (SAC) algorithm for training robotic agents. The implementation specifically focuses on four environments:
- BipedalWalker-v3: A 2D bipedal robot learning to walk from box2d by gym
- Humanoid : A 3D humanoid robot from mujoco 
- NAO Humanoid Walking: A humanoid robot learning bipedal locomotion
- NAO Standup: A humanoid robot learning to stand up from a prone position

## Features
- Modular implementation of SAC with separate policy and value networks
- Support for automatic entropy tuning
- Checkpoint system for saving and loading training progress
- Detailed metrics tracking and visualization tools
- Video recording capabilities for evaluation episodes
- Support for both standard and prioritized experience replay
- GPU acceleration support with CUDA

## Project Structure

main.py                 # Main training script for BipedalWalker
main_nao_standup.py     # NAO standup training implementation
main_nao_walk.py        # NAO walking training implementation
networks_model1.py      # Neural network architectures (Version 1)
networks_model2.py      # Enhanced neural network architectures (Version 2)
replay_buffer.py        # Experience replay implementations
sac_imp.py              # Core SAC algorithm implementation
trainer.py              # Training loop and utilities
utility.py              # Visualization and helper functions

## Usage

### Training a Model
You can train models using any of the main scripts with various command-line arguments:

```bash
# Train BipedalWalker, walker2d and Humanoid
# change also the env names in the file
python main.py --train

# Train NAO to walk
python main_nao_walk.py --train

# Train NAO to stand up
python main_nao_standup.py --train
```

### Evaluation
To evaluate a trained model:

```bash
# Evaluate with rendering
python main.py --evaluate --render

# Evaluate without rendering
python main.py --evaluate
```

## Implementation Details

### SAC Architecture
The implementation follows the original SAC paper with several enhancements:
- Dual Q-networks to reduce overestimation bias
- Automatic entropy tuning
- Gaussian policy with state-dependent mean and standard deviation

### Network Architectures
- **Policy Network**: A Gaussian policy network that outputs mean and log standard deviation
- **Q-Networks**: Dual Q-networks with shared architecture but independent parameters
- **Features**:
  - ReLU activation functions
  - Layer normalization
  - Xavier/Orthogonal initialization
  - Adjustable hidden dimensions

### Training Features
- **Experience Replay**: Both standard and prioritized replay buffers
- **Automatic Entropy**: Dynamic adjustment of temperature parameter
- **Early Stopping**: Training stops if no improvement is seen after a set number of episodes
- **Checkpointing**: Regular saving of model state and training progress

### Visualization and Monitoring
The `TrainingVisualizer` class provides comprehensive visualization tools:
- Training progress plots
- Episode length analysis
- Statistical summaries
- Performance metrics tracking

## Performance Metrics
The implementation tracks several key metrics:
- Episode rewards and lengths
- Q-value losses
- Policy losses
- Action statistics
- Forward progress (for walking tasks)
- Success rates

## Configuration
Key hyperparameters can be adjusted in the respective main scripts:
- Learning rate: 3e-4
- Batch size: 256
- Hidden layer dimensions: 256/512
- Discount factor (gamma): 0.99
- Soft update coefficient (tau): 0.005
- Initial temperature: 0.2

## Authors
Filippo Croce
Federico Occelli

## Acknowledgments
This implementation is based on the original SAC papers:
- "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" by Tuomas Haarnoja et al. https://arxiv.org/abs/1801.01290
- Soft Actor-Critic Algorithms and Applications by Haarnoja et al. (2019): https://arxiv.org/abs/1812.05905