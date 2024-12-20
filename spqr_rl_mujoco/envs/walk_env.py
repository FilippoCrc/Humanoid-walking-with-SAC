from pathlib import Path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.8925)),
    "elevation": -20.0,
}


class NaoWalk(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, **kwargs) -> None:
        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(661,),
            dtype=np.float64,
        )

        MujocoEnv.__init__(
            self,
            str(Path.cwd().joinpath("spqr_rl_mujoco", "model", "scene.xml")),
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        utils.EzPickle.__init__(self, **kwargs)

    def _get_obs(self) -> np.ndarray:
        data = self.data
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ],
        )

    def step(self, a):
        pos_before = self.data.qpos[0]  # x position before simulation step
        self.do_simulation(a, self.frame_skip)
        data = self.data
        
        # Get the position after simulation step
        pos_after = data.qpos[0]
        
        # Calculate forward velocity
        forward_vel = (pos_after - pos_before) / self.dt
        
        # Get z position of torso
        torso_z = data.qpos[2]
        
        # Calculate aliv bonus (similar to Humanoid-v4)
        alive_bonus = 3.5 #this value must be a lot lower becouse the robot stays on balance much easier, in mujoco it's 5
        
        # Calculate rewards components (similar to Humanoid-v4)
        reward_forward = forward_vel
        reward_alive = alive_bonus
        reward_ctrl = -0.1 * np.square(data.ctrl).sum()
        reward_contact = -0.5e-6 * np.square(data.cfrc_ext).sum()
        reward_contact = np.clip(reward_contact, -10, 10)
        
        # Combine rewards
        reward = reward_forward + reward_alive + reward_ctrl + reward_contact
        
        # Check termination conditions
        terminated = False
        if torso_z < 0.2 or torso_z > 0.5:  # Similar to Humanoid-v4 termination conditions but smaller to match nao height
        
            terminated = True

            reward = 0.0

        if self.render_mode == "human":
            self.render()
            
        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            {
                "reward_forward": reward_forward,
                "reward_ctrl": reward_ctrl,
                "reward_contact": reward_contact,
                "reward_alive": reward_alive,
                "x_position": pos_after,
                "forward_vel": forward_vel
            },
        )

    def reset_model(self):
        # Similar to Humanoid-v4 initial position
        c = 0.01  # noise scaling factor
        
        # Initialize with standing pose
        qpos = np.array([
            0.0, 0.0, 0.3386,  # 0.3386 exact z position needed
            1.0, 0.0, 0.0, 0.0,  # root orientation (quaternion)
            # Rest of the joints initialized near zero with small noise
            *[0.0 + self.np_random.uniform(low=-c, high=c) for _ in range(self.model.nq - 7)]
        ])
        
        qvel = np.array([
            0.0, 0.0, 0.0,  # root linear velocity
            0.0, 0.0, 0.0,  # root angular velocity
            # Rest of the joint velocities initialized with small noise
            *[0.0 + self.np_random.uniform(low=-c, high=c) for _ in range(self.model.nv - 6)]
        ])
        
        self.set_state(qpos, qvel)
        return self._get_obs()