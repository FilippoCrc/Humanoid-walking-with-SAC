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
        forward_vel = 1.25*(pos_after - pos_before) / self.dt #1.25 forward weight
        
        # Get z position of torso
        torso_z = data.qpos[2]
        
        # Calculate aliv bonus (similar to Humanoid-v4)
        alive_bonus = 2 #this value must be a lot lower becouse the robot stays on balance much easier, in mujoco it's 5
        
        # Calculate rewards components (similar to Humanoid-v4)
        reward_forward = forward_vel
        reward_alive = alive_bonus
        reward_ctrl = -0.1 * np.square(data.ctrl).sum()
        reward_contact = -0.5e-6 * np.square(data.cfrc_ext).sum()
        reward_contact = np.clip(reward_contact, -10, 10)
        
        # Combine rewards
        reward = reward_forward + reward_alive + reward_ctrl + reward_contact
        
        
        quat = data.qpos[3:7]
        # Convert quaternion to euler angles
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (quat[0] * quat[1] + quat[2] * quat[3])
        cosr_cosp = 1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2])
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (quat[0] * quat[2] - quat[3] * quat[1])
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2
        else:
            pitch = np.arcsin(sinp)
        
        # Check both height and orientation
   # Check termination conditions
        terminated = False
        if (torso_z < 0.20 or torso_z > 0.5 or  # height check
            abs(roll) > 2.0 or    # roll > ~57 degrees
            abs(pitch) > 2.0):    # pitch > ~57 degrees
            
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
        c = 0.01  # small noise scaling factor
        
       
        
        # Standing keyframe with 33 positions (31 + 2 for the missing joints)
        standing_keyframe_qpos = [
            0.0,
            0.0,
            0.3464,
            1.0,
            0.0,
            0.0,
            0.0,
            -0.000571484,
            0.0239414,
            0.000401842,
            -3.89047e-05,
            -0.00175077,
            0.357233,
            0.0114063,
            0.000212495,
            0.000422366,
            3.92127e-05,
            -0.00133669,
            0.356939,
            0.0112884,
            -0.000206283,
            1.46985,
            0.110264,
            0.000766453,
            -0.034298,
            3.65047e-05,
            1.47067,
            -0.110094,
            -0.00201064,
            0.0342998,
            -0.00126886,
           
           
        ]
        
        
        
        # Convert to numpy array and add noise
        noisy_qpos = np.array(standing_keyframe_qpos) + self.np_random.uniform(
            low=-c,
            high=c,
            size=self.model.nq
        )
        
        # Initialize velocities as numpy array
        qvel = self.np_random.uniform(
            low=-c,
            high=c,
            size=self.model.nv
        )
        
        self.set_state(noisy_qpos, qvel)
        return self._get_obs()