#!/usr/bin/env python3

from typing import Any, SupportsFloat
import numpy as np
import rospy 
import time
from gen3_testing.gen3_movement_utils import Arm
from custom_arm_reaching.msg import ObsMessage
from kortex_driver.srv import *
from kortex_driver.msg import *
from gazebo_rl.environments.arm_reaching import ArmReacher
import gymnasium as gym

class LiquidReacher2D(ArmReacher):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    # inherits from ArmReacher
    def __init__(self, max_action=1, min_action=-1, n_actions=3, action_duration=.2, reset_pose=None, episode_time=60, 
        stack_size=4, sparse_rewards=False, success_threshold=.01, wrist_rotate_limit=.3,home_arm=True, with_pixels=False, max_vel=.3,
        cartesian_control=True, relative_commands=True, sim=True, workspace_limits=None, observation_topic="/rl_observation",
        goal_dimensions=3, goal_pose=None, action_movement_threshold=.01):
        ArmReacher.__init__(self, max_action=max_action, min_action=min_action, n_actions=n_actions, input_size=6,
            action_duration=action_duration, reset_pose=reset_pose, episode_time=episode_time,
            stack_size=stack_size, sparse_rewards=sparse_rewards, success_threshold=success_threshold, home_arm=home_arm, with_pixels=with_pixels, max_vel=max_vel,
            cartesian_control=cartesian_control, relative_commands=relative_commands, sim=sim, workspace_limits=workspace_limits, observation_topic=observation_topic,
            goal_dimensions=goal_dimensions)
        
        
        if goal_pose is None:
            if workspace_limits is None:
                self.goal_pose = np.array([.5, .5])
            else:
                self.goal_pose = [np.random.uniform(workspace_limits[0], workspace_limits[1]),
                                  np.random.uniform(workspace_limits[2], workspace_limits[3])]
        else:
            self.goal_pose = goal_pose
        self.goal_pose = np.array(self.goal_pose)
        self.wrist_rotate_limit = wrist_rotate_limit
        # penalty for moving while the wrist is out of bounds
        self.action_movement_threshold = action_movement_threshold


    def step(self, action):
        return super().step(self._get_action(action))
    
    def reset(self):
        self.goal_pose = [np.random.uniform(self.workspace_limits[0], self.workspace_limits[1]),
                                    np.random.uniform(self.workspace_limits[2], self.workspace_limits[3])]
        return super().reset()
    
    def _get_obs(self):
        # append the goal pose to the observation
        obs = super()._get_obs() 
        obs = np.append(obs, self.goal_pose)
        return obs 
    
    def _get_reward(self, observation, action):
        current_pose = observation[:2]
        wrist_pose = observation[3]
        goal_pose = observation[-2:]
        dist = np.linalg.norm(current_pose - goal_pose)
        print("dist: ", dist)
        print("goal_pose: ", goal_pose)
        print("current_pose: ", current_pose)
        rew = 0
        if np.abs(wrist_pose) > self.wrist_rotate_limit:
            if action[0] > self.action_movement_threshold or action[1] > self.action_movement_threshold:
                rew = -100
                return rew, False
            else:
                rew = -20
                return rew, False
        else:
            if dist < self.success_threshold:
                rew = 10
                return rew, True
            if self.sparse_rewards:
                return 0, False
            else:
                rew = -dist
                return -dist, False
        
    def _get_action(self, action):
        print("action TEST: ", action)
        if self.sim:
            return np.array([action[0], action[1], 0, 0, action[2], 0])
        else:
            return np.array([action[0], action[1], 0, 0, -action[2]])
    
        