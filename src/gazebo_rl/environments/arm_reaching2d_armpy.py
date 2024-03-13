#!/usr/bin/env python3

import numpy as np
import rospy 
import time
from gen3_testing.gen3_movement_utils import Arm
from gazebo_rl.msg import ObsMessage
from kortex_driver.srv import *
from kortex_driver.msg import *
from gazebo_rl.environments.arm_reaching_armpy import ArmReacher
import gymnasium as gym

class ArmReacher2D(ArmReacher):
    # inherits from ArmReacher
    def __init__(self, max_action=.1, min_action=-.1, n_actions=3, action_duration=.2, reset_pose=None, episode_time=60, 
        stack_size=4, sparse_rewards=False, success_threshold=.08, wrist_rotate_limit=.3,home_arm=True, with_pixels=False, max_vel=.3,
        cartesian_control=True, relative_commands=True, sim=True, workspace_limits=None, observation_topic="rl_observation",
        goal_dimensions=3, goal_pose=None, action_movement_threshold=.01,input_size=5, discrete_actions=False):
        ArmReacher.__init__(self, max_action=max_action, min_action=min_action, n_actions=n_actions, input_size=input_size,
            action_duration=action_duration, reset_pose=reset_pose, episode_time=episode_time,
            stack_size=stack_size, sparse_rewards=sparse_rewards, success_threshold=success_threshold, home_arm=home_arm, with_pixels=with_pixels, max_vel=max_vel,
            cartesian_control=cartesian_control, relative_commands=relative_commands, sim=sim, workspace_limits=workspace_limits, observation_topic=observation_topic,
            goal_dimensions=goal_dimensions, discrete_actions=discrete_actions)
        
        
        if goal_pose is None:
            if workspace_limits is None:
                self.goal_pose = np.array([.5, .5])
            else:
                self.goal_pose = [np.random.uniform(workspace_limits[0], workspace_limits[1]),
                                  np.random.uniform(workspace_limits[2], workspace_limits[3])]
        else:
            self.goal_pose = goal_pose
        self.goal_pose = np.array(self.goal_pose)

    def get_obs(self):
        # append the goal pose to the observation
        obs = super()._get_obs() 
        # obs = np.append(obs, self.goal_pose)
        return obs  
    
    def get_reward(self, observation):
        return 0, False
        
    def get_action(self, action):
        return np.array([action[0], action[1], 0, 0, 0, 0])
    
        