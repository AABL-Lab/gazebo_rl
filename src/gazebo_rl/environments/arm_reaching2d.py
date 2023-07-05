#!/usr/bin/env python3

import numpy as np
import rospy 
import time
from gen3_testing.gen3_movement_utils import Arm
from custom_arm_reaching.msg import ObsMessage
from kortex_driver.srv import *
from kortex_driver.msg import *
from gazebo_rl.environments.arm_reaching import ArmReacher

class ArmReacher2D(ArmReacher):
    # inherits from ArmReacher
    def __init__(self, max_action=1, min_action=-1, n_actions=1, action_duration=.2, reset_pose=None, episode_time=60, 
        stack_size=4, sparse_rewards=False, success_threshold=.1, home_arm=True, with_pixels=False, max_vel=.3,
        cartesian_control=True, relative_commands=True, sim=True, workspace_limits=None, observation_topic="/rl_observation",
        goal_dimensions=3):
        super().__init__(max_action, min_action, n_actions, action_duration, reset_pose, episode_time,
            stack_size, sparse_rewards, success_threshold, home_arm, with_pixels, max_vel,
            cartesian_control, relative_commands, sim, workspace_limits, observation_topic,
            goal_dimensions)
        