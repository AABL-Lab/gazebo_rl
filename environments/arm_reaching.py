#!/usr/bin/env python3

import numpy as np
import rospy 
import time
from gen3_testing.gen3_movement_utils import Arm
from custom_arm_reaching.msg import ObsMessage
from kortex_driver.srv import *
from kortex_driver.msg import *

class ArmReacher():
    def __init__(self, max_action=1, min_action=-1, n_actions=1, action_duration=.2, reset_pose=None, episode_time=60, 
        stack_size=4, sparse_rewards=False, success_threshold=.1, home_arm=True, with_pixels=False, max_vel=.3,
        cartesian_control=True, relative_commands=True, sim=True, workspace_limits=None, observation_topic="/rl_observation",
        goal_dimensions=3):
        
        """
            Generic point reaching class for the Gen3 robot.
            Args:
                max_action (float): maximum action value
                min_action (float): minimum action value
                n_actions (int): number of actions
                action_duration (float): duration of each action
                reset_pose (list): list of floats for the reset pose
                episode_time (float): duration of each episode
                stack_size (int): size of the stack
                sparse_rewards (bool): whether to use sparse rewards
                success_threshold (float): threshold for success
                home_arm (bool): whether to home the arm at the beginning of each episode
                with_pixels (bool): whether to use pixels
                max_vel (float): maximum velocity
                cartesian_control (bool): whether to use cartesian control
                relative_commands (bool): whether to use relative commands or absolute commands
                sim (bool): whether to use the simulation
                workspace_limits (list): list of floats for the workspace limits (x_min, x_max, y_min, y_max, z_min, z_max)
                observation_topic (str): topic to subscribe to for observations
                goal_dimensions (int): number of dimensions for the goal (assumes goal position is the last n dimensions)
        """
        self.max_action = max_action
        self.min_action = min_action
        self.action_duration = action_duration
        self.n_actions = n_actions
        self.reset_pose = reset_pose
        self.episode_time = episode_time
        self.stack_size = stack_size
        self.sparse_rewards = sparse_rewards
        self.success_threshhold = success_threshold
        self.home_arm = home_arm
        self.with_pixels = with_pixels
        self.max_vel = max_vel
        self.cartesian_control = cartesian_control
        self.relative_commands = relative_commands
        self.sim = sim
        self.observation_topic = observation_topic
        self.goal_dimensions = goal_dimensions
        self.arm = Arm()

        if workspace_limits is None:
            self.workspace_limits = [-1,1,-1,1,-1,1]
        else:
            self.workspace_limits = workspace_limits

    def get_obs(self):
        return np.array(rospy.wait_for_message(self.observation_topic, ObsMessage).obs)

    def reset(self):
        time.sleep(.5)
        if self.reset_pose is None:
            self.arm.home_arm()
        else:
            if self.sim:
                self.arm.goto_joint_pose_sim(self.reset_pose)
            else:
                self.arm.goto_joint_pose(self.reset_pose)
        return self.get_obs()

    def get_reward(self, action):
        """
            Returns the reward and whether the episode is done
        """
        raise NotImplementedError
    
    def get_action(self, action):
        """
            Applies any necessary transformations to the action 
        """
        raise NotImplementedError



    def step(self, action):
        if not len(action) == self.n_actions:
            raise ValueError("Action must have length {}".format(self.n_actions))
        
        try: 
            action = self.get_action(action)
        except:
            pass

        action = np.clip(np.array(action), self.min_action, self.max_action)
        
        if self.sim:
            if self.cartesian_control:
                if self.relative_commands:
                    self.arm.goto_cartesian_pose_sim(action, speed=self.max_vel)
                else:
                    self.arm.goto_cartesian_relative_sim(action, speed=self.max_vel)
            else:
                if self.relative_commands:
                    self.arm.goto_joint_pose_sim(action, speed=self.max_vel)
                else:
                    self.arm.goto_joint_pose(action, speed=self.max_vel)

        # check if we have reached the goal
        obs = self.get_obs()
        reward, done = self.get_reward(action)
        return obs, reward, done

if __name__ == '__main__':
    try:
        rospy.init_node("arm_reacher")
        ArmReacher()
    except rospy.ROSInterruptException:
        pass

        