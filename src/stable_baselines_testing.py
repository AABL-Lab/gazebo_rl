#!/usr/bin/env python3

import numpy as np
import rospy
import time
from gen3_testing.gen3_movement_utils import Arm
from gazebo_rl.environments.arm_reaching2d import ArmReacher
from gazebo_rl.environments.liquid_reaching2d import LiquidReacher2D


from stable_baselines3 import TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env

try:
    rospy.init_node('arm_reaching_node')
except:
    pass

arm = Arm()
arm.home_arm()

# setup sac training
env = LiquidReacher2D()
# check_env(env)

model = SAC("MlpPolicy", env, verbose=1, learning_rate=.003)
model.learn(total_timesteps=200000, log_interval=10)
model.save("sac_nav2d_MediumXandYpenalty_highLR")