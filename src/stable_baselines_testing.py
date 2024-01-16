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

# model = SAC("MlpPolicy", env, verbose=1, learning_rate=.0003)
# # model = SAC.load("src/xy_liquid_reacher_p1ST")
# # model.set_env(env)
# model.learn(total_timesteps=1250000, log_interval=10)
# model.save("src/xy_liquid_reacher_p1ST_v3")

# load model
model = SAC.load("src/xy_liquid_reacher_p1ST_v3.zip")

# # test model
obs = env.reset(visualize=True)
action = np.array([0.001,.001,0])
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    # if i % 10 == 0:
       # action = -np.array(action)
    action[2] = -.99
    obs, rewards, dones, info = env.step(action)
    print(action, rewards)
    # env.render(visualize=True)
    if rewards > 0:
        print("SUCCESS")
        env.reset(visualize=True)

    # if i % 1 == 0:
    #     obs = env.reset(visualize=True)