#!/usr/bin/env python3

import numpy as np
import rospy
import time
from gen3_testing.gen3_movement_utils import Arm
from gazebo_rl.environments.arm_reaching2d import ArmReacher2D
from gazebo_rl.environments.liquid_reaching2d import LiquidReacher2D

# print all available imports from gazebo_rl

try:
    rospy.init_node('arm_reaching_node')
except:
    pass

arm = Arm()
arm.home_arm()

# env = ArmReacher2D()
env = LiquidReacher2D()
current_time = time.time()
for i in range(10):
    #time.sleep(1)
    print(env.step([0,.1,0]))
end_time = time.time()
print("time: ", (end_time - current_time))