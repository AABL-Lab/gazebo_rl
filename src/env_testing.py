#!/usr/bin/env python3

import numpy as np
import rospy
import time
from gen3_testing.gen3_movement_utils import Arm
from gazebo_rl.environments.arm_reaching2d import ArmReacher2D

# print all available imports from gazebo_rl

try:
    rospy.init_node('arm_reaching_node')
except:
    pass

arm = Arm()
arm.home_arm()

env = ArmReacher2D()
print(env.step([.01, .01, .01, .01, .01, .01, .01, .01]))