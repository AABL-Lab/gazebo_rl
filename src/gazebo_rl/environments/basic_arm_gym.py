#!/usr/bin/env python3
'''
This implements a wrapper for the arm that takes actions an controls the arm, but does not implement a gym or reinforcement learning environment. It's intended for use with lerobot, but generally gives a little extra use-ability on top of armpy.
'''

import copy
import numpy as np
import rospy 
import time
import armpy
from kortex_driver.srv import *
from kortex_driver.msg import *
from collections import defaultdict, deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sensor_msgs.msg import Image, JointState
from gazebo_rl.sensors.reward_check import find_and_draw_circles_and_detect_reward
from std_msgs.msg import Float32, Float32MultiArray
import matplotlib.pyplot as plt
from gazebo_rl.replay.live_replay import draw_partial_circle

zero = lambda x: (x[0] + x[1]) / 2
nrange = lambda x: x[1] - x[0]
ybounds = -0.5, 0.5; yzero = zero(ybounds); yrange = nrange(ybounds)
xbounds = 0.0, 1.0; xzero = zero(xbounds); xrange = nrange(xbounds)
zbounds = 0.0, 0.5; zzero = zero(zbounds); zrange = nrange(zbounds)
z_flower_thresh = 0.365 # for flowerpot
y_flower_thresh = -0.04 # for flowerpot
x_flower_thresh = 0.52 # for flowerpot
VELOCITY_CAP = 0.11

import logging
import threading

from cv_bridge import CvBridge
import cv2
# import Xlib.threaded
# cv2.startWindowThread()

cv_bridge = CvBridge()

image_lock = threading.Lock()
current_image = np.zeros((96, 96, 1), dtype=np.uint8)
image_time = time.time()
def img_cb(data):
    with image_lock:
        global current_image, image_time
        # decode a grayscale image
        current_image = cv_bridge.imgmsg_to_cv2(data, "mono8")
        # current_image = cv_bridge.imgmsg_to_cv2(data, "bgr8")
        # add the grayscale channel
        current_image = np.expand_dims(current_image, axis=-1)

        dt = time.time() - image_time
        if dt > 5: print(f"WARN: image time: {dt} seconds.")
        image_time = time.time()

side_image_lock = threading.Lock()
current_side_image = np.zeros((96, 96, 1), dtype=np.uint8)
side_image_time = time.time()
def side_img_cb(data):
    with side_image_lock:
        global current_side_image, side_image_time
        current_side_image = cv_bridge.imgmsg_to_cv2(data, "mono8")
        # current_side_image = cv_bridge.imgmsg_to_cv2(data, "bgr8")
        # add the grayscale channel
        current_side_image = np.expand_dims(current_side_image, axis=-1)

        current_side_image = current_side_image
        # cv2.imshow("side_image", current_side_image)
        # cv2.waitKey(1)

        dt = time.time() - side_image_time
        if dt > 5: print(f"WARN: side image time: {dt} seconds.")
        side_image_time = time.time()

current_observation = np.zeros(4)    
eef_lock = threading.Lock()
eef_time = time.time()
def eef_pose(msg):
    with eef_lock:
        global current_observation, eef_time
        gripper_pos = msg.interconnect.oneof_tool_feedback.gripper_feedback[0].motor[0].position
        tool_pose = msg.base.tool_pose_x, msg.base.tool_pose_y, msg.base.tool_pose_z #, msg.base.tool_pose_theta_x, msg.base.tool_pose_theta_y, msg.base.tool_pose_theta_z 
        tool_v = msg.base.tool_twist_linear_x, msg.base.tool_twist_linear_y, msg.base.tool_twist_linear_z, msg.base.tool_twist_angular_x, msg.base.tool_twist_angular_y, msg.base.tool_twist_angular_z
        
        dt = time.time() - eef_time
        if dt > 5: print(f"WARN: EEF time: {dt} seconds.")
        eef_time = time.time()
        # current_observation = np.array([*tool_pose, *tool_v, gripper_pos], dtype=np.float32)
        current_observation = np.array([*tool_pose, gripper_pos], dtype=np.float32)

joints = np.zeros(7)
joint_lock = threading.Lock()
def joint_state_callback(msg):
    with joint_lock:
        global joints
        joints = np.array(msg.position)

current_reward = -1.0
def reward_cb(msg):
    global current_reward
    current_reward = msg.data
    print(f'REWARD: {current_reward}')

def sync_copy_joints():
    with joint_lock:
        global joints
        return joints.copy()

def sync_copy_eef():
    with eef_lock:
        global current_observation
        return current_observation.copy()

def sync_copy_image():
    global current_image
    with image_lock:
        img_np = current_image.copy()
    return img_np

def sync_copy_side_image():
    global current_side_image
    with side_image_lock:
        img_np = current_side_image.copy()
    return img_np


def draw_partial_circle(
    image,
    center,
    radius,
    fullness,
    color,
    thickness
):
    """
    Draws a partially filled circle (pie-slice) on the given image.

    :param image: The OpenCV image (numpy array) on which to draw.
    :param center: (x, y) center of the circle.
    :param radius: Radius of the circle.
    :param fullness: A float in [0.0, 1.0] indicating how full the circle should be.
                     0.0 = not filled, 1.0 = completely filled.
    :param color: A tuple (B, G, R) color for the fill/outline.
    :param thickness: Thickness of the shape boundary. If set to -1, it draws a filled pie-slice.

    ## RSSNOTE: This function is dupliccated in gazebo_rl/environments/basic_arm_gym.py If you change this, change that too!!
    """
    # Ensure fullness is clamped between 0 and 1
    fullness_clamped = max(0.0, min(fullness, 1.0))

    # Convert fullness to degrees (0 - 360)
    end_angle_deg = int(360 * fullness_clamped)

    return cv2.ellipse(
        image,
        center=center,
        axes=(radius, radius),  # same radius in x and y → circle
        angle=0,                # no rotation
        startAngle=0,
        endAngle=end_angle_deg,
        color=color,
        thickness=thickness
    )

class BasicArm(gym.Env):
    def __init__(self, max_action=.1, min_action=-.1, n_actions=2, input_size=4, action_duration=.5, reset_pose=None, velocity_control=False,
        stack_size=4, home_arm=True, max_vel=.3, cartesian_control=True, relative_commands=True, sim=True, workspace_limits=None, discrete_actions=False, robot_name='gen3', config=None):
        
        """
            Generic point reaching class for the Gen3 robot.
            Args:
                max_action (float): maximum action value
                min_action (float): minimum action value
                n_actions (int): number of actions
                action_duration (float): duration of each action
                reset_pose (list): list of floats for the reset pose
                home_arm (bool): whether to home the arm at the beginning of each episode
                max_vel (float): maximum velocity
                cartesian_control (bool): whether to use cartesian control
                relative_commands (bool): whether to use relative commands or absolute commands
                sim (bool): whether to use the simulation
                workspace_limits (list): list of floats for the workspace limits (x_min, x_max, y_min, y_max, z_min, z_max)
        """
        super().__init__()
        assert config is not None, "config is required"

        self.max_action = max_action
        self.min_action = min_action
        self.action_duration = action_duration
        self.reset_pose = reset_pose
        self.home_arm = home_arm
        self.max_vel = max_vel
        #self.action_timout = action_timout
        self.cartesian_control = cartesian_control
        self.relative_commands = relative_commands
        self.sim = sim
        self.velocity_control = velocity_control

        self.logger = logging.getLogger(self.__class__.__name__)

        print("Initializing arm...")
        self.arm = armpy.initialize(robot_name.replace('my_', ''))
        print("Initialized arm")
        
        if workspace_limits is None:
            self.workspace_limits = [*xbounds, *ybounds, *zbounds]
        else:
            self.workspace_limits = workspace_limits

        rospy.Subscriber(f"/{robot_name}/base_feedback", BaseCyclic_Feedback, eef_pose)
        rospy.Subscriber(f"/{robot_name}/base_feedback/joint_state", JointState, joint_state_callback)
        rospy.Subscriber(f"/camera_obs__dev_video4_96x96", Image, img_cb)
        rospy.Subscriber(f"/camera_obs__dev_video0_96x96", Image, side_img_cb)
        rospy.Subscriber('/reward', Float32, reward_cb)
        self.action_pub = rospy.Publisher('action', Float32MultiArray, queue_size=10)
        self.SAFETY_MODE = False
        self.safety_histories = {
            "x_tool_torque": deque(maxlen=10),
            "joint_1_torque": deque(maxlen=10), # the first bend, pressing down relieves the torque here.
        }
        self.x_tool_thresh, self.joint_1_thresh = 2.0, -0.01

        self.gripper_state = None
        self.current_step = 0
        self.prev_eef = None; self._eef = None
        self._eef_lock = threading.Lock()
        self._eef_time = time.time()
        self.LITE = 'lite' in robot_name
        if self.LITE: print(f"Using gen3_lite")
        else: print(f"Using gen3")


        self.n_img_ch = 1 if config.grayscale else 3
        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            ),
            "joints": spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            ),
            "image_top": spaces.Box(
                low=0, high=255, shape=(*config.size, self.n_img_ch), dtype=np.uint8
            ),
            "image_bottom": spaces.Box(
                low=0, high=255, shape=(*config.size, self.n_img_ch), dtype=np.uint8
            ),
            'reward': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'is_first': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
            'is_last': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
            'is_terminal': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
        })

        # Define action space
        # action -> shape=(7,), dtype=float32
        # Typically you'll bound velocity inputs in [-1,1] or something similar
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        self.last_step_time = time.time()
        
        self.crop_dim = 700
        self.crop_left_offset = 200
        self.config = config
        self.reward_check_global = 0; self.reward_check_period = 10
        self.start_time = time.time()
        fig, ax = plt.subplots(1, 2)
        self.fig = fig; self.ax = ax
        self.plot_circle_height = True

        self.add_circle_for_height = True

    def _base_feedback_callback(self, msg: BaseCyclic_Feedback):
        '''
        NOTE: This is not a generic safety check, and WILL NOT PREVENT MOST COLLISIONS.
        We're looking at specific safety check to prevent moving down in the z-direction when we are experiencing increased torques from vertical collision.
        '''
        self.safety_histories['x_tool_torque'].append(msg.base.tool_external_wrench_torque_x)
        self.safety_histories['joint_1_torque'].append(msg.actuators[1].torque)
        # if len(self.safety_histories['x_tool_torque']) == 10:
        #     xtool_mean = np.mean(self.safety_histories['x_tool_torque']); joint1_mean = np.mean(self.safety_histories['joint_1_torque'])
        #     if (xtool_mean > self.x_tool_thresh) or (joint1_mean <= self.joint_1_thresh):
        #         self.SAFETY_MODE = True
        #         print(f"SAFETY MODE ENGAGED: {np.mean(self.safety_histories['x_tool_torque'])} {np.mean(self.safety_histories['joint_1_torque'])}")
        #     else:
        #         self.SAFETY_MODE = False
        try:
            gripper_pos = msg.interconnect.oneof_tool_feedback.gripper_feedback[0].motor[0].position
        except:
            print("ERROR: no gripper feedback received")
            gripper_pos = 0.0

        with self._eef_lock:
            self._eef = np.array([msg.base.tool_pose_x, msg.base.tool_pose_y, msg.base.tool_pose_z, gripper_pos])
            dt = time.time() - self._eef_time
            if dt > 5: print(f"WARN: EEF time: {dt} seconds.")
            self._eef_time = time.time()
    
    def stop_motion(self):
        print("Stopping motion")
        self.arm.stop_arm()
        rospy.sleep(0.25)
        self.arm.cartesian_velocity_command([0 for _ in range(7)], duration=self.action_duration, radians=True)
        rospy.sleep(0.25)
        print("\tHopefully stopped motion")


    def _get_obs(self, is_first=False):
        try:
            top_img = sync_copy_image()
            bot_img = sync_copy_side_image()

            # display the images without cv2
            # self.ax[0].imshow(top_img)
            # self.ax[1].imshow(bot_img)
            # plt.show()

            joints = sync_copy_joints()
            state = sync_copy_eef()
            self.prev_eef = state
        except Exception as e:
            print("No image received. Sending out blank observation.", e)
            # return self._get_obs(is_first=is_first) #oof ugly
            return {
                "state": self.observation_space['state'].sample(),
                "joints": self.observation_space['joints'].sample(),
                "image_top": np.zeros((*self.config.size, self.n_img_ch), dtype=np.uint8),
                "image_bottom": np.zeros((*self.config.size, self.n_img_ch), dtype=np.uint8),
                'reward': -1.0,
                'is_first': is_first,
                'is_last': 0,
                'is_terminal': 0,
                }
        

        reward = current_reward
        if reward > 0: is_last = True; is_terminal = True
        else: is_last = False; is_terminal = False

        if self.plot_circle_height:
            z_value = state[2]
            # draw a circle on the GRAYSCALE image
            circle_r = int(np.ceil(top_img.shape[0] * 0.1))
            circle_midpoint = (circle_r, top_img.shape[0] - circle_r)
            top_img = draw_partial_circle(top_img, circle_midpoint, circle_r, z_value / 0.6, (255, 0, 0), -1)
            # write out the top img as a check
            cv2.imwrite("/home/j/workspace/top_img.png", top_img)

        return {
            "state": state,
            "joints": joints,
            "image_top": top_img,
            "image_bottom": bot_img,
            'reward': reward,
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
            }
    

    def reset(self):
        self.current_step = 0
        reset_complete = False
        while not reset_complete:
            try:
                if not self.sim:
                    self.arm.stop_arm()
                print(f"RESET {'- sim' if self.sim else ''}")
                while rospy.get_param("/pause", False):
                    print("Waiting for pause to be lifted before resetting.")
                    rospy.sleep(1)
                self.arm.clear_faults()
                rospy.sleep(.25)
                self.arm.open_gripper()
                rospy.sleep(0.5)

                if self.reset_pose is None:
                    self.arm.home_arm()
                else:
                    if self.sim:
                        self.arm.goto_joint_pose_sim(self.reset_pose)
                    else:
                        backup_position = [0.34551798719466237, -0.8454950565561763, 2.169129261535217, -1.232747441193471, 1.4586096006108726, -1.686383909690952] #, 0.5953540153613426]
                        target_joint_positions = [0.3268500269015339, -1.4471734542578538, 2.3453266624159497, -1.3502152158191212, 2.209384006676201, -1.5125125137062945] #, -0.0877648122691288]
                        
                        if np.allclose(target_joint_positions, joints[:6], atol=0.1):
                            print("Already at reset target")
                        else:
                            # which position is the arm closest to?
                            if np.linalg.norm(np.array(joints[:6]) - np.array(backup_position)) < np.linalg.norm(np.array(joints[:6]) - np.array(target_joint_positions)):
                                for tjp in [backup_position, target_joint_positions]:
                                    while not np.allclose(joints[:6], tjp, atol=0.1):
                                        print(f"\tMoving to {tjp}. Distance from target {np.linalg.norm(np.array(joints[:6]) - np.array(tjp))}")
                                        self.arm.goto_joint_pose(tjp, radians=True, block=False)
                                        rospy.sleep(4.0)
                            else:
                                print(f"Moving to target position since its closer {target_joint_positions}")
                                self.arm.goto_joint_pose(target_joint_positions, radians=True, block=False)
                                rospy.sleep(4.0)
                reset_complete = True
            except rospy.ServiceException as e:
                print("Service exception in stopping arm", e)
                rospy.sleep(3)
            except Exception as e:
                print("Unexpected exception in stopping arm", e)
                    

        while current_reward > -1.:
            print(f"REWARD IS NONZERO, RESET PUBLISHER!")
            rospy.sleep(2.0)


        obs = self._get_obs(is_first=True)
        self.start_time = time.time()
        return obs

    def ready(self):
        return True if self.prev_eef else False

    def step(self, action, orientation_speed=None, translation_speed=None, clip_wrist_action=False):
        step_start_time = time.time()
        # fastest way to track the time that has passed
        action = action['action']

        self.current_step += 1

        # NOTE: temporary mapping to align with config
        # Is this the same as for DfD as it was for lerobot?
        # action = [action[0], action[1], action[2], 0, action[5], 0, action[6]]
        
        # scale the first three action dimensions between minimum action and the max action. [-1,1] * max_action = [-max_action, max_action]
        action = [action[0] * 0.1222, action[1] * 0.1222, action[2] * 0.1222, 0., action[3], 0., action[4]]

        if self.velocity_control:
            # clip all but the last action idx
            action = [np.clip(a, -VELOCITY_CAP, VELOCITY_CAP) for a in action[:3]] + action[3:]
            buffered_move_xyz = [1.0 * (a * self.action_duration) for a in action[:3]]
            prev_xyz = self.prev_eef[:3]
            expected_new_position = newx, newy, newz = [prev_p + dp for prev_p, dp in zip(prev_xyz, buffered_move_xyz)]
        else: expected_new_position = newx, newy, newz = self.prev_eef[:3] + action[:3] # Do not allow an action to take us beyond the workspace limits

        prev_state_str = f"{self.prev_eef[0]:+1.2f} {self.prev_eef[1]:+1.2f} {self.prev_eef[2]:+1.2f}"
        pred_state_str = f"{newx:+1.2f} {newy:+1.2f} {newz:+1.2f}"
        # print(f"{self.current_step:4d} dp: {prev_state_str} -> {pred_state_str} from action {action[:3]}")

        ### DON"T FLIP THE SHELF
        # if (newz >= 0.1 and action[2] > 0) and newx >= 0.53:
        #     print("z > 0.1 and x > 0.53. stopping.")
        #     action[2] = 0
        # elif (newz <= 0.1) and (newx >= 0.51 and action[0] > 0):
        #     print("z > 0.1 and x > 0.53. stopping.")
        #     action[0] = 0
        # elif (newz <= 0.12 and action[2] < 0) and newx >= 0.53:
        #     print("z < 0.12 and x > 0.53. stopping.")
        #     action[2] = 0
        ####

        if (newz <= 0.015 and action[2] < 0) or (newz >= 0.6 and action[2] > 0):
            action[2] = 0; # print("z out of bounds. stopping.")
        if (newx <= 0.3 and action[0] < 0) or (newx >= 0.8 and action[0] > 0):
            action[0] = 0; # print("x out of bounds. stopping.")
        if (newy <= -0.25 and action[1] < 0) or (newy >= 0.25 and action[1] > 0):
            action[1] = 0; # print("y out of bounds. stopping.")

        # for newd, d in zip([newx, newy, newz], action[:3]):
        #     print(f"{newd:+1.2f} {d:+1.2f} || ", end=' ')
        # print(f'{ 1 / (time.time() - self.last_step_time):1.2f} HZ')
        self.last_step_time = time.time()

        FAULT = False
        if self.sim:
            if self.cartesian_control:
                if not self.relative_commands:
                    self.arm.goto_cartesian_pose_sim(action, speed=self.max_vel)
                    rospy.sleep(self.action_duration)
                else:
                    self.arm.goto_cartesian_relative_sim(action, speed=self.max_vel)
                    rospy.sleep(self.action_duration)
                    self.arm.stop_arm()
            else:
                if self.relative_commands:
                    self.arm.goto_joint_pose_sim(action, speed=self.max_vel)
                    rospy.sleep(self.action_duration)
                else:
                    self.arm.goto_joint_pose(action, speed=self.max_vel)
                    rospy.sleep(self.action_duration)
                    self.arm.stop_arm()
        else:
            if self.cartesian_control:
                if not self.relative_commands: # NOTE: wtf?
                    self.arm.goto_cartesian_pose_sim(action, speed=self.max_vel)
                    rospy.sleep(self.action_duration)
                else:
                    if self.velocity_control:
                        # if abs(action[6]) > 0.9:
                        #     self.arm.cartesian_velocity_command([0. for _ in range(7)], duration=self.action_duration, radians=True, block=False)
                        #     if action[6] > 0:
                        #         print(f"    CLOSE GRIPPER")
                        #         self.arm.close_gripper(block=False)
                        #         rospy.sleep(0.5)
                        #     else:
                        #         print(f"    OPEN GRIPPER")
                        #         self.arm.open_gripper(block=False)
                        #         rospy.sleep(0.5)
                        # else:
                        #     self.arm.cartesian_velocity_command(action[:6], duration=self.action_duration, radians=True, block=False)

                        try:
                            gripper = False
                            if abs(action[6]) > 0.8:
                                if action[6] > 0:
                                    # print(f"    CLOSE GRIPPER")
                                    self.arm.send_gripper_command(-1., mode = 'speed', duration = 200, relative=True, block=False)
                                else:
                                    # print(f"    OPEN GRIPPER")
                                    self.arm.send_gripper_command(0.1, mode = 'speed', duration = 200, relative=True, block=False)

                            # print(', '.join([f"{a:+1.2f}" for a in action]))
                            self.arm.cartesian_velocity_command(action[:6], duration=self.action_duration, radians=True, block=False)
                            # if not gripper:
                        except Exception as e:
                            print("Error in velocity command", e)
                            print(f"Returning done for a reset")
                            FAULT = True
                            
                    else:
                        # print("goto_cartesian_pose_old")
                        self.arm.goto_cartesian_pose_old(action, relative=True, radians=True, 
                                                        translation_speed=translation_speed, orientation_speed=orientation_speed)
                        rospy.sleep(self.action_duration)
                        # self.arm.stop_arm()
            else:
                if self.relative_commands:
                    self.arm.goto_joint_pose_sim(action, speed=self.max_vel)
                    rospy.sleep(self.action_duration)
                else:
                    self.arm.goto_joint_pose(action, speed=self.max_vel)
                    rospy.sleep(self.action_duration)
                    self.arm.stop_arm()
            

        # for k,v in obs.items():
        #     print(f"{k}: {v.shape if hasattr(v, 'shape') else v}")
        # print('--'*20)
        

        obs = self._get_obs(is_first=False)
        # rospy.sleep(1 / 30.) # 30 Hz

        # sleep for the rest of the time
        # rospy.sleep((self.action_duration) - (time.time() - step_start_time)) # for random agent
        # rospy.sleep((self.action_duration - 0.01) - (time.time() - step_start_time)) # for real model training


        if FAULT:
            obs['is_last'] = True
            obs['is_terminal'] = True
            obs['reward'] = -1.0 * (self.config.time_limit - self.current_step) # put a negative reward for the time limit to prevent the agent from trying to fail early
            print(f"FAULT: returning done and {obs['reward']} reward")

        return obs, obs['reward'], obs['is_last'], False, {}

        # self.prev_eef = sync_copy_eef()
        # return True
    
    def close(self):
        self.arm.stop_arm()
        self.arm.home_arm()
        rospy.sleep(.5)

if __name__ == '__main__':
    try:
        rospy.init_node("arm_reacher")
        rospy.sleep(1.0)
        robot_name = rospy.get_param('robot_name', ',my_gen3')
        sim = rospy.get_param('sim', False)
        arm = BasicArm(robot_name=robot_name, velocity_control=True, sim=sim)
        arm.reset()
        action = [0 for _ in range(7)]
        
        vx = 0.1
        for i in range(20):
            if i % 4 == 0: vx *= -1

            action[0] = vx

            arm.step(action)
            
            rospy.sleep(0.1)
            print(i)
        arm.close()
    except rospy.ROSInterruptException as E:
        print(E)


        