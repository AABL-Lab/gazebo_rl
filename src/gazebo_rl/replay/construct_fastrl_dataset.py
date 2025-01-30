#!/home/j/workspace/lerobot/venv/bin/python
import rospy
import rosbag
import os, sys, resource
from pathlib import Path
import time
import cv2
import numpy as np
import cv_bridge
from sensor_msgs.msg import Image, Joy
from collections import deque
from armpy import kortex_arm
import std_msgs.msg
import armpy
from gazebo_rl.sensors.reward_check import find_and_draw_circles_and_detect_reward
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as RosImage, Joy, JointState
from std_msgs.msg import Float32, Int8
import IPython
from kortex_driver.msg import BaseCyclic_Feedback

GOAL_X = 1279; GOAL_Y = 719; GOAL_MIN = 400; GOAL_MAX = 700; GOAL_Y_BOUNDARY = 200

def basecyclicfeedback_to_state(msg: BaseCyclic_Feedback):
    gripper_pos = msg.interconnect.oneof_tool_feedback.gripper_feedback[0].motor[0].position
    tool_pose = msg.base.tool_pose_x, msg.base.tool_pose_y, msg.base.tool_pose_z, msg.base.tool_pose_theta_x, msg.base.tool_pose_theta_y, msg.base.tool_pose_theta_z 
    tool_v = msg.base.tool_twist_linear_x, msg.base.tool_twist_linear_y, msg.base.tool_twist_linear_z, msg.base.tool_twist_angular_x, msg.base.tool_twist_angular_y, msg.base.tool_twist_angular_z
    return np.array([*tool_pose, *tool_v, gripper_pos], dtype=np.float32)

cvbridge = CvBridge()

def get_memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.name == "posix":
        # On Linux and macOS, ru_maxrss is in kilobytes
        usage = usage / 1024  # Convert to MB
    return usage

class VideoLoader:
    def __init__(self, video_dirs, threshold_ns=0., cache_size=100):
        """
        Initialize the VideoLoader with paths to videos and their corresponding timestamps.
        """
        self.dirs = [str(entry).split('/')[-1] for entry in video_dirs]
        self.video_paths = [Path(dirname).absolute() / 'output.mp4' for dirname in video_dirs]
        timestamp_paths = [Path(dirname).absolute() / 'video_frame_timestamps.txt' for dirname in video_dirs]
        self.timestamp_lists = []
       
        for timestamp_fn in timestamp_paths:
            timestamps = []
            with open(str(timestamp_fn), 'r') as fp:
                timestamps = fp.readlines()
            timestamps = [rospy.Time.from_seconds(int(t) / 1e9) for t in timestamps]
            self.timestamp_lists.append(timestamps)

        self.frames = [deque(maxlen=100) for _ in self.video_paths]  # To store frames of each video
        self.captures = []
        self.frame_caches = [deque(maxlen=cache_size) for _ in self.video_paths]
        # self.frame_timestamps = []  # To store timestamp lists of each video
        self.frame_idx = [0 for _ in self.video_paths] # NOTE: this class dumps its frames, it doesn't search them, when it runs into problems it yells and skips frames. Its like a bag.
        # self._load_videos()
        self._open_videos()



        self.threshold_ns = rospy.Time(secs=0, nsecs=threshold_ns)

    def _open_videos(self):
        """
        Open video files for streaming.
        """
        self.captures = [cv2.VideoCapture(str(path)) for path in self.video_paths]
        for i, cap in enumerate(self.captures):
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {self.video_paths[i]}")
            else:
                # load the first frame
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Cannot read from capture {self.video_paths[i]}")
                self.frames[i].append(frame)

    # NOTE: Better to drop frames than spend too much time getting them (frames will be dropped in the real world)
    def get_frame_if_available(self, target_timestamp):
        """
        Retrieve a frame if the next frame is close enough to the passed target_timestamp. NOTE: if you didn't have a fast signal in the rosbag you would miss 
        
        :param target_timestamp: Timestamp for which to retrieve the frame.
        :return: Frames at the given target_timestamp or None if no close match.
        """
        t0 = time.perf_counter()
        ret_frames = [None for _ in self.frames]
        read = 0
        percent_complete = {}
        for cam_idx, capture in enumerate(self.captures):
            idx = self.frame_idx[cam_idx]
            if len(self.frames[cam_idx]) == 0:
                ret, frame = capture.read(); read += 1
                self.frames[cam_idx].append(frame)
 
            if idx > len(self.timestamp_lists[cam_idx]):
                rospy.loginfo("Out of video frames for camera {self.video_paths[cam_idx]}")
                continue

            ts = self.timestamp_lists[cam_idx][idx]
            if (ts - self.threshold_ns).to_nsec() <= target_timestamp.to_nsec(): # We're late, on time, or ahead <= threshold_ns
                ret_frames[cam_idx] = self.frames[cam_idx].pop()
                self.frame_idx[cam_idx] = self.frame_idx[cam_idx] + 1

            remaining_frame_count = len(self.timestamp_lists[cam_idx]) - idx
            percent_complete[cam_idx] = 1 - (remaining_frame_count / len(self.timestamp_lists[cam_idx]))

        endT = time.perf_counter()
        if (endT - t0) > 0.01:
            rospy.logwarn(f"Video read took longer than 0.01 seconds: {(endT - t0)=}")

        return ret_frames, ts, percent_complete
    
import csv
class BagVideoPublisher():
    def __init__(self, path, args):
        stop_arm = args.stop_arm
        SHOW_FIRST_FRAME = False

        # award annotations file (for reward signal)
        award_annotations = Path('~/annotations.csv').expanduser()
        assert award_annotations.exists(), f"Annotations file not found: {award_annotations}"

        UID = str(path).split('user_')[1]
        print(f"UID: {UID}")
        
        self.no_annotation = False
        award_annotations = csv.DictReader(open(award_annotations, 'r'))
        success_time = None
        for row in award_annotations:
            if row['User ID'] == UID:
                print(f"Found user {UID} {row['Success Time ']}")
                success_time = float(row['Success Time '])
        if not success_time:
            self.no_annotation = True
            # raise ValueError(f"User {UID} not found in annotations file")

        # dirname is also the name of the camera topic live
        rospy.init_node('bagvideopublisher', anonymous=True)

        if not args.no_arm:
            arm = armpy.initialize('gen3_lite')

        bridge = cv_bridge.CvBridge()
        bag = rosbag.Bag(str(path / 'trial_data.bag'))

        print(f"Loading videos...", end='')
        video_dirs = [entry for entry in path.iterdir() if "cam_dev_video" in str(entry)]
        print(f"{video_dirs}", end=' -- ')
        video_loader = VideoLoader(video_dirs) # TODO: Need to buffer and stream as each video is about 8GB of RAM when preloaded and held as frames
        print(f"Loaded.")

        # Set-up ros publishers
        ros_publisher = {}; log_msg_types = {}
        for topic, msg, t in bag.read_messages():
            if topic in ros_publisher: continue
            else:
                ros_publisher[topic] = rospy.Publisher(topic, type(msg), queue_size=1); log_msg_types[topic] = type(msg)
                print("topic:", topic)
        for topic in video_loader.dirs:
            ros_publisher[topic] = rospy.Publisher(topic, Image, queue_size=1); log_msg_types[topic] = Image
            print(f"Video topic: {topic}")
        for k,v in ros_publisher.items():
            print(k, log_msg_types[k])

        # Reward publisher
        reward_pub = rospy.Publisher('/reward', std_msgs.msg.Float32, queue_size=1)
        rospy.sleep(0.1)
        ##

        rospy.loginfo(f"Using {get_memory_usage()} MB")

        import time
        
        t0 = None; walltime = rospy.Time.now(); pnum = 0

        if args.no_arm or args.dont_publish:
            AT_FIRST_POSE = True # don't need to move the arm
        else:
            for topic, msg, t in bag.read_messages('/my_gen3_lite/base_feedback/joint_states'):
                arm.goto_joint_pose(msg.position, radians=True, block=False)
                break
            time.sleep(5)


        crop_dim = rospy.get_param('crop_dim', 0); crop_left_offset = rospy.get_param('crop_left_offset', 0)
        top_index = rospy.get_param('top_index', '4'); bottom_index = rospy.get_param('bottom_index', '0')
        # image_w = None; image_h = None

        episode = {
            'is_first': [],
            'is_last': [],
            'reward': [],
            'action': [],
            'state': [],
            'joint_states': [],
            'is_terminal': [],
            'image_top': [],
            'image_bottom': [],
            'discount': [],
            'logprob': [], # unused
            'frame_timestamp': []
        }

        def add_frame_to_episode(fm):
            assert all(len(v) > 0 for v in fm.values()), f"Frame missing values: {[k for k,v in fm.items() if len(v) == 0]}"
    
            for k in ['reward', 'is_first', 'is_last', 'is_terminal']:
                episode[k].append(fm[k][0])

            # take the first frame we've got (rather than the last). Images should lag behind the state/action by a frame.
            episode['image_top'].append(fm['image_top'][0])
            episode['image_bottom'].append(fm['image_bottom'][0])
            episode['state'].append(fm['state'][0])
            episode['joint_states'].append(fm['joint_states'][0])

            episode['discount'].append(0 if fm['is_last'] or fm['is_terminal'] else 1)
            episode['logprob'].append(0) # unused

            # episode['frame_timestamp'].append(fm['frame_timestamp'][0])

            # take the mean of the action, since it's a continuous action space
            dof6 = np.mean(fm['action'], axis=0) if len(fm['action']) > 1 else fm['action'][0]
            if 1.0 in fm['gripper'] and -1.0 in fm['gripper']: print(f"WARN: both gripper states in frame: {fm['gripper']}")
            if 1.0 in fm['gripper']: gripper = 1.0
            elif -1.0 in fm['gripper']: gripper = -1.0
            else: gripper = 0.
            # print(f"Adding frame to episode: {dof6} {gripper}")
            episode['action'].append(list(dof6) + [gripper])

        def init_frame():
            fm = {k: [] for k in episode.keys() if k != 'logprob'}
            fm['gripper'] = []
            return fm

        frame = init_frame()
        first_frame = True

        episode_hz = 10 # frames / sec
        loop_t0 = frame_t0 = cam_t0 = -1; frame_num = 0; dropped_frames = 0
        last_dof6 = [[0. for _ in range(6)]]; last_twist_time = None
        BREAK_DUE_TO_REWARD = False
        for topic, msg, t in bag.read_messages():
            if loop_t0 == -1: loop_t0 = t.to_sec()
            if frame_t0 == -1: frame_t0 = t.to_sec()

            if t.to_sec() - frame_t0 >  1 / episode_hz:
                # print(f"Cutting frame {frame_t0 - loop_t0:1.2f}")
                frame['is_first'] = [first_frame]; first_frame = False
                frame['gripper'] = [0.] if len(frame['gripper']) == 0 else frame['gripper']
                frame['action'] = last_dof6 if len(frame['action']) == 0 else frame['action']
                frame['frame_timestamp'] = [frame_t0]

                if len(frame['reward']) > 0: # a reward signal was received
                    assert frame['reward'][0] == 1.0, f"Reward signal not 1.0: {frame['reward'][0]}"
                    BREAK_DUE_TO_REWARD = True
                    frame['discount'], frame['is_last'], frame['is_terminal'] = [0.], [True], [True]
                else:
                    frame['discount'], frame['is_last'], frame['is_terminal'], frame['reward'] = [1.], [False], [False], [-1.]
                # add the frame to the episode
                try:
                    add_frame_to_episode(frame)
                except Exception as e:
                    dropped_frames += 1
                    print(f"Error adding frame to episode: {e}")

                frame = init_frame()
                frame_t0 = t.to_sec()
                frame_num += 1

            if BREAK_DUE_TO_REWARD: break

            # accrue all the messages that make up the frame
            str_type = str(type(msg))
            # elif type(msg) == BaseCyclic_Feedback:
            if '__BaseCyclic_Feedback' in str_type:
                state = basecyclicfeedback_to_state(msg)
                frame['state'].append(state)
            # elif type(msg) == Joy: 
            elif '__Joy' in str_type:
                # NOTE: unfortunately this depends on the input device.

                # FOR XBOX CONTROLLER
                if len(msg.buttons) >= 6:
                    gripper_vel = -msg.buttons[4] if msg.buttons[4] else msg.buttons[5]
                    x, y = msg.axes[0], msg.axes[1] # NOTE: this is wrong actually, and should be switched. for RSS we're manually switching the dimensions in the robot control loop
                    z = msg.axes[4]
                    r, p, yaw = 0., 0., msg.axes[3]
                    np.array([x, y, z, r, p, yaw, gripper_vel], dtype=np.float32)
                else:
                    # FOR MOUSE & KEYBOARD
                    gripper_state = 1.0 if msg.buttons[0] else 0 #TODO: make continuous and align with gripper direction
                    gripper_state = -1.0 if msg.buttons[1] else 0
                    np.array([*msg.axes, gripper_state], dtype=np.float32)
                frame['gripper'].append(gripper_vel)
            elif '__TwistCommand' in str_type:
                dof6 = [msg.twist.linear_x, msg.twist.linear_y, msg.twist.linear_z, msg.twist.angular_x, msg.twist.angular_y, msg.twist.angular_z]
                frame['action'].append(dof6)
                last_dof6 = [dof6]
            elif '__JointState' in str_type and 'base_feedback' in topic:
                frame['joint_states'].append(list(msg.position))
            else:
                pass
            # elif type(msg) in [Float32, Int8]:
            #     return np.float32(msg.data)
            # elif type(msg) == std_msgs.msg.Bool:
            #     return np.float32(msg.data)

            reward = 0.
            camera_frames, ts, percent_complete = video_loader.get_frame_if_available(t)
            if cam_t0 == -1: cam_t0 = ts.to_sec()
            for cidx, ctopic in enumerate(video_loader.dirs):
                cframe = camera_frames[cidx]
                if cframe is not None:
                    if crop_dim > 0:
                        CAM_POSITION = 'top' if top_index in ctopic else 'bottom'

                        # # crop from the right edge for TOP image
                        if CAM_POSITION == 'top':
                            cframe = cframe[:crop_dim, crop_left_offset:crop_dim+crop_left_offset]
                        else:
                            cframe = cframe[:crop_dim, -crop_dim:]
                            # show circles
                        #     # reward = find_and_draw_circles_and_detect_reward(cframe)

                        #     if not (self.no_annotation) and (ts.to_sec() - cam_t0 > success_time):
                        #         reward = 1.0
                        #     else:
                        #         reward = 0.0

                            # crop from the left, no offset for BOTTOM image

                    # convert to grayscale and 96 x 96

                    if False and (args.show_video or SHOW_FIRST_FRAME or percent_complete[cidx] > args.percent_threshold):
                        cv2.imshow(f'{UID} {CAM_POSITION=} {cframe.shape=} {crop_dim} {crop_left_offset}', cframe)
                        if percent_complete[cidx] > args.percent_threshold and len(frame['reward']) == 0:
                            print(f"Percent complete: {percent_complete[cidx]}")
                            k = cv2.waitKey(0)
                            if k == ord('r'):
                                frame['reward'].append(1.0)
                                print(f"Reward signal added at {ts.to_sec() - cam_t0}")
                            elif k == ord('q'):
                                cv2.destroyAllWindows()
                                exit(0)
                        else:
                            cv2.waitKey(1)

                    cframe = cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)
                    cframe = cv2.resize(cframe, (96, 96))
                    frame[f'image_{CAM_POSITION}'].append(cframe)
            # if frame_num > 100: break

        print(f"Published {frame_num} frames.")

        # change the last_frame is_last, is_terminal, and discount
        episode['is_last'][-1] = True
        episode['is_terminal'][-1] = True
        episode['discount'][-1] = 0


        cv2.destroyAllWindows()

        def playback_npz(episode):
            # now go through the episode and replace it with arrows on the images corresponding to the actions
            for i in range(len(episode['action'])):
                action = episode['action'][i]
                if episode['image_bottom'][i] is not None:
                    img = episode['image_bottom'][i]

                    # make it bigger
                    img = cv2.resize(img, (0,0), fx=4, fy=4)

                    midpoint = (img.shape[1] // 2, img.shape[0] // 2)
                    cv2.arrowedLine(img, (midpoint[0], midpoint[1]), (midpoint[0] , midpoint[1] - int(action[2] * midpoint[1])), (0, 255, 0), 2)
                    cv2.imshow(f'bot', img)
                if episode['image_top'][i] is not None:
                    img = episode['image_top'][i]

                    img = cv2.resize(img, (0,0), fx=4, fy=4)

                    midpoint = (img.shape[1] // 2, img.shape[0] // 2)
                    cv2.arrowedLine(img, (midpoint[0], midpoint[1]), (midpoint[0] + int(action[0] * midpoint[0]), midpoint[1] - int(action[1] * midpoint[1])), (0, 255, 0), 2)
                    # put text about the gripper state
                    cv2.putText(img, f"{action[6]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # put the reward
                    cv2.putText(img, f"{episode['reward'][i]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(f'top', img)
                cv2.waitKey(1 if i < len(episode['action']) - 1 else 0)

        # out directory
        outdir = Path(f'~/workspace/HD_ros_53/eps/').expanduser()
        outdir.mkdir(exist_ok=True, parents=True)
            
            # .dump(outdir / f'{k}.npy')

        # for k,v in episode.items():
        #     print(f"{k=}, {v.shape if isinstance(v, np.ndarray) else len(v)}", end=' ')
        #     val = v[0]
        #     if isinstance(val, np.ndarray):
        #         print(f'{val.shape}')
        #     elif hasattr(val, '__len__'):
        #         print(f'{len(val)}')
        #     else:
        #         print(val)
        # for js in episode['joint_states']:
        #     print(len(js))
    
        np.savez(outdir / f'{UID}.npz', **episode)

        # now load it back and play it back
        # loaded_ep = None
        # with np.load(outdir / f'{UID}.npz') as ep:
        #     loaded_ep = {k: ep[k] for k in ep.keys()}
        # playback_npz(loaded_ep)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d0', '--dir0', type=str, required=True)
    # parser.add_argument('-d1', '--dir1', type=str, required=True)
    parser.add_argument('-p', '--root', type=str, default='~/')
    parser.add_argument('-dir', '--directory', type=str, required=True)
    parser.add_argument('-s', '--stop-arm', action='store_true')
    parser.add_argument('-na', '--no-arm', action='store_true')
    parser.add_argument('-c', '--crop-dim', type=int, default=0)
    parser.add_argument('-clo', '--crop-left-offset', type=int, default=0)
    parser.add_argument('-v', '--show-video', action='store_true')
    parser.add_argument('-ti', '--top-index', type=str, default='4')
    parser.add_argument('-bi', '--bottom-index', type=str, default='0')
    parser.add_argument('-b', '--dont-publish', action='store_true')
    parser.add_argument('-r', '--percent_threshold', type=float, default=0.8)
    args = parser.parse_args()

    # print args
    print(f"{args=}")

    rospy.set_param('crop_dim', args.crop_dim)
    rospy.set_param('crop_left_offset', args.crop_left_offset)
    rospy.set_param('top_index', args.top_index)
    rospy.set_param('bottom_index', args.bottom_index)

    path = Path(args.root).expanduser() / args.directory

    print(f"Playing back from {path}")
    # print("hahahah")
    # arm = kortex_arm.Arm()
    # arm.home_arm()
    # print('done')
    import std_msgs
    bag_complete_pub = rospy.Publisher('/playback_complete', std_msgs.msg.Bool, latch=False)
    try:
        BagVideoPublisher(path, args)
    except rospy.ROSInterruptException:
        pass
    finally:
        bag_complete_pub.publish(std_msgs.msg.Bool(True))
        print(f"Stopping")


    # example 