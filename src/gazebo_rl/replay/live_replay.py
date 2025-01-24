import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray
from collections import deque
import cv2
import Xlib.threaded
import numpy as np
from kortex_driver.msg import *

video_topics = ['camera_obs__dev_video4_96x96', 'camera_obs__dev_video0_96x96']

short_moving_avg = deque(maxlen=10)
long_moving_avg = deque(maxlen=100)
def action_callback(msg):
    short_moving_avg.append(msg.data)
    long_moving_avg.append(msg.data)

state = [0, 0, 0]
def state_callback(msg):
    global state
    tool_pose = msg.base.tool_pose_x, msg.base.tool_pose_y, msg.base.tool_pose_z
    state = tool_pose

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
    """
    # Ensure fullness is clamped between 0 and 1
    fullness_clamped = max(0.0, min(fullness, 1.0))

    # Convert fullness to degrees (0 - 360)
    end_angle_deg = int(360 * fullness_clamped)

    # cv2.ellipse parameters:
    #   - center: (x, y)
    #   - axes: (radius_x, radius_y)
    #   - angle of rotation of the ellipse (in degrees)
    #   - startAngle: where arc starts (in degrees)
    #   - endAngle: where arc ends (in degrees)
    #   - color: (B, G, R)
    #   - thickness: -1 for filled, > 0 for outline thickness
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

def callback(data, title):
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(e)

    # upscale the image
    cv_image = cv2.resize(cv_image, (0,0), fx=5, fy=5)

    # print(f'{title} {cv_image.shape}')
    midpoint = (cv_image.shape[1] // 2, cv_image.shape[0] // 2)
    # draw an arrow on the image corresponding to the long moving average of the first two dimensions
    if len(long_moving_avg) > 0:
        avg = np.mean(long_moving_avg, axis=0)
        cv2.arrowedLine(cv_image, midpoint, (midpoint[0] + int(avg[0] * midpoint[0]), midpoint[1] - int(avg[1] * midpoint[1])), (0, 255, 0), 2)

        short = np.mean(short_moving_avg, axis=0)
        cv2.arrowedLine(cv_image, midpoint, (midpoint[0] + int(short[0] * midpoint[0]), midpoint[1] - int(short[1] * midpoint[1])), (0, 0, 255), 2)
        
        # avg = np.mean(long_moving_avg)
        # cv2.arrowedLine(cv_image, (50, 50), (50 + int(avg * 10), 50), (0, 255, 0), 2)

    # draw a circle partially filled in based on the third dimension of state
    # put it in the lower left corner
    circle_r = 50
    circle_midpoint = (circle_r, cv_image.shape[0] - circle_r)
    cv_image = draw_partial_circle(cv_image, circle_midpoint, circle_r, state[2] / 0.6, (255, 0, 0), -1)

    # Process the image (e.g., apply filters, detect objects)
    cv2.imshow(f'{title} {cv_image.shape}', cv_image)
    cv2.waitKey(1)



def main():
    rospy.init_node('image_subscriber', anonymous=True)
    cv2.startWindowThread()
    rospy.Subscriber(video_topics[0], Image, lambda x: callback(x, 'top'))
    rospy.Subscriber(video_topics[1], Image, lambda x: callback(x, 'bottom'))
    rospy.Subscriber(f"/my_gen3_lite/base_feedback", BaseCyclic_Feedback, state_callback)

    rospy.Subscriber('action', Float32MultiArray, action_callback)
    rospy.spin()

if __name__ == '__main__':
    main()