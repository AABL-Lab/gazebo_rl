import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray
from collections import deque
import cv2
import Xlib.threaded
import numpy as np
video_topics = ['camera_obs__dev_video4_96x96', 'camera_obs__dev_video0_96x96']

short_moving_avg = deque(maxlen=10)
long_moving_avg = deque(maxlen=100)
def action_callback(msg):
    short_moving_avg.append(msg.data)
    long_moving_avg.append(msg.data)

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


    # Process the image (e.g., apply filters, detect objects)
    cv2.imshow(f'{title} {cv_image.shape}', cv_image)
    cv2.waitKey(1)



def main():
    rospy.init_node('image_subscriber', anonymous=True)
    cv2.startWindowThread()
    rospy.Subscriber(video_topics[0], Image, lambda x: callback(x, 'top'))
    rospy.Subscriber(video_topics[1], Image, lambda x: callback(x, 'bottom'))

    rospy.Subscriber('action', Float32MultiArray, action_callback)
    rospy.spin()

if __name__ == '__main__':
    main()