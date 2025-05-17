import cv2
import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t
from ultralytics import YOLO
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
# from tf2_geometry_msgs import do_transform_pose
import tf2_ros, tf
import numpy as np

class FinetunedYOLO:
    def __init__(self):
        # Load a COCO-pretrained YOLOv8n model
        model = YOLO("yolo11n.pt")
        # Display model information (optional)
        model.info()
        model.load("/home/j/workspace/lerobot/runs/detect/train2/weights/best.pt")
        self.model = model

    def __del__(self):
        # Release the model
        self.model = None

    def predict(self, image, show=True):
        results = self.model.predict(source=image, show=show, conf=0.25, iou=0.45, device="cuda:0")
        return results
    
    def get_box_midpoint(self, image):
        results = self.predict(image, show=False)
        boxes = []
        for result in results:
            boxes.append(result.boxes.xyxy)

        # return the midpoint of the box
        try:
            box_midpoints = []
            for box in boxes:
                box = box.cpu().numpy()
                if len(box) == 0: continue
                x, y, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
                midpoint = float((x + x2) / 2), float((y + y2) / 2)
                box_midpoints.append(midpoint)
        except Exception as e:
            print(f"Error in get_box_midpoint: {e}")
            box_midpoints = []
        return box_midpoints
    
class CanDetector:
    def __init__(self, publish_point_cloud=False):
        self.publish_point_cloud = publish_point_cloud
        print(f"Initializing CanDetector {publish_point_cloud=}")
        # Initialize the library, if the library is not found, add the library path as argument
        pykinect.initialize_libraries()

        # Modify camera configuration
        device_config = pykinect.default_configuration
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

        # Start device
        self.device = pykinect.start_device(config=device_config)
        self.model = FinetunedYOLO()
        self.color_image = None
        self.transformed_depth_image = None

        # publish the 3D pose to ROS
        self.pub = rospy.Publisher('/can_detector/pose', PoseStamped, queue_size=10)
        self.pose = PoseStamped()
        self.pose.header.frame_id = "kinect_link"
        self.pose.header.stamp = rospy.Time.now()

        # point cloud publisher
        self.point_cloud_pub = rospy.Publisher('/can_detector/point_cloud', PointCloud2, queue_size=1)

        # static transform publisher
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = "kinect_link"
        transform.transform.translation.x = 0.02
        transform.transform.translation.y = -0.32
        transform.transform.translation.z = 0.22
        euler = 1.23, 3.14, 1.97
        q = tf.transformations.quaternion_from_euler(*euler)
        transform.transform.rotation.x = q[0]; transform.transform.rotation.y = q[1]; transform.transform.rotation.z = q[2]; transform.transform.rotation.w = q[3]
        self.broadcaster.sendTransform(transform)

        # create a buffer and listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.tf_listener = tf.TransformListener()
        self.tf_listener.waitForTransform("base_link", "kinect_link", rospy.Time(0), rospy.Duration(10.0))


    def step(self):
        # Get capture
        capture = self.device.update()

        # Get the color image from the capture
        ret_color, self.color_image = capture.get_color_image()

        if not ret_color:
            return

        # get the yolo results
        # conver bgra to bgr
        yolo_color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGRA2BGR)
        box_midpoints = self.model.get_box_midpoint(yolo_color_image)

        if box_midpoints := self.model.get_box_midpoint(yolo_color_image):
            try:
                pix_x = int(box_midpoints[0][0]) # grab first as default
                pix_y = int(box_midpoints[0][1]) # grab first as default

                for box in box_midpoints:
                    cv2.circle(yolo_color_image, (int(box[0]), int(box[1])), 5, (0, 255, 0), -1)
                cv2.imshow('Transformed Color Image', yolo_color_image)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Error in box_midpoints: {e}")
                from IPython import embed; embed()
        
            # Get the colored depth
            ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

            if not ret_color or not ret_depth:
                return
            
            rgb_depth = transformed_depth_image[pix_y, pix_x]

            pixels = k4a_float2_t((pix_x, pix_y))

            pos3d_color = self.device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
            pos3d_depth = self.device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)

            # print(f"RGB depth: {rgb_depth}, RGB pos3D: {pos3d_color}, Depth pos3D: {pos3d_depth}")

            self.pose.header.stamp = rospy.Time.now()
            self.pose.pose.position.x = pos3d_depth.xyz.x / 1000
            self.pose.pose.position.y = pos3d_depth.xyz.y / 1000
            self.pose.pose.position.z = pos3d_depth.xyz.z / 1000
            self.pose.pose.orientation.x = 0
            self.pose.pose.orientation.y = 0
            self.pose.pose.orientation.z = 0
            self.pose.pose.orientation.w = 1

            # convert the pose from kinect_link to base_link
            # transform = self.tf_buffer.lookup_transform("base_link", self.pose.header.frame_id, self.pose.header.stamp, timeout=rospy.Duration(0.1))
            # pose_out = do_transform_pose(self.pose.pose, transform)
        
            pose_out = self.tf_listener.transformPose("base_link", self.pose)

            self.pub.publish(pose_out)

        if self.publish_point_cloud:
            ret_points, points = capture.get_pointcloud()
            if not ret_points:
                return
            points = points.reshape((-1, 3))
            # Align with rviz convention. z = -y, y = x
            # points[:, 0], points[:, 1], points[:, 2] = points[:, 1], points[:, 0], -points[:, 2]
            # Create a PointCloud2 message
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "kinect_link"
            # point_cloud_msg = PointCloud2()
            # point_cloud_msg.header = header
            # point_cloud_msg.height = 1
            # point_cloud_msg.width = len(points)
            # point_cloud_msg.fields = [
            #     PointField(name='x', offset=0, datatype=PointField.FLOAT16, count=1),
            #     PointField(name='y', offset=4, datatype=PointField.FLOAT16, count=1),
            #     PointField(name='z', offset=8, datatype=PointField.FLOAT16, count=1),
            # ]
            # point_cloud_msg.is_bigendian = False
            # point_cloud_msg.point_step = 12
            # point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
            # point_cloud_msg.is_dense = True
            # point_cloud_msg.data = points.tobytes()
            # Create a PointCloud2 message using the point_cloud2 module
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points.astype(np.float32)/1000)

            self.point_cloud_pub.publish(point_cloud_msg)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Can Detector")
    parser.add_argument('-pc', '--publish_point_cloud', action='store_true', help="Publish point cloud")
    args = parser.parse_args()

    # Initialize the ROS node
    rospy.init_node('can_detector', anonymous=True)
    # Initialize the CanDetector class
    can_detector = CanDetector(args.publish_point_cloud)
    # Create a named window for displaying the color image
    cv2.namedWindow('Transformed Color Image', cv2.WINDOW_NORMAL)
    rate = rospy.Rate(30)  # 30 Hz
    while not rospy.is_shutdown():
        can_detector.step()
        rate.sleep()

    # # Load a COCO-pretrained YOLOv8n model
    # model = FinetunedYOLO()

    # # Initialize the library, if the library is not found, add the library path as argument
    # pykinect.initialize_libraries()

    # # Modify camera configuration
    # device_config = pykinect.default_configuration
    # device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    # device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    # device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # # print(device_config)

    # # Start device
    # device = pykinect.start_device(config=device_config)

    # cv2.namedWindow('Transformed Color Image',cv2.WINDOW_NORMAL)
    # while True:
        
    #     # Get capture
    #     capture = device.update()

    #     # Get the color image from the capture
    #     ret_color, color_image = capture.get_color_image()

    #     if not ret_color:
    #         continue
    #     # get the yolo results
    #     # 		# conver bgra to bgr
    #     yolo_color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    #     # results = model.model.predict(source=yolo_color_image, show=True, conf=0.25, iou=0.45, device="cuda:0")
    #     # results = model.predict(yolo_color_image, show=True)
    #     box_midpoints = model.get_box_midpoint(yolo_color_image)

    #     if box_midpoints := model.get_box_midpoint(yolo_color_image):
    #         pix_x = int(box_midpoints[0][0])
    #         pix_y = int(box_midpoints[0][1])

    #         for box in box_midpoints:
    #             box = (box[0], box[1])
    #             cv2.circle(color_image, (int(box[0]), int(box[1])), 5, (0, 255, 0), -1)

    #     # Get the colored depth
    #     ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

    #     if not ret_color or not ret_depth:
    #         continue

    #     # pix_x = color_image.shape[1] // 2
    #     # pix_y = color_image.shape[0] // 2
    #     rgb_depth = transformed_depth_image[pix_y, pix_x]

    #     pixels = k4a_float2_t((pix_x, pix_y))

    #     pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
    #     pos3d_depth = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)
    #     print(f"RGB depth: {rgb_depth}, RGB pos3D: {pos3d_color}, Depth pos3D: {pos3d_depth}")

    #     # Overlay body segmentation on depth image
    #     cv2.imshow('Transformed Color Image',color_image)

    #     # Press q key to stop
    #     if cv2.waitKey(1) == ord('q'):
    #         break