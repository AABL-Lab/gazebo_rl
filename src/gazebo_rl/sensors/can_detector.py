import cv2
import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t
from ultralytics import YOLO
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped

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
    
    def get_bounding_box(self, image):
        results = self.predict(image, show=False)
        boxes = []
        for result in results:
            boxes.append(result.boxes.xyxy)

        # return the midpoint of the box
        box_midpoints = []
        for box in boxes:
            box = box.cpu().numpy()
            box = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
            box_midpoints.append(box)
        return box_midpoints
    
class CanDetector:
    def __init__(self):
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
        self.pose.header.frame_id = "map"
        self.pose.header.stamp = rospy.Time.now()

        # point cloud publisher
        self.point_cloud_pub = rospy.Publisher('/can_detector/point_cloud', PointCloud2, queue_size=1)

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
        box_midpoints = self.model.get_bounding_box(yolo_color_image)

        if box_midpoints := self.model.get_bounding_box(yolo_color_image):
            pix_x = int(box_midpoints[0][0])
            pix_y = int(box_midpoints[0][1])

            for box in box_midpoints:
                box = (box[0], box[1])
                cv2.circle(self.color_image, (int(box[0]), int(box[1])), 5, (0, 255, 0), -1)

        # Get the colored depth
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

        if not ret_color or not ret_depth:
            return
        
        rgb_depth = transformed_depth_image[pix_y, pix_x]

        pixels = k4a_float2_t((pix_x, pix_y))

        pos3d_color = self.device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
        pos3d_depth = self.device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)

        print(f"RGB depth: {rgb_depth}, RGB pos3D: {pos3d_color}, Depth pos3D: {pos3d_depth}")

        self.pose.header.stamp = rospy.Time.now()
        self.pose.pose.position.x = pos3d_depth.xyz.x / 1000
        self.pose.pose.position.y = pos3d_depth.xyz.y / 1000
        self.pose.pose.position.z = pos3d_depth.xyz.z / 1000
        self.pose.pose.orientation.x = 0
        self.pose.pose.orientation.y = 0
        self.pose.pose.orientation.z = 0
        self.pose.pose.orientation.w = 1
        self.pub.publish(self.pose)

        if PUBLISH_POINT_CLOUD := False:
            ret_points, points = capture.get_pointcloud()
            if not ret_points:
                return
            points = points.reshape((-1, 3))
            # Create a PointCloud2 message
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "map"
            point_cloud_msg = PointCloud2()
            point_cloud_msg.header = header
            point_cloud_msg.height = 1
            point_cloud_msg.width = len(points)
            point_cloud_msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            point_cloud_msg.is_bigendian = False
            point_cloud_msg.point_step = 12
            point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
            point_cloud_msg.is_dense = True
            point_cloud_msg.data = points.tobytes()
            self.point_cloud_pub.publish(point_cloud_msg)


if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node('can_detector', anonymous=True)
    # Initialize the CanDetector class
    can_detector = CanDetector()
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
    #     box_midpoints = model.get_bounding_box(yolo_color_image)

    #     if box_midpoints := model.get_bounding_box(yolo_color_image):
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