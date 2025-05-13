import cv2
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import pykinect_azure as pykinect
from std_msgs.msg import Header
import numpy as np

if __name__ == "__main__":
    # init ros node
    rospy.init_node('kinect_node', anonymous=True)
    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()
    
    depth_pub = rospy.Publisher('/camera/depth_registered/points', PointCloud2, queue_size=1)

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)

    # cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)
    while True:

        # Get capture
        capture = device.update()

        # Get the color image from the capture
        ret, color_image = capture.get_color_image()
        if not ret:
            continue

        # Plot the image
        # cv2.imshow("Color Image",color_image)
        
        # ret, depth_image = capture.get_depth_image()

        if use_color := False:
            # Define PointFields for x, y, z, and RGB
            fields = [
                PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
                # PointField for RGB is a single 32-bit UINT that packs the r,g,b channels
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
            ]
            
            # Convert each point to the format: (x, y, z, rgb_packed)
            # We pack r,g,b into a single float32 or uint32 field
            cloud_data = []
            for pt in capture.get:
                x, y, z, r, g, b = pt
                # Pack RGB into a single 24-bit word, then store as int
                rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                # Alternatively, you can use float32 packing, but this approach is straightforward
                cloud_data.append((x, y, z, rgb))
        else:
            # publish the depth image
            ret, depth_data = capture.get_pointcloud()
            if not ret:
                continue

            # list all the fields in the point cloud
            # from IPython import embed; embed()

            # print the mean depth value across the point cloud
            # mins = np.min(depth_data, axis=0)
            # maxs = np.max(depth_data, axis=0)
            # for i in range(3):
            #     print(f"min: {mins[i]:.2f}, max: {maxs[i]:.2f}", end=', ')
            # print()

            # normalize the 3d point values
            depth_data = depth_data / 1000.0 # mm to meters


            fields = [
                PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            ]
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'camera_link'

        # pc2_msg.header.stamp = rospy.Time.now()
        # pc2_msg.header.frame_id = 'camera_link'

        # pc2_msg.height = 1 # since it's an unordered point cloud
        # pc2_msg.width = depth_msg.shape[0] # number of points
        # pc2_msg.fields = [PointField('x', 0, PointField.INT16, 1),
        #                   PointField('y', 2, PointField.INT16, 1),
        #                   PointField('z', 4, PointField.INT16, 1)]


        # pc2_msg.is_bigendian = False

        # print(f'{depth_image.shape=}', depth_msg.shape, depth_msg.dtype, depth_msg[depth_msg != 0].min(), depth_msg[depth_msg != 0].max())

        # # length of a point in bytes
        # pc2_msg.point_step = 6
        # pc2_msg.row_step = pc2_msg.point_step * depth_msg.shape[0]
        # pc2_msg.is_dense = True
        # pc2_msg.data = depth_msg.tobytes()

        pc2_msg = pc2.create_cloud(header, fields, depth_data)


        depth_pub.publish(pc2_msg)
            
        # Press q key to stop
        # if cv2.waitKey(1) == ord('q'): 
        #     break
    
        rospy.sleep(0.0333333)
