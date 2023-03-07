import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point32
import random
import cv2
from cv2 import aruco

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            CompressedImage,
            'camera_ar',
            self.listener_callback,
            10
        )
        self.publisher = self.create_publisher(
            Point32,
            'displaying_info',
            10
        )
        self.timer = self.create_timer(0.5, self.timer_callback)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        self.detectorAruco = aruco.ArucoDetector(aruco_dict, parameters)

    def timer_callback(self):
        msg = Point32()
        msg.x = random.randint(100, 200)
        msg.y = random.randint(100, 200)
        self.publisher.publish(msg)
        self.get_logger().info(f'Publish: x-{msg.x} y-{msg.y}');

    def listener_callback(self, msg):
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        corners, ids, rejectedCandidates = self.detectorAruco.detectMarkers(cv_image)
        if ids is not None:
            self.get_logger().info(f'Aruco marker detected: {corners[0][0][0][0]} : {corners[0][0][0][1]}')
