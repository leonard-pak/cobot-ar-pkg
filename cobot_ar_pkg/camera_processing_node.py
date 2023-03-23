import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
from cv_bridge import CvBridge
import cv2
from cv2 import aruco


class CameraProcessing(Node):
    def __init__(self):
        super().__init__('camera_processing')
        self.declare_parameters('', [
            ('camera_sub_topic',),
            ('info_pub_topic',),
            ('rps',)
        ])
        self.subscription = self.create_subscription(
            CompressedImage,
            self.get_parameter('camera_sub_topic')
                .get_parameter_value()
                .string_value,
            self.ListenerCallback,
            10
        )
        self.publisher = self.create_publisher(
            CompressedImage,
            self.get_parameter('info_pub_topic')
                .get_parameter_value()
                .string_value,
            10
        )
        self.bridge = CvBridge()
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        self.detectorAruco = aruco.ArucoDetector(aruco_dict, parameters)
        self.infoImage = np.zeros((640, 360, 4), dtype=np.uint8)
        self.timer = self.create_timer(
            1 / self.get_parameter('rps').get_parameter_value().integer_value,
            self.TimerCallback
        )

    def TimerCallback(self):
        msg = self.bridge.cv2_to_compressed_imgmsg(self.infoImage, 'png')
        self.publisher.publish(msg)

    def ListenerCallback(self, msg):
        cvImage = cv2.rotate(
            self.bridge.compressed_imgmsg_to_cv2(msg), cv2.ROTATE_180)
        self.ArucoDetect(cvImage)

    def ArucoDetect(self, image):
        corners, ids, rejectedCandidates = self.detectorAruco.detectMarkers(
            image)
        res = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        if ids is not None:
            self.get_logger().info('Aruco marker detected')
            aruco.drawDetectedMarkers(res, corners)
        alpha = np.uint8((np.sum(res, axis=-1) > 0) * 255)
        self.infoImage = np.dstack((res, alpha))
        cv2.imshow('Receive', image)
        cv2.imshow('Detect', res)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = CameraProcessing()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
