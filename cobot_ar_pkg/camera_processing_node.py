import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
from cv_bridge import CvBridge
import cv2
from cobot_ar_pkg.detectors import BlobDetector


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
        self.infoImage = np.zeros((640, 360, 4), dtype=np.uint8)
        self.timer = self.create_timer(
            1 / self.get_parameter('rps').get_parameter_value().integer_value,
            self.TimerCallback
        )
        self.detector = BlobDetector()

    def TimerCallback(self):
        msg = self.bridge.cv2_to_compressed_imgmsg(self.infoImage, 'png')
        self.publisher.publish(msg)

    def ListenerCallback(self, msg):
        cvImage = cv2.rotate(
            self.bridge.compressed_imgmsg_to_cv2(msg), cv2.ROTATE_180)
        self.infoImage = self.detector.Detect(cvImage)


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = CameraProcessing()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
