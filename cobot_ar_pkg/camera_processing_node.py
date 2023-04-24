import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import numpy as np
from cv_bridge import CvBridge
import cv2
from cobot_ar_pkg.detectors import BlobDetectorV2, MatchDetector
from cobot_ar_pkg.utils import NoDetectionException


class CameraProcessing(Node):
    def __init__(self):
        super().__init__('camera_processing')
        self.declare_parameters('', [
            ('mobile_frame_topic',),
            ('fixed_frame_topic',),
            ('info_frame_topic',),
            ('rps',)
        ])
        self.subscriberMobileFrame = self.create_subscription(
            CompressedImage,
            self.get_parameter('mobile_frame_topic')
                .get_parameter_value()
                .string_value,
            self.MobileFrameCallback,
            10
        )
        self.subscriberFixedFrame = self.create_subscription(
            Image,
            self.get_parameter('fixed_frame_topic')
                .get_parameter_value()
                .string_value,
            self.FixedFrameCallback,
            10
        )
        self.publisherInfoImage = self.create_publisher(
            CompressedImage,
            self.get_parameter('info_frame_topic')
                .get_parameter_value()
                .string_value,
            10
        )
        self.bridge = CvBridge()
        # Кадр с подвижной камеры на очках
        self.frameMobile = np.zeros((640, 360, 3), dtype=np.uint8)
        # Кадр с фиксорованной камеры в помещении
        self.frameFixed = np.zeros((640, 360, 3), dtype=np.uint8)
        self.timer = self.create_timer(
            1 / self.get_parameter('rps').get_parameter_value().integer_value,
            self.TimerCallback
        )
        self.featureDetector = BlobDetectorV2()
        self.matchDetector = MatchDetector()

    def __publishDetection(self):
        imageWithDetection = self.featureDetector.Detect(self.frameMobile)
        msg = self.bridge.cv2_to_compressed_imgmsg(imageWithDetection, 'png')
        self.publisherInfoImage.publish(msg)
        cv2.imshow('output', imageWithDetection)

    def TimerCallback(self):
        # self.__publishDetection()
        try:
            imageAnnotated, rectPoints = self.matchDetector.Detect(
                # self.frameMobile, self.frameFixed
                self.frameFixed, self.frameMobile
            )
        except NoDetectionException:
            self.get_logger().warning("No annotated image")
            return
        cv2.imshow('orb', imageAnnotated)
        cv2.waitKey(1)

    def MobileFrameCallback(self, msg):
        self.frameMobile = self.bridge.compressed_imgmsg_to_cv2(msg)
        # self.frameMobile = cv2.rotate(
        #     self.bridge.compressed_imgmsg_to_cv2(msg),
        #     cv2.ROTATE_180
        # )
        cv2.imshow('mobile_camera', self.frameMobile)

    def FixedFrameCallback(self, msg):
        self.frameFixed = self.bridge.imgmsg_to_cv2(msg)
        cv2.imshow('fixed_camera', self.frameFixed)


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = CameraProcessing()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
