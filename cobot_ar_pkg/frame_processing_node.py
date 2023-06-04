import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point
import numpy as np
from cv_bridge import CvBridge
import cv2
from cobot_ar_pkg.detectors import BlobDetector, MatchDetectorORB, PointTransformer, MatchDetectorSIFT
from cobot_ar_pkg.utils import NoDetectionException
from ament_index_python.packages import get_package_share_path


class CameraProcessing(Node):
    def __init__(self):
        super().__init__('frame_processing')
        self.declare_parameters('', [
            ('mobile_frame_topic',),
            ('fixed_frame_topic',),
            ('info_frame_topic',),
            ('target_point_topic',),
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
        self.publisherPoint = self.create_publisher(
            Point,
            self.get_parameter('target_point_topic')
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
        self.featureDetector = BlobDetector()
        self.matchDetector = MatchDetectorORB()
        self.pointTransformer = PointTransformer(
            get_package_share_path('cobot_ar_pkg') / 'config' / 'calibration_data_static.json')

        self.lastSaveTime = time.monotonic()
        self.lastSavePoints = {}

    def __findBlobAndPublish(self):
        try:
            # Find blob
            imageWithDetection, blob = self.featureDetector.Detect(
                self.frameMobile)
            msg = self.bridge.cv2_to_compressed_imgmsg(
                imageWithDetection, 'png')
            self.publisherInfoImage.publish(msg)
            cv2.imshow('displaying', imageWithDetection)
            return blob
        except NoDetectionException as ex:
            self.get_logger().warning(str(ex))
            empty = np.zeros(
                (self.frameMobile.shape[0], self.frameMobile.shape[1], 4), dtype=np.uint8
            )
            msg = self.bridge.cv2_to_compressed_imgmsg(
                empty, 'png')
            self.publisherInfoImage.publish(msg)
            return None

    def __matchAndTransformBlob(self, blob):
        try:
            # Matching
            imageAnnotated, projMtx = self.matchDetector.Detect(
                self.frameMobile, self.frameFixed
            )
            cv2.imshow('match', imageAnnotated)
            x, y, z = self.pointTransformer.Transform(blob, projMtx)
            return (x[0], y[0], z[0])
        except NoDetectionException as ex:
            self.get_logger().warning(str(ex))

    def __pointPublish(self, point):
        msg = Point()
        msg.x = point[0]
        msg.y = point[1]
        msg.z = point[2]
        self.publisherPoint.publish(msg)

    def TimerCallback(self):
        t = time.monotonic()
        if ((blob := self.__findBlobAndPublish()) != None) and ((point := self.__matchAndTransformBlob(blob)) != None):
            self.get_logger().error(
                f'Blob at: x-{point[0]} y-{point[1]} z-{point[2]}'
            )
            self.__pointPublish(point)
            self.get_logger().info(f'Time: {time.monotonic() - t}')
        cv2.waitKey(1)

    def MobileFrameCallback(self, msg):
        # self.frameMobile = self.bridge.compressed_imgmsg_to_cv2(msg)
        self.frameMobile = cv2.rotate(
            self.bridge.compressed_imgmsg_to_cv2(msg),
            cv2.ROTATE_180
        )
        cv2.imshow('mobile_camera', self.frameMobile)

    def FixedFrameCallback(self, msg):
        self.frameFixed = self.bridge.imgmsg_to_cv2(msg)
        cv2.imshow('fixed_camera', self.frameFixed)


def main(args=None):
    rclpy.init(args=args)
    cameraProcessingNode = CameraProcessing()
    rclpy.spin(cameraProcessingNode)
    cameraProcessingNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
