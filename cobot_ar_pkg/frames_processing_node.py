import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point
import numpy as np
from cv_bridge import CvBridge
import cv2
from cobot_ar_pkg.detectors import BlobDetector, FeatureDetectorORB, FeatureDetectorBRISK, FeatureDetectorSIFT, PointTransformer
from cobot_ar_pkg.utils.utils import NoDetectionException
from cobot_ar_pkg.utils.undistort import UndistortImage
from ament_index_python.packages import get_package_share_path


class FramesProcessing(Node):
    ''' Нода по обработки кадров с камер. '''

    def __init__(self):
        super().__init__('frames_processing')
        self.declare_parameters('', [
            ('mobile_frame_topic',),
            ('fixed_frame_topic',),
            ('info_frame_topic',),
            ('raw_point_topic',),
            ('rps',)
        ])
        self.subscriberMobileFrame = self.create_subscription(
            CompressedImage,
            self.get_parameter('mobile_frame_topic')
                .get_parameter_value()
                .string_value,
            self.HmFrameCallback,
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
            self.get_parameter('raw_point_topic')
                .get_parameter_value()
                .string_value,
            10
        )
        self.bridge = CvBridge()
        # Кадр с подвижной камеры на очках
        self.hmCamFrame = np.zeros((640, 360, 3), dtype=np.uint8)
        # Кадр с фиксорованной камеры в помещении
        self.fixedCamFrame = np.zeros((640, 360, 3), dtype=np.uint8)
        self.timer = self.create_timer(
            1 / self.get_parameter('rps').get_parameter_value().integer_value,
            self.TimerCallback
        )
        self.featureDetector = BlobDetector()
        self.matchDetector = FeatureDetectorBRISK()
        self.pointTransformer = PointTransformer(
            get_package_share_path('cobot_ar_pkg') /
            'config' / 'calibration_data_static.json'
        )

        self.fixedFrameUndistort = UndistortImage(
            get_package_share_path('cobot_ar_pkg') /
            'config' / 'calibration_data_static.json'
        )
        self.hmFrameUndistort = UndistortImage(
            get_package_share_path('cobot_ar_pkg') /
            'config' / 'calibration_data_hd.json'
        )

    def __findBlobAndPublishInfoImage(self, mobileFrame):
        ''' Поиск ближайщего отвертия и отправка изображения с его выделением. '''
        try:
            # Find blob
            imageWithDetection, blob = self.featureDetector.Detect(
                mobileFrame)
            msg = self.bridge.cv2_to_compressed_imgmsg(
                imageWithDetection, 'png')
            # Publish
            self.publisherInfoImage.publish(msg)
            cv2.imshow('displaying', imageWithDetection)
            return blob
        except NoDetectionException as ex:
            self.get_logger().warning(str(ex))
            empty = np.zeros(
                (mobileFrame.shape[0], mobileFrame.shape[1], 4), dtype=np.uint8
            )
            msg = self.bridge.cv2_to_compressed_imgmsg(
                empty, 'png')
            self.publisherInfoImage.publish(msg)
            return None

    def __matchImagesAndTransformBlob(self, blob, fixedFrame, mobileFrame):
        ''' Совмещение изображений и преобразование координат отвертия. '''
        try:
            # Matching
            imageAnnotated, projMtx = self.matchDetector.Detect(
                mobileFrame, fixedFrame
            )
            cv2.imshow('match', imageAnnotated)
            # Transform
            x, y, z = self.pointTransformer.Transform(blob, projMtx)
            return (x[0], y[0], z[0])
        except NoDetectionException as ex:
            self.get_logger().warning(str(ex))

    def __pointPublish(self, point):
        ''' Отправка коордиат отвертия. '''
        msg = Point()
        msg.x = point[0]
        msg.y = point[1]
        msg.z = point[2]
        self.publisherPoint.publish(msg)

    def TimerCallback(self):
        ''' Callback функция для обработки данных с камер. '''
        # Undistort
        undistHmFrame = self.hmFrameUndistort.Undistort(self.hmCamFrame)
        undistFixFrame = self.fixedFrameUndistort.Undistort(self.fixedCamFrame)
        # Processing
        if (blob := self.__findBlobAndPublishInfoImage(undistHmFrame)) != None:
            if (point := self.__matchImagesAndTransformBlob(blob, undistFixFrame, undistHmFrame)) != None:
                self.__pointPublish(point)
        cv2.waitKey(1)

    def HmFrameCallback(self, msg):
        ''' Callback функция для сохранения кадров с подвижной камеры. '''
        cvImg = self.bridge.compressed_imgmsg_to_cv2(msg)
        self.hmCamFrame = cv2.rotate(cvImg, cv2.ROTATE_180)
        cv2.imshow('mobile_camera', self.hmCamFrame)

    def FixedFrameCallback(self, msg):
        ''' Callback функция для сохранения кадров с неподвижной камеры. '''
        self.fixedCamFrame = self.bridge.imgmsg_to_cv2(msg)
        cv2.imshow('fixed_camera', self.fixedCamFrame)


def main(args=None):
    rclpy.init(args=args)
    framesProcessingNode = FramesProcessing()
    rclpy.spin(framesProcessingNode)
    framesProcessingNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
