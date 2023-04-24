import imp
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__('camera')
        self.declare_parameters('', [
            ('camera_idx',),
            ('frame_topic',),
            ('fps',)
        ])
        self.publisher = self.create_publisher(
            Image,
            self.get_parameter('frame_topic')
            .get_parameter_value()
            .string_value,
            10
        )
        self.cap = cv2.VideoCapture(
            self.get_parameter('camera_idx')
            .get_parameter_value()
            .integer_value
        )
        self.timer = self.create_timer(
            1 / self.get_parameter('fps')
            .get_parameter_value()
            .integer_value, self.TimerCallback
        )
        self.cvBridge = CvBridge()

    def __del__(self):
        self.cap.release()

    def TimerCallback(self):
        if not self.cap.isOpened():
            self.get_logger().error('Camera is not open.')
            return
        success, image = self.cap.read()
        if not success:
            self.get_logger().warning('Ignoring empty camera frame.')
            return
        self.publisher.publish(self.cvBridge.cv2_to_imgmsg(image))


def main(args=None):
    rclpy.init(args=args)
    fixed_camera = CameraNode()
    rclpy.spin(fixed_camera)
    fixed_camera.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
