import time
from typing import List, Tuple

from cobot_ar_pkg.utils.utils import CalcLength
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from cobot_ar_pkg.utils.filters import MedianFilter
from leonard_interfaces.srv import GoToPoint


class PointProcessing(Node):
    ''' Нода по обработки точек. '''

    def __init__(self) -> None:
        super().__init__('point_processing')
        self.declare_parameters('', [
            ('raw_point_topic',),
            ('go_to_point_service',),
            ('window_time_sec',),
            ('rps',),
            ('max_error_meters',),
            ('filter_memory_size',),
        ])
        self.subscriberRawPoint = self.create_subscription(
            Point,
            self.get_parameter('raw_point_topic')
            .get_parameter_value()
                .string_value,
            self.RawPointCallback,
            10
        )
        # self.publisherTargetPoint = self.create_publisher(
        #     Point,
        #     self.get_parameter('go_to_point_service')
        #     .get_parameter_value()
        #         .string_value,
        #     10
        # )
        self.goToPointClient = self.create_client(
            GoToPoint,
            self.get_parameter('go_to_point_service')
            .get_parameter_value()
            .string_value
        )
        while not self.goToPointClient.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('service not available, waiting again...')
        self.req = GoToPoint.Request()

        self.timer = self.create_timer(
            1 / self.get_parameter('rps').get_parameter_value().integer_value,
            self.Callback
        )
        filterSize = self.get_parameter(
            'filter_memory_size'
        ).get_parameter_value().integer_value
        self.filterX = MedianFilter(filterSize)
        self.filterY = MedianFilter(filterSize)

        self.point = None
        self.lastTimestamp = 0.0
        self.lastPointstamp = None
        self.timeWindow = self.get_parameter(
            'window_time_sec'
        ).get_parameter_value().double_value
        self.maxError = self.get_parameter(
            'max_error_meters'
        ).get_parameter_value().double_value

        self.future = None

    def __publishTargetPoint(self, point):
        ''' Публикация обработанной точки. '''
        msg = Point()
        msg.x = point[0]
        msg.y = point[1]
        # self.publisherTargetPoint.publish(msg)
        self.req.target = msg
        self.future = self.goToPointClient.call_async(self.req)

    def __filterRawPoint(self, point):
        ''' Фильтрация координат  точки. '''
        x = self.filterX.Filtering(point[0])
        y = self.filterY.Filtering(point[1])
        return (x, y)

    def Callback(self):
        ''' Callback функция таймера. '''
        if self.point == None or (self.future != None and not self.future.done()):
            return

        now = time.monotonic()
        if self.lastTimestamp == 0.0:
            self.lastTimestamp = now
            self.lastPointstamp = self.point
        elif CalcLength(self.point, self.lastPointstamp) >= self.maxError:
            self.lastTimestamp = now
            self.lastPointstamp = self.point
        elif now - self.lastTimestamp > self.timeWindow:
            self.__publishTargetPoint(self.lastPointstamp)
            self.lastTimestamp = now
            self.lastPointstamp = self.point
        self.point = None

    def RawPointCallback(self, point):
        ''' Callback функция по приему необработанных точек. '''
        filterPoint = self.__filterRawPoint((point.x, point.y))
        self.point = filterPoint


def main():
    rclpy.init()
    pointProcessingNode = PointProcessing()
    rclpy.spin(pointProcessingNode)
    pointProcessingNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
