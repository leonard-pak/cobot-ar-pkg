from typing import List
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

from cobot_ar_pkg.filters import MedianFilter


class PointProcessing(Node):
    ''' Нода по обработки точек. '''

    def __init__(self) -> None:
        super().__init__('point_processing')
        self.declare_parameters('', [
            ('raw_point_topic',),
            ('target_point_topic',),
            ('window_time_sec',),
        ])
        self.subscriberRawPoint = self.create_subscription(
            Point,
            self.get_parameter('raw_point_topic')
            .get_parameter_value()
                .string_value,
            self.RawPointCallback,
            10
        )
        self.publisherTargetPoint = self.create_publisher(
            Point,
            self.get_parameter('target_point_topic')
            .get_parameter_value()
                .string_value,
            10
        )
        self.filterX = MedianFilter(9)
        self.filterY = MedianFilter(9)
        # self.filterZ = MedianFilter(5)

    def __publishTargetPoint(self, point):
        ''' Публикация обработанной точки. '''
        msg = Point()
        msg.x = point[0]
        msg.y = point[1]
        msg.z = point[2]
        self.publisherTargetPoint.publish(msg)

    def __filterRawPoint(self, point):
        ''' Фильтрация координат  точки. '''
        x = self.filterX.Filtering(point[0])
        y = self.filterY.Filtering(point[1])
        # z = self.filterZ.Filtering(point[2])
        return (x, y, 0.0)

    def RawPointCallback(self, point):
        ''' Callback функция по приему необработанных точек. '''
        filterPoint = self.__filterRawPoint((point.x, point.y, point.z))
        self.__publishTargetPoint(filterPoint)


def main():
    rclpy.init()
    pointProcessingNode = PointProcessing()
    rclpy.spin(pointProcessingNode)
    pointProcessingNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
