import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('cobot_ar_pkg'),
        'config',
        'config.yaml'
    )

    unityEndpoint = Node(
        package="ros_tcp_endpoint",
        executable="default_server_endpoint",
        name="server_endpoint",
        output="screen",
    )

    calibration = Node(
        package="cobot_ar_pkg",
        executable="camera_calibration",
        name="camera_calibration",
        output="screen",
        prefix="xterm -e",
        parameters=[config]
    )

    framesProcessing = Node(
        package="cobot_ar_pkg",
        executable="frames_processing",
        name="frames_processing",
        output="screen",
        parameters=[config]
    )

    pointProcessing = Node(
        package="cobot_ar_pkg",
        executable="point_processing",
        name="point_processing",
        output="screen",
        parameters=[config]
    )

    fixedCamera = Node(
        package="cobot_ar_pkg",
        executable="camera",
        name="fixed_camera",
        output="screen",
        parameters=[config]
    )

    return LaunchDescription([
        # unityEndpoint,
        # calibration,
        framesProcessing,
        pointProcessing,
        # fixedCamera,
    ])
