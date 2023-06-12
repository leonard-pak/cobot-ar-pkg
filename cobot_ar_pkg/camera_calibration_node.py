import json
import sys
import traceback

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cobot_ar_pkg.utils.calibration as calib
import numpy as np
from enum import Enum


class Status(Enum):
    DISABLE = 0
    CHESS_CALIBRATION = 1
    PROJECTIVE_CALIBRATION = 3


class CameraCalibration(Node):
    ''' Нода для калибровки камеры '''

    def __init__(self):
        super().__init__('camera_calibrarion')
        self.declare_parameters('', [
            ('image_topic',),
            ('save_conf_prefix',),
            ('save_conf_postfix',),
            ('images_dir',),
            ('is_compressed',),
        ])
        if self.get_parameter('is_compressed').get_parameter_value().bool_value:
            self.subscriber = self.create_subscription(
                CompressedImage,
                self.get_parameter('image_topic')
                .get_parameter_value()
                .string_value,
                self.CompressedImageCallback,
                10
            )
        else:
            self.subscriber = self.create_subscription(
                Image,
                self.get_parameter('image_topic')
                .get_parameter_value()
                .string_value,
                self.ImageCallback,
                10
            )

        self.bring = CvBridge()
        self.img = np.zeros((480, 640, 3), dtype=np.uint8)
        calib.SaveDirs(
            self.get_parameter('save_conf_prefix')
            .get_parameter_value()
            .string_value,
            self.get_parameter('save_conf_postfix')
            .get_parameter_value()
            .string_value,
            self.get_parameter('images_dir')
            .get_parameter_value()
            .string_value,
        )
        self.timer = self.create_timer(0.1, self.ProcessCallback)
        self.status = Status.DISABLE

    def Calibration(self):
        '''
            Метод калибровки камеры.
            Запускает отдельные части калибровки в зависимости от выставленного статуса калибровки.
        '''
        if self.status == Status.CHESS_CALIBRATION:
            self.calibData = calib.ChessCalibration()
            if self.calibData == None:
                self.get_logger().info(
                    f'Chess calibration failed!'
                )
                self.status = Status.DISABLE
                return
            self.status = Status.PROJECTIVE_CALIBRATION
        if self.status == Status.PROJECTIVE_CALIBRATION:
            undistortImg = calib.Undistort(
                self.img, self.calibData[0], self.calibData[1], self.calibData[2], self.calibData[3])
            imageCenter = calib.DrawCenter(
                self.calibData[0], undistortImg)
            if imageCenter == None:
                return
            worldCenter = [
                float(i) for i in input(f'Add "x y z" for center: ').split(' ')
            ]
            imagePts = calib.BlobDetect(undistortImg)
            worldPts = calib.InputKpWorldCoords(
                imagePts, worldCenter, undistortImg
            )
            imagePts.append(imageCenter)
            worldPts.append(worldCenter)
            rvec, tvec, scale = calib.ProjectiveCalibration(
                worldPts, imagePts, self.calibData[2], self.calibData[1])

            data = {
                "camera_matrix": self.calibData[0].tolist(),
                "dist_coeff": self.calibData[1].tolist(),
                "new_camera_matrix": self.calibData[2].tolist(),
                "roi": self.calibData[3].tolist(),
                "image_points": imagePts,
                "world_points": worldPts,
                "rvec": rvec.tolist(),
                "tvec": tvec.tolist(),
                "scale": scale
            }
            calib.SaveConfig(data)
            self.status = Status.DISABLE

    def Runtime(self):
        ''' Метод отображения картинки и обработки нажатой клавиши '''
        c = calib.RuntimeShow(self.img, True)
        if c & 0xFF == ord('q'):
            rclpy.shutdown()
        if c & 0xFF == ord('s'):
            name = calib.MakeShot(self.img)
            self.get_logger().info(
                f'Image save {name}'
            )
        if c & 0xFF == ord('b'):
            calib.BlobDetect(self.img)
        if c & 0xFF == ord('v'):
            blobs = calib.FindBlobs(self.img)
            for blob in blobs:
                self.get_logger().info(
                    f'Find blob: x-{blob[0]} y-{blob[1]}'
                )
        if c & 0xFF == ord('c'):
            calibData = calib.ChessCalibration()
            data = {
                "camera_matrix": calibData[0].tolist(),
                "dist_coeff": calibData[1].tolist(),
                "new_camera_matrix": calibData[2].tolist(),
                "roi": calibData[3].tolist()
            }
            calib.SaveConfig(data)
            self.get_logger().info(
                f'SUCCESS chess calibration at {calibData[4]} images and save config'
            )
        if c & 0xFF == ord('f'):
            self.status = Status.CHESS_CALIBRATION
        if c & 0xFF == ord('r'):
            with open(calib.saveConfPrefix + 'calibration_data' + calib.saveConfPostfix + '.json') as f:
                data = json.load(f)
                cameraMtx = np.array(data['camera_matrix'])
                distCoeff = np.array(data['dist_coeff'])
                undistorCameraMtx = np.array(data['new_camera_matrix'])
                roi = np.array(data['roi'])
                self.calibData = [cameraMtx,
                                  distCoeff, undistorCameraMtx, roi]
            self.status = Status.PROJECTIVE_CALIBRATION

    def ProcessCallback(self):
        ''' Callback метод для таймера'''
        try:
            if self.status != Status.DISABLE:
                self.Calibration()
            else:
                self.Runtime()
        except Exception:
            self.status = Status.DISABLE
            exType, _, tb = sys.exc_info()
            print(f"Expected {exType}")
            traceback.print_tb(tb)

    def CompressedImageCallback(self, frame):
        ''' Callback метод получения сжатой картинки'''
        self.img = self.bring.compressed_imgmsg_to_cv2(frame)

    def ImageCallback(self, frame):
        ''' Callback метод получения картинки'''
        self.img = self.bring.imgmsg_to_cv2(frame)


def main():
    rclpy.init()
    cameraCalibrationNode = CameraCalibration()
    rclpy.spin(cameraCalibrationNode)
    cameraCalibrationNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
