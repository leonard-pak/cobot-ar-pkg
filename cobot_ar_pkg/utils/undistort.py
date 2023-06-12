import cv2
import numpy as np
import json


class UndistortImage:
    ''' Класс компенсации дисторсии. '''

    def __init__(self, configDataPath) -> None:
        self.__initConfig(configDataPath)

    def __initConfig(self, configDataPath):
        ''' Иницализация матриц преобразования.'''
        with open(configDataPath) as f:
            data = json.load(f)
            self.camMtx = np.array(data['camera_matrix'])
            self.newCamMtx = np.array(data['new_camera_matrix'])
            self.dist = np.array(data['dist_coeff'])
            self.roi = np.array(data['roi'])

    def Undistort(self, frame):
        ''' Компенсация дисторсии и образание кадра '''

        # undistort
        dst = cv2.undistort(frame, self.camMtx, self.dist,
                            None, self.newCamMtx)
        # crop the image
        x, y, w, h = self.roi
        dst = dst[y:y+h, x:x+w]
        return dst
