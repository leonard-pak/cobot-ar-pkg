import math
from cobot_ar_pkg.utils.utils import Rotate, Mask, CalcLength
import cv2
import numpy as np
import json
from numpy.linalg import inv
from rclpy.logging import get_logger
import time


def BuildPointTransformer(configDataPath):
    ''' Строитель функции преобразования координат '''

    with open(configDataPath) as f:
        data = json.load(f)
        A = np.array(data['new_camera_matrix'])
        R, _ = cv2.Rodrigues(np.array(data['rvec']))
        t = np.array(data['tvec'])
        s = float(data['scale'])

    def wrapper(point, mtx):
        '''
        Преобразует координаты точки
        в начале через перспективное преобразованиепо переданной матрице,
        а затем по формуле:

        xyzW = R^(-1)*(A^(-1) * s * uv1 - t)
        '''
        npPoint = np.array([[[point[0], point[1]]]], dtype=np.float32)
        newPoint = cv2.perspectiveTransform(npPoint, mtx)

        uv1 = np.array(
            [[newPoint[0][0][0], newPoint[0][0][1], 1]],
            dtype=np.float32
        ).T
        suv1 = s * uv1
        xyzC = inv(A).dot(suv1)
        xyzW = inv(R).dot(xyzC - t)
        return xyzW

    return wrapper


def BuildMaskDetectionWindowInImage(xMargin=50, yMargin=50):
    ''' Строитель функции выделения области поиска '''

    window = (
        (-xMargin / 2, 0),
        (-xMargin / 2, yMargin),
        (xMargin / 2, yMargin),
        (xMargin / 2, 0),
    )

    def wrapper(tip, dip, image):
        '''
        Рассчитывает окно, которое перпендикулярно расположенно к концу прямой, переданной в функцию как 2 точки.

        Parameters
        ----------
        tip: tuple | list
            конец прямой
        dip: tuple | list
            начало прямой
        '''
        xBase = tip[0] - dip[0]
        yBase = tip[1] - dip[1]
        theta = math.atan2(yBase, xBase)
        if xBase < 0 < yBase:
            theta -= math.pi / 2
        else:
            theta += math.pi * 1.5
        points = []
        for point in window:
            pointRotated = Rotate(theta, point)
            points.append([
                int(pointRotated[0] + tip[0]),
                int(pointRotated[1] + tip[1])
            ])
        return Mask(image, points)

    return wrapper


def BuildUndistortImage(configDataPath):
    ''' Строить функции компенсации дисторсии. '''

    with open(configDataPath) as f:
        data = json.load(f)
        camMtx = np.array(data['camera_matrix'])
        newCamMtx = np.array(data['new_camera_matrix'])
        dist = np.array(data['dist_coeff'])
        roi = np.array(data['roi'])

    def wrapper(frame):
        ''' Компенсация дисторсии и образание кадра '''
        # undistort
        dst = cv2.undistort(frame, camMtx, dist,
                            None, newCamMtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst

    return wrapper


def DecorateAverageTimeExecute(func):
    ''' Декоратор подсчета времени выполнения '''
    buffer = []

    def wrapper(*args, **kwargs):
        start = time.monotonic()
        returnVal = func(*args, **kwargs)
        end = time.monotonic()
        buffer.append(end - start)
        get_logger('UTILS').warning(f'Time: {np.average(buffer)}')
        return returnVal
    return wrapper


def DecorateAverageAbsoluteError(func):
    ''' Декоратор подсчета ошибки вычисления координат '''
    buffer = []
    realPoint = (0.806, 0.363)

    def wrapper(*args, **kwargs):
        returnVal = func(*args, **kwargs)
        calcPoint = (returnVal[0], returnVal[1])
        buffer.append(CalcLength(realPoint, calcPoint))
        get_logger('UTILS').warning(
            f'Error mm: {np.average(buffer) * 100}')
        return returnVal
    return wrapper


def DecorateAbsoluteError(func):
    ''' Декоратор подсчета ошибки вычисления координат '''
    realPoint = (0.463, 0.158)

    def wrapper(*args, **kwargs):
        returnVal = func(*args, **kwargs)
        calcPoint = (returnVal[0], returnVal[1])
        error = CalcLength(realPoint, calcPoint)
        get_logger('UTILS').warning(
            f'Error mm: {error * 1000}')
        return returnVal
    return wrapper
