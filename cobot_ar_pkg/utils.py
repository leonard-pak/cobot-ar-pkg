import math
import cv2
import numpy as np
import time

from rclpy.logging import get_logger


class NoDetectionException(Exception):
    """ Raise, if the detector can not find the target object in the image """


def Rotate(theta: float, point):
    ''' Поворот точки на угол theta. '''
    return (
        point[0] * math.cos(theta) - point[1] * math.sin(theta),
        point[0] * math.sin(theta) + point[1] * math.cos(theta)
    )


def Mask(image, points):
    ''' Наложение маски на изображение по точкам. '''
    mask = np.zeros(image.shape[: 2], dtype='uint8')
    cv2.drawContours(
        mask, [np.array(points)], -1,
        (255, 255, 255), -1, cv2.LINE_4
    )
    return cv2.bitwise_and(image, image, mask=mask)


def GetTimestamp() -> int:
    ''' Возвращает монотонное время в секундах. '''
    return round(time.monotonic() * 1000)


def CalcAngle(p1, p2, p3):
    ''' Расчитывает угол между 3-мя точками. '''
    a = math.hypot(p2[0]-p3[0], p2[1]-p3[1])
    b = math.hypot(p1[0]-p3[0], p1[1]-p3[1])
    c = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
    try:
        return math.acos((b**2-a**2-c**2)/(-2*a*c))
    except:
        get_logger('UTILS').warning(f'a: {a} b: {b} c: {c} ')
        return 0.0


def CalcLength(pt1, pt2):
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])


def averageTimeExecute(func):
    import time
    buffer = []

    def wrapper(*args, **kwargs):
        start = time.monotonic()
        returnVal = func(*args, **kwargs)
        end = time.monotonic()
        buffer.append(end - start)
        get_logger('UTILS').warning(f'Time: {np.average(buffer)}')
        return returnVal
    return wrapper


def averageAbsoluteError(func):
    buffer = []
    realPoint = (0.463, 0.158)

    def wrapper(*args, **kwargs):
        returnVal = func(*args, **kwargs)
        calcPoint = (returnVal[0], returnVal[1])
        buffer.append(CalcLength(realPoint, calcPoint))
        get_logger('UTILS').warning(
            f'Error: {np.average(buffer)}')
        return returnVal
    return wrapper
