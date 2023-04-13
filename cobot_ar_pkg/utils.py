import math
import cv2
import numpy as np
from abc import ABC, abstractmethod
import typing
import time


class NoDetectionException(Exception):
    """ Raise, if the detector can not find the target object in the image """


class Detector(ABC):
    @abstractmethod
    def Detect(self, image) -> typing.Any:
        pass


def Rotate(theta: float, point):
    return (
        point[0] * math.cos(theta) - point[1] * math.sin(theta),
        point[0] * math.sin(theta) + point[1] * math.cos(theta)
    )


def Mask(image, points):
    mask = np.zeros(image.shape[: 2], dtype='uint8')
    cv2.drawContours(
        mask, [np.array(points)], -1,
        (255, 255, 255), -1, cv2.LINE_4
    )
    return cv2.bitwise_and(image, image, mask=mask)


def GetTimestamp() -> int:
    return round(time.time() * 1000)


def CalcAngle(p1, p2, p3):
    a = math.hypot(p2[0]-p3[0], p2[1]-p3[1])
    b = math.hypot(p1[0]-p3[0], p1[1]-p3[1])
    c = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
    try:
        return math.acos((b**2-a**2-c**2)/(-2*a*c))
    except ZeroDivisionError:
        return 0.0
