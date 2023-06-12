import cv2
import numpy as np
from cobot_ar_pkg.utils.utils import CalcLength, NoDetectionException


class NearestBlobDetector():
    ''' Класс обнаружения ближайшего отвертия. '''

    def __init__(self) -> None:
        self.__initBlob()

    def __initBlob(self):
        ''' Инициализация параметров детектора. '''
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
        params.thresholdStep = 10
        params.minDistBetweenBlobs = 10

        # Filter by Area.s
        # params.filterByArea = True
        # params.minArea = 5

        # # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8

        # Filter by Convexity
        # params.filterByConvexity = True
        # params.minConvexity = 1

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.55
        self.__blobDetector = cv2.SimpleBlobDetector_create(params)

    def __findNearestPoint(self, pt0, keypoints):
        ''' Поиск ближайшего отвертия к точке pt0. '''
        def getLength(point): return CalcLength(pt0, point)
        idxNearestPoint = 0
        minDist = getLength(keypoints[0].pt)
        for idx in range(1, len(keypoints)):
            if (dist := getLength(keypoints[idx].pt)) < minDist:
                minDist = dist
                idxNearestPoint = idx
        return idxNearestPoint

    def Detect(self, image, point):
        '''
        Обнаружение обнаружения ближайшего отвертия к точке на изобраении. Возвращает изображение, где выделены все обнаруженные отвретия (зеленным - ближайшее), и координаты ближайшей.

        Raises
        ------
        utils.NoDetectionException
            Если не обнаружено ни одно отвертие.
        '''
        keypoints = self.__blobDetector.detect(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )

        imgDetection = np.zeros(
            (image.shape[0], image.shape[1], 3), dtype=np.uint8
        )
        imgDetection = cv2.drawKeypoints(
            imgDetection, keypoints, np.array([]), (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        if len(keypoints) == 0:
            raise NoDetectionException(
                "NearestBlobDetector dont detect any blobs"
            )

        nearestKeypoint = keypoints[self.__findNearestPoint(point, keypoints)]
        imgDetection = cv2.drawKeypoints(
            imgDetection, [nearestKeypoint], np.array([]), (0, 255, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return imgDetection, nearestKeypoint.pt
