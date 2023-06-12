import cv2
import numpy as np
from cobot_ar_pkg.utils.utils import NoneType, NoDetectionException


class FeatureDetectorSIFT:
    ''' Класс сопоставления изображений на основе SIFT алгоритма. '''
    COLOR_ANNOTATED = (0, 0, 255)

    def __init__(self) -> None:
        self.__initSIFT()

    def __initSIFT(self):
        ''' Инициализация детектора и мэтчера. '''
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def Detect(self, imageQuery, imageTrain):
        '''
        Сопоставление изображения imageTrain в изображении imageQuery. Возвращает изображение, на котором обозначен область сопоставляемого изображения в исходном, и матрица перспективного преобразования из train в query.

        Raises
        -----
        NoDetectionException
            Если ключевые точки / дескрипторы не обнаружены или сопоставлено слишком мало ключевых точек.
        '''
        queryKps, queryDesc = self.sift.detectAndCompute(imageQuery, None)
        trainKps, trainDesc = self.sift.detectAndCompute(imageTrain, None)
        if type(queryDesc) == NoneType or type(trainDesc) == NoneType:
            raise NoDetectionException("Images dont have descripters.")
        matches = self.flann.knnMatch(queryDesc, trainDesc, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) < 10:
            raise NoDetectionException(
                'PictureInPictureDetector does not find the required number of key points'
            )
        queryPts = np.float32(
            [queryKps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        trainPts = np.float32(
            [trainKps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        retv, _ = cv2.findHomography(queryPts, trainPts, cv2.RANSAC)
        h, w = imageQuery.shape[:2]
        pts = np.float32([
            [0, 0], [0, h-1], [w-1, h-1],
            [w-1, 0]
        ]).reshape(-1, 1, 2)
        dst = np.int32(cv2.perspectiveTransform(pts, retv)).reshape((4, 2))
        imageAnnotated = cv2.polylines(
            imageTrain, [dst], True, self.COLOR_ANNOTATED, lineType=cv2.LINE_AA)
        return imageAnnotated, retv


class FeatureDetectorBRISK:
    ''' Класс сопоставления изображений на основе BRISK алгоритма. '''
    COLOR_ANNOTATED = (0, 0, 255)

    def __init__(self) -> None:
        self.__initORB()

    def __initORB(self):
        ''' Инициализация детектора и мэтчера. '''
        self.brisk = cv2.BRISK_create()
        self.bfMatcher = cv2.BFMatcher_create(
            cv2.NORM_HAMMING, crossCheck=True)

    def Detect(self, imageQuery, imageTrain):
        '''
        Сопоставление изображения imageTrain в изображении imageQuery. Возвращает изображение, на котором обозначен область сопоставляемого изображения в исходном, и матрица перспективного преобразования из train в query.

        Raises
        -----
        NoDetectionException
            Если ключевые точки / дескрипторы не обнаружены или сопоставлено слишком мало ключевых точек.
        '''
        queryKps, queryDesc = self.brisk.detectAndCompute(imageQuery, None)
        trainKps, trainDesc = self.brisk.detectAndCompute(imageTrain, None)
        if type(queryDesc) == NoneType or type(trainDesc) == NoneType:
            raise NoDetectionException("Images dont have descripters.")
        matches = self.bfMatcher.match(queryDesc, trainDesc)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 10:
            raise NoDetectionException(
                'PictureInPictureDetector does not find the required number of key points'
            )
        queryPts = np.float32([
            queryKps[m.queryIdx].pt for m in matches
        ]).reshape(-1, 1, 2)
        trainPts = np.float32([
            trainKps[m.trainIdx].pt for m in matches
        ]).reshape(-1, 1, 2)

        retv, _ = cv2.findHomography(queryPts, trainPts, cv2.RANSAC)
        h, w = imageQuery.shape[:2]
        pts = np.float32([
            [0, 0], [0, h-1], [w-1, h-1],
            [w-1, 0]
        ]).reshape(-1, 1, 2)
        dst = np.int32(cv2.perspectiveTransform(pts, retv)).reshape((4, 2))
        imageAnnotated = cv2.polylines(
            imageTrain, [dst], True, self.COLOR_ANNOTATED, lineType=cv2.LINE_AA)
        return imageAnnotated, retv


class FeatureDetectorORB:
    ''' Класс сопоставления изображений на основе ORB алгоритма. '''
    COLOR_ANNOTATED = (0, 0, 255)

    def __init__(self) -> None:
        self.__initORB()

    def __initORB(self):
        ''' Инициализация детектора и мэтчера. '''
        self.orb = cv2.ORB_create()
        self.bfMatcher = cv2.BFMatcher_create(
            cv2.NORM_HAMMING, crossCheck=True)

    def Detect(self, imageQuery, imageTrain):
        '''
        Сопоставление изображения imageTrain в изображении imageQuery. Возвращает изображение, на котором обозначен область сопоставляемого изображения в исходном, и матрица перспективного преобразования из train в query.

        Raises
        -----
        NoDetectionException
            Если ключевые точки / дескрипторы не обнаружены или сопоставлено слишком мало ключевых точек.
        '''
        queryKps, queryDesc = self.orb.detectAndCompute(imageQuery, None)
        trainKps, trainDesc = self.orb.detectAndCompute(imageTrain, None)
        if type(queryDesc) == NoneType or type(trainDesc) == NoneType:
            raise NoDetectionException("Images dont have descripters.")
        matches = self.bfMatcher.match(queryDesc, trainDesc)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 10:
            raise NoDetectionException(
                'PictureInPictureDetector does not find the required number of key points'
            )
        queryPts = np.float32([
            queryKps[m.queryIdx].pt for m in matches
        ]).reshape(-1, 1, 2)
        trainPts = np.float32([
            trainKps[m.trainIdx].pt for m in matches
        ]).reshape(-1, 1, 2)

        retv, _ = cv2.findHomography(queryPts, trainPts, cv2.RANSAC)
        h, w = imageQuery.shape[:2]
        pts = np.float32([
            [0, 0], [0, h-1], [w-1, h-1],
            [w-1, 0]
        ]).reshape(-1, 1, 2)
        dst = np.int32(cv2.perspectiveTransform(pts, retv)).reshape((4, 2))
        imageAnnotated = cv2.polylines(
            imageTrain, [dst], True, self.COLOR_ANNOTATED, lineType=cv2.LINE_AA)
        return imageAnnotated, retv
