from abc import ABC, abstractmethod
import typing
import cv2
from cv2 import aruco
import numpy as np


class Detector(ABC):
    @abstractmethod
    def Detect(self, image) -> typing.Any:
        pass


class BlobDetector(Detector):
    def __init__(self) -> None:
        self.InitBlob()
        self.InitAruco(aruco.DICT_6X6_50)

    def InitBlob(self):
        self.blobParams = cv2.SimpleBlobDetector_Params()
        self.blobParams.filterByArea = True
        self.blobParams.minArea = 5
        self.blobParams.filterByCircularity = True
        self.blobParams.minCircularity = 0.5
        self.blobParams.minDistBetweenBlobs = 5
        self.blobDetector = cv2.SimpleBlobDetector_create(self.blobParams)

    def InitAruco(self, defDict):
        aruco_dict = aruco.getPredefinedDictionary(defDict)
        parameters = aruco.DetectorParameters()
        self.arucoDetector = aruco.ArucoDetector(aruco_dict, parameters)

    def DetectAruco(self, image) -> list:
        corners, ids, rejected = self.arucoDetector.detectMarkers(image)
        return corners

    def FindRectangle(self, arucoPoints):
        a = []
        b = []
        if len(arucoPoints) != 0:
            for bbox in arucoPoints:
                if np.array(arucoPoints).shape[0] == 4:
                    bbox = bbox[0]
                for i in bbox:
                    a.append(i[0])
                    b.append(i[1])
        a1 = int(min(a))
        a2 = int(max(a))
        b1 = int(min(b))
        b2 = int(max(b))
        return (a1, b1), (a2, b2)

    def Mask(self, image, pt1, pt2):
        mask = np.zeros(image.shape[:2], dtype='uint8')
        cv2.rectangle(mask, pt1, pt2, 255, -1)
        return cv2.bitwise_and(image, image, mask=mask)

    def Detect(self, image) -> typing.Any:
        arucofound = self.DetectAruco(image)
        if arucofound.__len__() != 4:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        point1, point2 = self.FindRectangle(arucofound)
        gray = self.Mask(gray, point1, point2)
        im = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
        keypoints = self.blobDetector.detect(im)
        res = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
        im_with_keypoints = cv2.drawKeypoints(res, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return im_with_keypoints


class ArucoDetector(Detector):
    def __init__(self) -> None:
        self.InitAruco(aruco.DICT_6X6_50)

    def InitAruco(self, defDict):
        aruco_dict = aruco.getPredefinedDictionary(defDict)
        parameters = aruco.DetectorParameters()
        self.arucoDetector = aruco.ArucoDetector(aruco_dict, parameters)

    def Detect(self, image) -> typing.Any:
        corners, ids, rejectedCandidates = self.detectorAruco.detectMarkers(
            image)
        res = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        if ids is not None:
            self.get_logger().info('Aruco marker detected')
            aruco.drawDetectedMarkers(res, corners)
        alpha = np.uint8((np.sum(res, axis=-1) > 0) * 255)
        infoImage = np.dstack((res, alpha))
        return infoImage
