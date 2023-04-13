import math

import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks.python.vision import HandLandmarkerResult
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

import cv2
from cv2 import aruco

import numpy as np

from cobot_ar_pkg import utils


class SimpleBlobDetector(utils.Detector):
    def __init__(self) -> None:
        self.__initBlob()

    def __initBlob(self):
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

    def Detect(self, image):
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
        return imgDetection


class IndexHandDetector(utils.Detector):
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
    MODEL_PATH = 'install/cobot_ar_pkg/share/cobot_ar_pkg/config/hand_landmarker.task'

    def __init__(self) -> None:
        self.__initHandDetection()

    def __initHandDetection(self):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1
        )
        self.__landmarker = vision.HandLandmarker.create_from_options(options)

    def __drawHandednessName(self, image, hand_landmarks, handedness):
        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = image.shape
        xCoordinates = [landmark.x for landmark in hand_landmarks]
        yCoordinates = [landmark.y for landmark in hand_landmarks]
        textX = int(min(xCoordinates) * width)
        textY = int(min(yCoordinates) * height) - self.MARGIN
        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            image, f"{handedness[0].category_name}",
            (textX, textY), cv2.FONT_HERSHEY_DUPLEX,
            self.FONT_SIZE, self.HANDEDNESS_TEXT_COLOR,
            self.FONT_THICKNESS, cv2.LINE_AA
        )

    def __makeAnnotatedImage(self, image, detection_result: HandLandmarkerResult):
        imgAnnotated = np.copy(image)
        handLandmarksList = detection_result.hand_landmarks
        handednessList = detection_result.handedness

        # Loop through the detected hands to visualize.
        for idx in range(len(handLandmarksList)):
            handLandmarks = handLandmarksList[idx]
            handedness = handednessList[idx]
            # Draw the hand landmarks.
            handLandmarksProto = landmark_pb2.NormalizedLandmarkList()
            handLandmarksProto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in handLandmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                imgAnnotated,
                handLandmarksProto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style()
            )
            # Add left | right hint
            self.__drawHandednessName(
                imgAnnotated, handLandmarks, handedness
            )
        return imgAnnotated

    def Detect(self, image):
        # image = cv2.flip(image, 1)
        cv2.imshow('input', image)
        impMP = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )
        handLandmarkerResult = self.__landmarker.detect_for_video(
            impMP, utils.GetTimestamp())
        if len(handLandmarkerResult.hand_landmarks) == 0:
            raise utils.NoDetectionException(
                "IndexHandDetector dont detect any hand"
            )
        imgAnnotated = cv2.cvtColor(self.__makeAnnotatedImage(
            impMP.numpy_view(), handLandmarkerResult), cv2.COLOR_RGB2BGR)
        return imgAnnotated, (
            (
                int(
                    handLandmarkerResult.hand_landmarks[0]
                    [solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]
                ),
                int(
                    handLandmarkerResult.hand_landmarks[0]
                    [solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]
                )
            ),
            (
                int(
                    handLandmarkerResult.hand_landmarks[0]
                    [solutions.hands.HandLandmark.INDEX_FINGER_DIP].x * image.shape[1]
                ),
                int(
                    handLandmarkerResult.hand_landmarks[0]
                    [solutions.hands.HandLandmark.INDEX_FINGER_DIP].y * image.shape[0]
                )
            )
        ),


class BlobDetectorV2(utils.Detector):
    X_MARGIN = 50
    Y_MARGIN = 50
    WINDOW = (
        (-X_MARGIN / 2, 0),
        (-X_MARGIN / 2, Y_MARGIN),
        (X_MARGIN / 2, Y_MARGIN),
        (X_MARGIN / 2, 0),
    )

    def __init__(self) -> None:
        self.__blobDetector = SimpleBlobDetector()
        self.__indexHandDetector = IndexHandDetector()

    def __calculateWindowPoints(self, tip, dip):
        xBase = tip[0] - dip[0]
        yBase = tip[1] - dip[1]
        theta = math.atan2(yBase, xBase)
        if xBase < 0 < yBase:
            theta -= math.pi / 2
        else:
            theta += math.pi * 1.5
        points = []
        for point in self.WINDOW:
            pointRotated = utils.Rotate(theta, point)
            points.append([
                int(pointRotated[0] + tip[0]),
                int(pointRotated[1] + tip[1])
            ])
        return points

    def Detect(self, image):
        try:
            imgHandDetect, [tip, dip] = self.__indexHandDetector.Detect(
                image
            )
        except utils.NoDetectionException:
            return image
        points = self.__calculateWindowPoints(tip, dip)
        imgMasked = utils.Mask(image, points)
        imgDetection = self.__blobDetector.Detect(imgMasked)
        cv2.imshow('hand', imgHandDetect)
        cv2.imshow('masked', imgMasked)
        return imgDetection


class ArucoDetector(utils.Detector):
    ''' Obsolete '''

    def __init__(self) -> None:
        self.__initAruco(aruco.DICT_6X6_50)

    def __initAruco(self, defDict):
        aruco_dict = aruco.getPredefinedDictionary(defDict)
        parameters = aruco.DetectorParameters()
        self.__arucoDetector = aruco.ArucoDetector(aruco_dict, parameters)

    def Detect(self, image):
        corners, ids, rejectedCandidates = self.__arucoDetector.detectMarkers(
            image)
        res = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        if ids is not None:
            aruco.drawDetectedMarkers(res, corners)
        alpha = np.uint8((np.sum(res, axis=-1) > 0) * 255)
        infoImage = np.dstack((res, alpha))
        return infoImage


class BlobDetector(utils.Detector):
    ''' Obsolete '''

    def __init__(self) -> None:
        self.__initBlob()
        self.__initAruco(aruco.DICT_6X6_50)

    def __initBlob(self):
        blobParams = cv2.SimpleBlobDetector_Params()
        blobParams.filterByArea = True
        blobParams.minArea = 5
        blobParams.filterByCircularity = True
        blobParams.minCircularity = 0.5
        blobParams.minDistBetweenBlobs = 5
        self.__blobDetector = cv2.SimpleBlobDetector_create(blobParams)

    def __initAruco(self, defDict):
        aruco_dict = aruco.getPredefinedDictionary(defDict)
        parameters = aruco.DetectorParameters()
        self.__arucoDetector = aruco.ArucoDetector(aruco_dict, parameters)

    def __findRectangle(self, arucoPoints):
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

    def __mask(self, image, pt1, pt2):
        mask = np.zeros(image.shape[:2], dtype='uint8')
        cv2.rectangle(mask, pt1, pt2, 255, -1)
        return cv2.bitwise_and(image, image, mask=mask)

    def Detect(self, image):
        arucofound, ids, rejected = self.__arucoDetector.detectMarkers(image)
        if arucofound.__len__() != 4:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        point1, point2 = self.__findRectangle(arucofound)
        gray = self.__mask(gray, point1, point2)
        im = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
        keypoints = self.__blobDetector.detect(im)
        res = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
        im_with_keypoints = cv2.drawKeypoints(
            res, keypoints, np.array([]), (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return im_with_keypoints
