import math
from rclpy import logging

import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks.python.vision import HandLandmarkerResult
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

import cv2

import numpy as np
from numpy.linalg import inv
import json

from cobot_ar_pkg import utils

NoneType = type(None)


class PointTransformer:

    def __init__(self, configDataPath) -> None:
        self.__initTransform(configDataPath)

    def __initTransform(self, configDataPath):
        with open(configDataPath) as f:
            data = json.load(f)
            self.cameraMtx = np.array(data['undistor_camera_matrix'])
            self.R, _ = cv2.Rodrigues(np.array(data['rvec']))
            self.t = np.array(data['tvec'])
            self.s = float(data['scale'])

    def Transform(self, point, mtx):
        npPoint = np.array([[[point[0], point[1]]]], dtype=np.float32)
        newPoint = cv2.perspectiveTransform(npPoint, mtx)

        uv1 = np.array([[newPoint[0][0][0], newPoint[0][0][1], 1]],
                       dtype=np.float32).T
        suv1 = self.s * uv1
        xyzC = inv(self.cameraMtx).dot(suv1)
        xyzW = inv(self.R).dot(xyzC - self.t)
        return xyzW


class MatchDetectorSIFT:
    COLOR_ANNOTATED = (0, 0, 255)

    def __init__(self) -> None:
        self.__initSIFT()

    def __initSIFT(self):
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def Detect(self, imageQuery, imageTrain):
        queryKps, queryDesc = self.sift.detectAndCompute(imageQuery, None)
        trainKps, trainDesc = self.sift.detectAndCompute(imageTrain, None)
        if type(queryDesc) == NoneType or type(trainDesc) == NoneType:
            raise utils.NoDetectionException("Images dont have descripters.")
        matches = self.flann.knnMatch(queryDesc, trainDesc, k=2)
        # matches = sorted(matches, key=lambda x: x.distance)
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good) < 10:
            raise utils.NoDetectionException(
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


class MatchDetectorORB:
    COLOR_ANNOTATED = (0, 0, 255)

    def __init__(self) -> None:
        self.__initORB()

    def __initORB(self):
        self.orb = cv2.BRISK_create()
        self.brMatcher = cv2.BFMatcher_create(
            cv2.NORM_HAMMING, crossCheck=True)

    def Detect(self, imageQuery, imageTrain):
        queryKps, queryDesc = self.orb.detectAndCompute(imageQuery, None)
        trainKps, trainDesc = self.orb.detectAndCompute(imageTrain, None)
        if type(queryDesc) == NoneType or type(trainDesc) == NoneType:
            raise utils.NoDetectionException("Images dont have descripters.")
        matches = self.brMatcher.match(queryDesc, trainDesc)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 10:
            raise utils.NoDetectionException(
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


class NearestBlobDetector():
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

    def __findNearestPoint(self, pt0, keypoints):
        def getLength(point): return utils.CalcLength(pt0, point)
        idxNearestPoint = 0
        minDist = getLength(keypoints[0].pt)
        for idx in range(1, len(keypoints)):
            if (dist := getLength(keypoints[idx].pt)) < minDist:
                minDist = dist
                idxNearestPoint = idx
        return idxNearestPoint

    def Detect(self, image, point):
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
            raise utils.NoDetectionException(
                "NearestBlobDetector dont detect any blobs"
            )

        nearestKeypoint = keypoints[self.__findNearestPoint(point, keypoints)]
        imgDetection = cv2.drawKeypoints(
            imgDetection, [nearestKeypoint], np.array([]), (0, 255, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return imgDetection, nearestKeypoint.pt
        # return imgDetection, keypoints[0].pt


class IndexHandDetector():
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
        self.__shape = [0, 0]

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

    def __convertLandmarkToPoint(self, landmark):
        return (int(landmark.x * self.__shape[1]), int(landmark.y * self.__shape[0]))

    def __checkIndexGesture(self, handLanmarksList):
        wrist = self.__convertLandmarkToPoint(
            handLanmarksList[solutions.hands.HandLandmark.WRIST]
        )
        finger = (
            self.__convertLandmarkToPoint(
                handLanmarksList[5]
            ),
            self.__convertLandmarkToPoint(
                handLanmarksList[8]
            )
        )
        angle = utils.CalcAngle(wrist, finger[0], finger[1])
        if angle < math.pi * 0.8:
            return False
        for fingerIdx in range(2, 5):
            finger = (
                self.__convertLandmarkToPoint(
                    handLanmarksList[4 * fingerIdx + 1]
                ),
                self.__convertLandmarkToPoint(
                    handLanmarksList[4 * fingerIdx + 4]
                )
            )
            angle = utils.CalcAngle(wrist, finger[0], finger[1])
            if angle > (math.pi * 0.8):
                return False
        return True

    def Detect(self, image):
        # image = cv2.flip(image, 1)
        self.__shape = image.shape[:2]
        # cv2.imshow('input', image)
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
        if not self.__checkIndexGesture(handLandmarkerResult.hand_landmarks[0]):
            raise utils.NoDetectionException(
                "IndexHandDetector dont detect index guest"
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


class BlobDetector():
    X_MARGIN = 50
    Y_MARGIN = 50
    WINDOW = (
        (-X_MARGIN / 2, 0),
        (-X_MARGIN / 2, Y_MARGIN),
        (X_MARGIN / 2, Y_MARGIN),
        (X_MARGIN / 2, 0),
    )

    def __init__(self) -> None:
        self.__blobDetector = NearestBlobDetector()
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
        imgHandDetect, [tip, dip] = self.__indexHandDetector.Detect(
            image
        )
        cv2.imshow('hand', imgHandDetect)
        points = self.__calculateWindowPoints(tip, dip)
        imgMasked = utils.Mask(image, points)
        cv2.imshow('masked', imgMasked)
        imgDetection, nearestBlob = self.__blobDetector.Detect(imgMasked, tip)

        alpha = np.uint8((np.sum(imgDetection, axis=-1) > 0) * 255)
        infoImage = np.dstack((imgDetection, alpha))

        return infoImage, nearestBlob
