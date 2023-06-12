import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks.python.vision import HandLandmarkerResult
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

import cv2
import numpy as np
import math

from cobot_ar_pkg.utils.utils import CalcAngle, GetTimestamp, NoDetectionException


class IndexHandDetector():
    '''
    Класс обнаружения указательного жеста.
    '''
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
    MODEL_PATH = 'install/cobot_ar_pkg/share/cobot_ar_pkg/config/hand_landmarker.task'

    def __init__(self) -> None:
        self.__initHandDetection()

    def __initHandDetection(self):
        ''' Инициализация основных атрибутов '''
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1
        )
        self.__landmarker = vision.HandLandmarker.create_from_options(options)
        self.__shape = [0, 0]

    def __makeAnnotatedImage(self, image, detection_result: HandLandmarkerResult):
        ''' Добавление обнаруженных ключевых точек кисте на изображение. Возвращает новое изображение. '''
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
        return imgAnnotated

    def __convertLandmarkToPoint(self, landmark):
        ''' Денормализация координат точек '''
        return (int(landmark.x * self.__shape[1]), int(landmark.y * self.__shape[0]))

    def __checkIndexGesture(self, handLanmarksList):
        ''' Проверка, что обнаруженные ключевые точки кисти формируют указательный жест. '''
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
        angle = CalcAngle(wrist, finger[0], finger[1])
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
            angle = CalcAngle(wrist, finger[0], finger[1])
            if angle > (math.pi * 0.8):
                return False
        return True

    def Detect(self, image):
        '''
        Обнаружение указательного жеста в изображении. Возвращает изображение, где выделен обнаруженный жест, и точки конца и середины указательного пальца.

        Raises
        ------
        NoDetectionException
            Если кисть не обнаружена или жест неуказательный.
        '''
        self.__shape = image.shape[:2]
        impMP = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )
        handLandmarkerResult = self.__landmarker.detect_for_video(
            impMP, GetTimestamp())
        if len(handLandmarkerResult.hand_landmarks) == 0:
            raise NoDetectionException(
                "IndexHandDetector dont detect any hand"
            )
        if not self.__checkIndexGesture(handLandmarkerResult.hand_landmarks[0]):
            raise NoDetectionException(
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
        )
