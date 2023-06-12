import math
import cv2
import time
import os
import glob
import numpy as np
import json

saveConfPrefix = ''
saveConfPostfix = ''
saveImagesDir = ''


def SaveDirs(confPrefix, confPostfix, imagesDir):
    ''' Сохранение директорий в глобальных переменных '''
    global saveConfPostfix, saveConfPrefix, saveImagesDir
    saveConfPostfix = confPostfix
    saveConfPrefix = confPrefix
    saveImagesDir = imagesDir


def SaveConfig(data):
    ''' Сохранение конфигурации '''
    with open(saveConfPrefix + 'calibration_data' + saveConfPostfix + '.json', 'w') as f:
        json.dump(data, f)


def MakeShot(frame):
    ''' Сохрание кадра. '''
    if not (os.path.exists(saveImagesDir)):
        os.mkdir(saveImagesDir)
    timestamp = time.strftime('%H.%M.%S', time.gmtime(time.time()))
    name = f'{saveImagesDir}/image_{timestamp}.png'
    cv2.imwrite(name, frame)
    return name


def ChessCalibration(N=9, M=6):
    '''
        Калибровка по шахматному рисунку.
        Возвращает матрицу камеры, матрицу камеры с учетом дисторсии и коэфициенты дистории.
    '''
    images = glob.glob(f'{saveImagesDir}/*.png')
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((N*M, 3), np.float32)
    objp[:, :2] = np.mgrid[0:N, 0:M].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    validImgs = 0
    for fimage in images:
        img = cv2.imread(fimage)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (N, M), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            validImgs += 1
            cv2.drawChessboardCorners(img, (N, M), corners2, ret)
            cv2.imshow('calibration', img)
            cv2.waitKey(250)
    cv2.destroyWindow('calibration')
    if len(objpoints) == 0:
        return
    _, cameraMtx, distCoeff, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    h, w = gray.shape[:2]
    undistorCameraMtx, roi = cv2.getOptimalNewCameraMatrix(
        cameraMtx, distCoeff, (w, h), 1, (w, h))

    return cameraMtx, distCoeff, undistorCameraMtx, np.array(roi), validImgs


def DrawCenter(cameraMatrix, frame):
    '''
        Обозначить центр изображения по внутренней матрици камеры.
        Возвращает координаты центра в плоскости изображения и в глобальных координатах.
    '''
    imageCenter = [round(cameraMatrix[0, 2]), round(cameraMatrix[1, 2])]
    cv2.circle(frame, imageCenter, radius=5,
               color=(0, 0, 255), thickness=-1)
    cv2.imshow('Center', frame)
    c = cv2.waitKey(2)
    if c & 0xFF != ord(' '):
        return
    cv2.destroyWindow('Center')
    return imageCenter


def BlobDetect(frame) -> list:
    ''' Обнаружение точке. Возвращает список координат точке. '''
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
    params.thresholdStep = 10
    params.minDistBetweenBlobs = 10
    # # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.55
    blobDetector = cv2.SimpleBlobDetector_create(params)

    keypoints = blobDetector.detect(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    )

    frame = cv2.drawKeypoints(
        frame, keypoints, np.array([]), (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imshow('blob', frame)
    cv2.waitKey(100)
    # cv2.destroyWindow('blob')
    kps = [point.pt for point in keypoints]

    return kps


def CalcWorldZ(center, point):
    ''' Пересчет координаты Z для точки. Возвращает Z координату в глобальной системе отсчета. '''
    wd = math.hypot(point[0] - center[0], point[1] - center[1])
    cosA = (
        math.pow(point[2], 2) + math.pow(center[2], 2) - math.pow(wd, 2)
    ) / (
        2 * point[2] * center[2]
    )
    wz = point[2] * cosA
    return wz


def InputKpWorldCoords(kps, center, frame):
    ''' Ввод рельных координат для каждой переданной точки. Возвращает список координат точке в глобальной системе отсчета. '''
    worldKps = []
    for kp in kps:
        cv2.circle(frame, (round(kp[0]), round(kp[1])), radius=5,
                   color=(0, 0, 255), thickness=-1)
        cv2.imshow('Input', frame)
        cv2.waitKey(200)
        x, y, z = [
            float(i) for i in input(f'Add "x y z\'" for point {kp[0]} : {kp[1]}: ').split(' ')
        ]
        worldKps.append([x, y, CalcWorldZ(center, (x, y, z))])
    cv2.destroyWindow('Input')
    return worldKps


def ProjectiveCalibration(worldPts, imagePts, cameraMtx, distCoeff):
    ''' Калибровка камеры для получения вектора поворота и смещения. Возвращает вектор поворота, смещения и коэфициент маштабирования. '''
    worldPts = np.array(worldPts, dtype=np.float32)
    imagePts = np.array(imagePts, dtype=np.float32)
    _, rvec, tvec = cv2.solvePnP(
        worldPts, imagePts, cameraMtx, distCoeff)
    rmtx, _ = cv2.Rodrigues(rvec)
    Rt = np.column_stack((rmtx, tvec))
    Pmtx = cameraMtx.dot(Rt)
    scales = []
    for i in range(len(worldPts)):
        XYZ1 = np.array(
            [[worldPts[i, 0], worldPts[i, 1], worldPts[i, 2], 1]], dtype=np.float32).T
        sUV1 = Pmtx.dot(XYZ1)
        scale = sUV1[2, 0]
        UV1 = sUV1 / scale
        print(
            f'Scale: {scale} Error: {imagePts[i, 0] - UV1[0]} - {imagePts[i, 1] - UV1[1]}'
        )
        scales.append(scale)
    meanScale = np.mean(scales)

    return rvec, tvec, meanScale


def FindBlobs(frame):
    ''' Поиск всех точке и расчет их координат. '''
    from numpy.linalg import inv
    with open(saveConfPrefix + 'calibration_data' + saveConfPostfix + '.json') as f:
        data = json.load(f)
        cameraMtx = np.array(data['new_camera_matrix'])
        R, _ = cv2.Rodrigues(np.array(data['rvec']))
        t = np.array(data['tvec'])
        s = float(data['scale'])
    kps = BlobDetect(frame)
    blobs = []
    for kp in kps:
        uv1 = np.array([[kp[0], kp[1], 1]], dtype=np.float32).T
        suv1 = s * uv1
        xyzC = inv(cameraMtx).dot(suv1)
        xyzW = inv(R).dot(xyzC - t)
        blobs.append(xyzW)
    return blobs


def RuntimeShow(frame, withCalibration=False, N=9, M=6):
    ''' Отображение кадров. '''
    if withCalibration:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (N, M), None)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(gray, (N, M), corners2, ret)
            cv2.imshow('camera_calib', gray)
    cv2.imshow('camera', frame)
    return cv2.waitKey(2)


def Undistort(frame, cameraMtx, distCoef, newCamMtx, roi):
    ''' Компенсация дисторсии и образание кадра '''
    # undistort
    dst = cv2.undistort(frame, cameraMtx, distCoef,
                        None, newCamMtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst
