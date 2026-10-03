import cv2
import numpy as np

# 多距离标定参数：各测量点（距离 × 视差）的中位数，单位 cm·px
K = 3650.0

LEFT_INDEX = 0
RIGHT_INDEX = 2
LOWER_HSV = np.array([5, 0, 60], dtype=np.uint8)
UPPER_HSV = np.array([50, 90, 255], dtype=np.uint8)
MORPH_KERNEL = np.ones((9, 9), np.uint8)

cameras = [
    cv2.VideoCapture(index, cv2.CAP_DSHOW)
    for index in (LEFT_INDEX, RIGHT_INDEX)
]


def find_center(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if cv2.contourArea(contour) < 100 or moments["m00"] == 0:
        return None

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    return center_x, center_y, contour


try:
    if not all(camera.isOpened() for camera in cameras):
        raise RuntimeError("摄像头打开失败")

    for _ in range(12):
        for camera in cameras:
            camera.read()

    while True:
        frames = []
        for camera in cameras:
            ok, frame = camera.read()
            if not ok:
                break
            frames.append(frame)
        if len(frames) != len(cameras):
            break

        detections = [find_center(frame) for frame in frames]

        if all(detections):
            (cxL, cyL, contourL), (cxR, cyR, contourR) = detections
            disparity = abs(cxL - cxR)
            distance = K / disparity if disparity else 0.0

            for frame, (center_x, center_y, contour) in zip(frames, detections):
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

            cv2.putText(frames[0], f"d = {disparity} px", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if disparity != 0:
                cv2.putText(frames[0], f"Z = {distance:.1f} cm", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            print(f"L:({cxL},{cyL})  R:({cxR},{cyR})  d = {disparity} px  Z = {distance:.1f} cm")

        for name, frame in zip(("Left Camera", "Right Camera"), frames):
            cv2.imshow(name, frame)

        if cv2.waitKey(1) == 27:
            break
finally:
    for camera in cameras:
        camera.release()
    cv2.destroyAllWindows()