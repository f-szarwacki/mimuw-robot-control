# Franciszek Szarwacki

import cv2
import pickle
import numpy as np

MARKER_SIDE = 0.015
SQUARE_LENGTH = 0.026

with open("calibration.pckl", "rb") as f:
	cameraMatrix, distCoeffs, _, _ = pickle.load(f)


img1 = cv2.imread("frame-002.png")
img2 = cv2.imread("frame-253.png")

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
corners, ids, _ = cv2.aruco.detectMarkers(img1, dictionary)

# a)
img0_draw = img1.copy()
cv2.aruco.drawDetectedMarkers(img0_draw, corners, ids)
cv2.imshow('part1', img0_draw)
cv2.waitKey()

# b)
rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIDE, cameraMatrix, distCoeffs)

img1_draw = img1.copy()
for rvec, tvec in zip(rvecs, tvecs):
	cv2.drawFrameAxes(img1_draw, cameraMatrix, distCoeffs, rvec, tvec, 0.005)
cv2.imshow('part2', img1_draw)
cv2.waitKey()

# c)
rvec_1 = rvecs[np.where(ids == 1)]
tvec_1 = tvecs[np.where(ids == 1)]
print(f"Part 3: Pose of marker 1: tvec={tvec_1}, rvec={rvec_1}")

# d)
tvec_2 = np.array([[0.01428826],
    [0.02174878],
    [0.37597986]])
rvec_2 = np.array([[ 1.576368  ],
    [-1.03584672],
    [ 0.89579336]])

img2_draw = img2.copy()
cv2.drawFrameAxes(img2_draw, cameraMatrix, distCoeffs, rvec_2, tvec_2, 0.005)
cv2.imshow('part4', img2_draw)
cv2.waitKey()