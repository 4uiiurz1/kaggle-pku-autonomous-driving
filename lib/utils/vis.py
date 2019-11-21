import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .utils import convert_3d_to_2d


def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                  [0, 1, 0],
                  [-math.sin(yaw), 0, math.cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, math.cos(pitch), -math.sin(pitch)],
                  [0, math.sin(pitch), math.cos(pitch)]])
    R = np.array([[math.cos(roll), -math.sin(roll), 0],
                  [math.sin(roll), math.cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


def visualize(img, labels, scale_w=1, scale_h=1):
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31

    img = img.copy()
    for yaw, pitch, roll, x, y, z, _ in labels:
        yaw *= -1
        pitch *= -1
        roll *= -1

        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]

        P = np.array([
            [x_l,  -y_l, -z_l, 1],
            [x_l,   y_l, -z_l, 1],
            [x_l,  -y_l, -z_l, 1],

            [x_l,  -y_l,  z_l, 1],
            [x_l,   y_l,  z_l, 1],
            [x_l,  -y_l,  z_l, 1],

            [-x_l, -y_l,  z_l, 1],
            [-x_l,  y_l,  z_l, 1],
            [-x_l, -y_l,  z_l, 1],

            [-x_l, -y_l, -z_l, 1],
            [-x_l,  y_l, -z_l, 1],
            [-x_l, -y_l, -z_l, 1],

            [x_l,  -y_l, -z_l, 1],

            [x_l,   y_l, -z_l, 1],
            [-x_l,  y_l, -z_l, 1],
            [-x_l,  y_l,  z_l, 1],
            [ x_l,  y_l,  z_l, 1],
            [x_l,   y_l, -z_l, 1],

            [0,       0,    0, 1]]).T
        P = Rt @  P
        P = P.T

        xs, ys = convert_3d_to_2d(P[:, 0], P[:, 1], P[:, 2])
        xs *= scale_w
        ys *= scale_h
        pts = np.hstack((xs[:, None], ys[:, None])).astype('int')
        pts = pts.reshape((-1, 1, 2))

        cv2.polylines(img, [pts[:-1]], False, (44, 160, 44), 4)
        cv2.circle(img, tuple(pts[-1, 0]), int(1000 / P[-1, 2]), (180, 119, 31), -1)

    return img
