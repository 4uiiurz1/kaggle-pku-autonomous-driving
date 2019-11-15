import random
import math
from PIL import Image
import numpy as np

import torch


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def convert_str_to_labels(s):
    names = ['model_type', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']

    labels = []
    for l in np.array(s.split()).reshape([-1, 7]):
        labels.append(dict(zip(names, l.astype('float'))))
        if 'model_type' in labels[-1]:
            labels[-1]['model_type'] = int(labels[-1]['model_type'])

    return labels


def convert_euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - \
         np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + \
         np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - \
         np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + \
         np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return qx, qy, qz, qw


def convert_quaternion_to_euler(qx, qy, qz, qw):
    t0 = 2.0 * (qw * qx + qy * qz)
    t1 = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (qw * qy - qz * qx)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = 2.0 * (qw * qz + qx * qy)
    t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(t3, t4)

    return yaw, pitch, roll


def convert_3d_to_2d(x, y, z, fx=2304.5479, fy=2305.8757, cx=1686.2379, cy=1354.9849):
    x = fx * x / z + cx
    y = fy * y / z + cy

    return x, y


def convert_2d_to_3d(x, y, z, fx=2304.5479, fy=2305.8757, cx=1686.2379, cy=1354.9849):
    x = (x - cx) / fx * z
    y = (y - cy) / fy * z

    return x, y


def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi

    return x
