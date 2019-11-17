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


def convert_str_to_labels(s, names=['model_type', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    labels = []
    for l in np.array(s.split()).reshape([-1, 7]):
        labels.append(dict(zip(names, l.astype('float'))))
        if 'model_type' in labels[-1]:
            labels[-1]['model_type'] = int(labels[-1]['model_type'])

    return labels


def convert_labels_to_str(labels):
    s = []
    for label in labels:
        for l in label:
            s.append(str(l))
    return ' '.join(s)


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
