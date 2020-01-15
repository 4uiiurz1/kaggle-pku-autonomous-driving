import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import yaml
import gc

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import joblib
import cv2

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from skimage.io import imread

from apex import amp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.backends.cudnn as cudnn
import torchvision

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf, KeypointParams
from albumentations.pytorch.transforms import ToTensor
from albumentations.core.transforms_interface import NoOp

from lib.datasets import Dataset
from lib.utils.utils import *
from lib.models.model_factory import get_model
from lib.optimizers import RAdam
from lib import losses
from lib.decodes import decode
from lib.utils.image import get_bbox


def main():
    df = pd.read_csv('inputs/train.csv')
    img_ids = df['ImageId'].values
    img_paths = np.array('inputs/train_images/' + df['ImageId'].values + '.jpg')
    labels = np.array([convert_str_to_labels(s) for s in df['PredictionString']])

    pose_df = {
        'ImageId': [],
        'img_path': [],
        'yaw': [],
        'pitch': [],
        'roll': [],
    }

    output_dir = 'processed/pose_images/train'
    os.makedirs(output_dir, exist_ok=True)

    for img_id, img_path, label in tqdm(zip(img_ids, img_paths, labels), total=len(img_ids)):
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        kpts = []
        poses = []
        for k in range(len(label)):
            ann = label[k]
            kpts.append([ann['x'], ann['y'], ann['z']])
            poses.append([ann['yaw'], ann['pitch'], ann['roll']])
        kpts = np.array(kpts)
        poses = np.array(poses)

        kpts = np.array(convert_3d_to_2d(kpts[:, 0], kpts[:, 1], kpts[:, 2])).T

        for k, ((x, y), (yaw, pitch, roll)) in enumerate(zip(kpts, poses)):
            label[k]['x'] = x
            label[k]['y'] = y
            label[k]['yaw'] = yaw
            label[k]['pitch'] = pitch
            label[k]['roll'] = roll

        for k in range(len(label)):
            ann = label[k]
            x, y = ann['x'], ann['y']

            bbox = get_bbox(
                ann['yaw'],
                ann['pitch'],
                ann['roll'],
                *convert_2d_to_3d(ann['x'], ann['y'], ann['z']),
                ann['z'],
                width,
                height,
                width,
                height,
                car_hw=1.21,
                car_hh=0.95,
                car_hl=2.80)
            bbox = np.round(bbox).astype('int')
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)

            output_path = '%s_%d.jpg' %(img_id, k)

            cv2.imwrite(
                os.path.join(output_dir, output_path),
                img[bbox[1]:bbox[3], bbox[0]:bbox[2]])

            pose_df['ImageId'].append(img_id)
            pose_df['img_path'].append(output_path)
            pose_df['yaw'].append(ann['yaw'])
            pose_df['pitch'].append(ann['pitch'])
            pose_df['roll'].append(ann['roll'])

    pose_df = pd.DataFrame(pose_df)
    pose_df.to_csv('processed/pose_train.csv', index=False)


if __name__ == '__main__':
    main()
