import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import joblib
import cv2
import yaml

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

from lib.datasets import Dataset
from lib.utils.utils import *
from lib.models import resnet_fpn
from lib.optimizers import RAdam
from lib import losses
from lib.decodes import decode


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None)
    parser.add_argument('--score_thr', default=0.3, type=float)

    args = parser.parse_args()

    return args


def main():
    test_args = parse_args()

    with open('models/%s/config.yaml' % test_args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    df = pd.read_csv('inputs/train.csv')
    img_paths = np.array('inputs/train_images/' + df['ImageId'].values + '.jpg')
    mask_paths = np.array('inputs/train_masks/' + df['ImageId'].values + '.jpg')
    labels = np.array([convert_str_to_labels(s) for s in df['PredictionString']])

    heads = {
        'hm': 1,
        'reg': 2,
        'depth': 1,
    }

    criterion = {
        'hm': losses.__dict__[config['hm_loss']](),
        'reg': losses.__dict__[config['reg_loss']](),
        'depth': losses.__dict__[config['depth_loss']](),
    }

    if config['rot'] == 'eular':
        heads['eular'] = 3
        criterion['eular'] = losses.__dict__[config['eular_loss']]()
    elif config['rot'] == 'trig':
        heads['trig'] = 6
        criterion['trig'] = losses.__dict__[config['trig_loss']]()
    elif config['rot'] == 'quat':
        heads['quat'] = 4
        criterion['quat'] = losses.__dict__[config['quat_loss']]()
    else:
        raise NotImplementedError

    for head in criterion.keys():
        criterion[head] = criterion[key].cuda()

    folds = []
    best_losses = []
    # best_scores = []

    kf = KFold(n_splits=config['n_splits'], shuffle=True, random_state=41)
    for fold, (train_idx, val_idx) in enumerate(kf.split(img_paths)):
        print('Fold [%d/%d]' %(fold + 1, config['n_splits']))

        train_img_paths, val_img_paths = img_paths[train_idx], img_paths[val_idx]
        train_mask_paths, val_mask_paths = mask_paths[train_idx], mask_paths[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        train_transform = None
        val_transform = None

        # train
        train_set = Dataset(
            train_img_paths,
            train_mask_paths,
            train_labels,
            transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['batch_size'],
            # pin_memory=True,
        )

        val_set = Dataset(
            val_img_paths,
            val_mask_paths,
            val_labels,
            transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            num_workers=config['batch_size'],
            # pin_memory=True,
        )

        model = resnet_fpn.ResNetFPN(backbone='resnet18', heads=heads)
        model = model.cuda()

        model_path = 'models/%s/model_%d.pth' % (config['name'], fold+1)
        if not os.path.exists(model_path):
            print('%s is not exists.' %model_path)
            continue
        model.load_state_dict(torch.load(model_path))

        model.eval()

        with torch.no_grad():
            pbar = tqdm(total=len(val_loader))
            for i, batch in enumerate(val_loader):
                input = batch['input'].cuda()
                mask = batch['mask'].cuda()
                hm = batch['hm'].cuda()
                reg_mask = batch['reg_mask'].cuda()
                reg = batch['reg'].cuda()
                depth = batch['depth'].cuda()
                # eular = batch['eular'].cuda()
                quat = batch['quat'].cuda()

                output = model(input)

                if config['rot'] == 'eular':
                    dets = decode(output['hm'], output['reg'], output['depth'], eular=output['eular'])
                elif config['rot'] == 'trig':
                    dets = decode(output['hm'], output['reg'], output['depth'], trig=output['trig'])
                elif config['rot'] == 'quat':
                    dets = decode(output['hm'], output['reg'], output['depth'], quat=output['quat'])
                dets = dets.detach().cpu().numpy()[0]
                dets = dets[dets[:, 0] > 0.3]

                gt = batch['gt'].numpy()[0]
                gt = gt[gt[:, 0] > test_args.score_thr]

                img = cv2.imread(batch['img_path'][0])
                img_gt = visualize(img, gt)
                img_pred = visualize(img, dets)

                plt.subplot(121)
                plt.imshow(img_gt)
                plt.subplot(122)
                plt.imshow(img_pred)
                plt.show()

                # pbar.set_description('loss %.4f' %losses.avg)
                # pbar.set_description('loss %.4f - score %.4f' %(losses.avg, scores.avg))
                pbar.update(1)
            pbar.close()

        print('val_loss:  %f' % best_loss)
        # print('val_score: %f' % best_score)

        folds.append(str(fold + 1))
        best_losses.append(best_loss)
        # best_scores.append(best_score)

        torch.cuda.empty_cache()

        break
        if not args.cv:
            break


from math import sin, cos

# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
#         if p_x > image.shape[1] or p_y > image.shape[0]:
#             print('Point', p_x, p_y, 'is out of image with shape', image.shape)
    return image


def visualize(img, coords):
    # You will also need functions from the previous cells
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31

    camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

    img = img.copy()
    for point in coords:
        # Get values
        _, yaw, pitch, roll, x, y, z = point
        yaw = -yaw
        pitch = -pitch
        roll = -roll
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])

    return img


if __name__ == '__main__':
    main()
