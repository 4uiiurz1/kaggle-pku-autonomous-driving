import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import json

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

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf, KeypointParams
from albumentations.pytorch.transforms import ToTensor
from albumentations.core.transforms_interface import NoOp

from lib.datasets import PoseDataset
from lib.utils.utils import *
from lib.models.model_factory import get_pose_model
from lib.optimizers import RAdam
from lib import losses
from lib.decodes import decode
from lib.utils.vis import visualize
from lib.utils.nms import nms


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--det_name', default=None)
    parser.add_argument('--pose_name', default=None)
    parser.add_argument('--score_th', default=0.3, type=float)
    parser.add_argument('--nms', default=True, type=str2bool)
    parser.add_argument('--nms_th', default=0.1, type=float)
    parser.add_argument('--min_samples', default=1, type=int)
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/pose/%s/config.yml' % args.pose_name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    df = pd.read_csv('inputs/sample_submission.csv')
    img_ids = df['ImageId'].values
    img_paths = np.array('inputs/test_images/' + df['ImageId'].values + '.jpg')
    mask_paths = np.array('inputs/test_masks/' + df['ImageId'].values + '.jpg')
    labels = np.array([convert_str_to_labels(s, names=['yaw', 'pitch', 'roll',
                       'x', 'y', 'z', 'score']) for s in df['PredictionString']])
    with open('outputs/decoded/test/%s.json' %args.det_name, 'r') as f:
        dets = json.load(f)

    if config['rot'] == 'eular':
        num_outputs = 3
    elif config['rot'] == 'trig':
        num_outputs = 6
    elif config['rot'] == 'quat':
        num_outputs = 4
    else:
        raise NotImplementedError

    test_transform = Compose([
        transforms.Resize(config['input_w'], config['input_h']),
        transforms.Normalize(),
        ToTensor(),
    ])

    det_df = {
        'ImageId': [],
        'img_path': [],
        'det': [],
        'mask': [],
    }

    name = '%s_%.2f' %(args.det_name, args.score_th)
    if args.nms:
        name += '_nms%.2f' %args.nms_th
    if args.min_samples > 0:
        name += '_min%d' %args.min_samples

    output_dir = 'processed/pose_images/test/%s' % name
    os.makedirs(output_dir, exist_ok=True)
    for img_id, img_path in tqdm(zip(img_ids, img_paths), total=len(img_ids)):
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        det = np.array(dets[img_id])
        if np.sum(det[:, 6] > args.score_th) >= args.min_samples:
            det = det[det[:, 6] > args.score_th]
        else:
            det = det[:args.min_samples]
        if args.nms:
            det = nms(det, dist_th=args.nms_th)

        for k in range(len(det)):
            pitch, yaw, roll, x, y, z, score, w, h = det[k]

            det_df['ImageId'].append(img_id)
            det_df['det'].append(det[k])
            output_path = '%s_%d.jpg' %(img_id, k)
            det_df['img_path'].append(output_path)

            x, y = convert_3d_to_2d(x, y, z)
            w *= 1.1
            h *= 1.1
            xmin = int(round(x - w / 2))
            xmax = int(round(x + w / 2))
            ymin = int(round(y - h / 2))
            ymax = int(round(y + h / 2))

            cropped_img = img[ymin:ymax, xmin:xmax]
            if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                cv2.imwrite(os.path.join(output_dir, output_path), cropped_img)
                det_df['mask'].append(1)
            else:
                det_df['mask'].append(0)

    det_df = pd.DataFrame(det_df)

    test_set = PoseDataset(
        output_dir + '/' + det_df['img_path'].values,
        det_df['det'].values,
        transform=test_transform,
        masks=det_df['mask'].values)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        # pin_memory=True,
    )

    dets = []
    for fold in range(config['n_splits']):
        print('Fold [%d/%d]' %(fold + 1, config['n_splits']))

        # create model
        model = get_pose_model(config['arch'],
                          num_outputs=num_outputs,
                          freeze_bn=config['freeze_bn'])
        model = model.cuda()

        model_path = 'models/pose/%s/model_%d.pth' % (config['name'], fold+1)
        if not os.path.exists(model_path):
            print('%s is not exists.' %model_path)
            continue
        model.load_state_dict(torch.load(model_path))

        model.eval()

        fold_dets = []
        with torch.no_grad():
            for input, batch_det, mask in tqdm(test_loader, total=len(test_loader)):
                input = input.cuda()
                batch_det = batch_det.numpy()
                mask = mask.numpy()

                output = model(input)
                output = output.cpu()

                if config['rot'] == 'trig':
                    yaw = torch.atan2(output[..., 1:2], output[..., 0:1])
                    pitch = torch.atan2(output[..., 3:4], output[..., 2:3])
                    roll = torch.atan2(output[..., 5:6], output[..., 4:5])
                    roll = rotate(roll, -np.pi)

                pitch = pitch.cpu().numpy()[:, 0]
                yaw = yaw.cpu().numpy()[:, 0]
                roll = roll.cpu().numpy()[:, 0]

                batch_det[mask, 0] = pitch[mask]
                batch_det[mask, 1] = yaw[mask]
                batch_det[mask, 2] = roll[mask]

                fold_dets.append(batch_det)
        fold_dets = np.vstack(fold_dets)
        dets.append(fold_dets)
    dets = np.array(dets)
    dets = np.mean(dets, axis=0)
    det_df['det'] = dets.tolist()
    det_df = det_df.groupby('ImageId')['det'].apply(list)
    det_df = pd.DataFrame({
        'ImageId': det_df.index.values,
        'PredictionString': det_df.values,
    })

    for i in tqdm(range(len(df))):
        img_id = df.loc[i, 'ImageId']
        det = np.array(np.array(det_df.loc[det_df.ImageId == img_id, 'PredictionString'])[0])
        print(det)

        if args.show:
            img = cv2.imread('inputs/test_images/%s.jpg' %img_id)
            img_pred = visualize(img, det)
            plt.imshow(img_pred[..., ::-1])
            plt.show()

        df.loc[i, 'PredictionString'] = convert_labels_to_str(det[:, :7])

    name += '_%s' %args.pose_name

    df.to_csv('outputs/submissions/test/%s.csv' %name, index=False)


if __name__ == '__main__':
    main()
