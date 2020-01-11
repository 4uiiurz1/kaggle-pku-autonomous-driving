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

from lib.datasets import CropPoseDataset
from lib.utils.utils import *
from lib.models.model_factory import get_pose_model
from lib.optimizers import RAdam
from lib import losses
from lib.decodes import decode
from lib.utils.vis import visualize
from lib.postprocess.nms import nms


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
        decoded = json.load(f)

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

    dets = {img_id: [] for img_id in img_ids}
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

        for img_id, img_path in tqdm(zip(img_ids, img_paths), total=len(img_ids)):
            det = decoded[img_id]
            det = np.array(det)
            det = det[det[:, 6] > args.score_th]
            if args.nms:
                det = nms(det, dist_th=args.nms_th)

            if len(det) == 0:
                continue

            test_set = CropPoseDataset(
                img_path,
                det,
                transform=test_transform)
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                # pin_memory=True,
            )

            det = []
            with torch.no_grad():
                for i, (input, batch_det) in enumerate(test_loader):
                    input = input.cuda()
                    batch_det = batch_det.numpy()

                    output = model(input)
                    output = output.cpu()

                    if config['rot'] == 'trig':
                        yaw = torch.atan2(output[..., 1:2], output[..., 0:1])
                        pitch = torch.atan2(output[..., 3:4], output[..., 2:3])
                        roll = torch.atan2(output[..., 5:6], output[..., 4:5])
                        roll = rotate(roll, -np.pi)

                    batch_det[:, 0] = pitch.cpu().numpy()[:, 0]
                    batch_det[:, 1] = yaw.cpu().numpy()[:, 0]
                    batch_det[:, 2] = roll.cpu().numpy()[:, 0]

                    det.append(batch_det)
            det = np.concatenate(det, axis=0)
            dets[img_id].append(det)

            if args.show:
                img = cv2.imread(img_path)
                img_pred = visualize(img, det)
                plt.imshow(img_pred[..., ::-1])
                plt.show()

        if not config['cv']:
            df['PredictionString'] = preds_fold
            name = '%s_1_%.2f' %(args.name, args.score_th)
            if args.nms:
                name += '_nms%.2f' %args.nms_th
            df.to_csv('outputs/submissions/test/%s.csv' %name, index=False)
            return

    # decode
    decoded = {}
    for i in tqdm(range(len(df))):
        img_id = df.loc[i, 'ImageId']

        output = merged_outputs[img_id]

        det = decode(
            config,
            output['hm'],
            output['reg'],
            output['depth'],
            eular=output['eular'] if config['rot'] == 'eular' else None,
            trig=output['trig'] if config['rot'] == 'trig' else None,
            quat=output['quat'] if config['rot'] == 'quat' else None,
            wh=output['wh'] if config['wh'] else None,
            mask=output['mask'],
        )
        det = det.numpy()[0]

        decoded[img_id] = det.tolist()

        if args.nms:
            det = nms(det, dist_th=args.nms_th)

        if np.sum(det[:, 6] > args.score_th) >= args.min_samples:
            det = det[det[:, 6] > args.score_th]
        else:
            det = det[:args.min_samples]

        if args.show:
            img = cv2.imread('inputs/test_images/%s.jpg' %img_id)
            img_pred = visualize(img, det)
            plt.imshow(img_pred[..., ::-1])
            plt.show()

        df.loc[i, 'PredictionString'] = convert_labels_to_str(det[:, :7])

    with open('outputs/decoded/test/%s.json' %name, 'w') as f:
        json.dump(decoded, f)

    name = '%s_%.2f' %(args.name, args.score_th)
    if args.nms:
        name += '_nms%.2f' %args.nms_th
    if args.hflip:
        name += '_hf'
    if args.min_samples > 0:
        name += '_min%d' %args.min_samples
    df.to_csv('outputs/submissions/test/%s.csv' %name, index=False)


if __name__ == '__main__':
    main()
