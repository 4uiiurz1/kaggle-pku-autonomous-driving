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
    parser.add_argument('--score_th', default=0.9, type=float)
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yaml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    df = pd.read_csv('inputs/sample_submission.csv')
    img_paths = np.array('inputs/test_images/' + df['ImageId'].values + '.jpg')
    mask_paths = np.array('inputs/test_masks/' + df['ImageId'].values + '.jpg')
    labels = np.array([convert_str_to_labels(s, names=['yaw', 'pitch', 'roll',
                       'x', 'y', 'z', 'score']) for s in df['PredictionString']])

    test_set = Dataset(
        img_paths,
        mask_paths,
        labels,
        transform=None,
        test=True)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        # pin_memory=True,
    )

    heads = OrderedDict([
        ('hm', 1),
        ('reg', 2),
        ('depth', 1),
    ])

    if config['rot'] == 'eular':
        heads['eular'] = 3
    elif config['rot'] == 'trig':
        heads['trig'] = 6
    elif config['rot'] == 'quat':
        heads['quat'] = 4
    else:
        raise NotImplementedError

    preds = []
    for fold in range(config['n_splits']):
        print('Fold [%d/%d]' %(fold + 1, config['n_splits']))

        model = resnet_fpn.ResNetFPN(backbone='resnet18', heads=heads)
        model = model.cuda()

        model_path = 'models/%s/model_%d.pth' % (config['name'], fold+1)
        if not os.path.exists(model_path):
            print('%s is not exists.' %model_path)
            continue
        model.load_state_dict(torch.load(model_path))

        model.eval()

        preds_fold = []
        with torch.no_grad():
            pbar = tqdm(total=len(test_loader))
            for i, batch in enumerate(test_loader):
                input = batch['input'].cuda()
                mask = batch['mask'].cuda()

                output = model(input)

                if config['rot'] == 'eular':
                    dets = decode(output['hm'], output['reg'], output['depth'], eular=output['eular'])
                elif config['rot'] == 'trig':
                    dets = decode(output['hm'], output['reg'], output['depth'], trig=output['trig'])
                elif config['rot'] == 'quat':
                    dets = decode(output['hm'], output['reg'], output['depth'], quat=output['quat'])
                dets = dets.detach().cpu().numpy()

                for k, det in enumerate(dets):
                    preds_fold.append(convert_labels_to_str(det[det[:, -1] > args.score_th]))

                    if args.show and len(det[det[:, -1] > args.score_th]) != 0:
                        img = cv2.imread(batch['img_path'][k])
                        img_pred = visualize(img, det[det[:, -1] > args.score_th])
                        plt.imshow(img_pred)
                        plt.show()

                pbar.update(1)
            pbar.close()

        df['PredictionString'] = preds_fold
        df.to_csv('submissions/%s_%d_%.2f.csv' %(args.name, fold + 1, args.score_th), index=False)
        print(df.head())

        torch.cuda.empty_cache()

        if not config['cv']:
            break


if __name__ == '__main__':
    main()
