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
from lib.models.model_factory import get_model
from lib.optimizers import RAdam
from lib import losses
from lib.decodes import decode
from lib.utils.vis import visualize
from lib.postprocess.nms import nms


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None)
    parser.add_argument('--models', default=None)
    parser.add_argument('--score_th', default=0.3, type=float)
    parser.add_argument('--nms', default=True, type=str2bool)
    parser.add_argument('--nms_th', default=0.1, type=float)
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()

    return args


def main():
    config = vars(parse_args())

    if config['name'] is None:
        config['name'] = 'ensemble_%s' % datetime.now().strftime('%m%d%H')

    config['models'] = config['models'].split(',')

    if not os.path.exists('models/detection/%s' % config['name']):
        os.makedirs('models/detection/%s' % config['name'])

    with open('models/detection/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    with open('models/detection/%s/config.yml' % config['models'][0], 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    df = pd.read_csv('inputs/sample_submission.csv')
    img_paths = np.array('inputs/test_images/' + df['ImageId'].values + '.jpg')
    mask_paths = np.array('inputs/test_masks/' + df['ImageId'].values + '.jpg')
    labels = np.array([convert_str_to_labels(s, names=['yaw', 'pitch', 'roll',
                       'x', 'y', 'z', 'score']) for s in df['PredictionString']])

    # merge
    merged_outputs = {}
    for i in tqdm(range(len(df))):
        img_id = df.loc[i, 'ImageId']

        output = {
            'hm': 0,
            'reg': 0,
            'depth': 0,
            'eular': 0 if model_config['rot'] == 'eular' else None,
            'trig': 0 if model_config['rot'] == 'trig' else None,
            'quat': 0 if model_config['rot'] == 'quat' else None,
            'wh': 0 if model_config['wh'] else None,
            'mask': 0,
        }

        merged_outputs[img_id] = output

    for model_name in config['models']:
        outputs = torch.load('outputs/raw/test/%s.pth' %model_name)

        for i in tqdm(range(len(df))):
            img_id = df.loc[i, 'ImageId']

            output = outputs[img_id]

            merged_outputs[img_id]['hm'] += output['hm'] / len(config['models'])
            merged_outputs[img_id]['reg'] += output['reg'] / len(config['models'])
            merged_outputs[img_id]['depth'] += output['depth'] / len(config['models'])
            merged_outputs[img_id]['trig'] += output['trig'] / len(config['models'])
            merged_outputs[img_id]['wh'] += output['wh'] / len(config['models'])
            merged_outputs[img_id]['mask'] += output['mask'] / len(config['models'])

    torch.save(merged_outputs, 'outputs/raw/test/%s.pth' %config['name'])

    # decode
    for i in tqdm(range(len(df))):
        img_id = df.loc[i, 'ImageId']

        output = merged_outputs[img_id]

        det = decode(
            model_config,
            output['hm'],
            output['reg'],
            output['depth'],
            eular=output['eular'] if model_config['rot'] == 'eular' else None,
            trig=output['trig'] if model_config['rot'] == 'trig' else None,
            quat=output['quat'] if model_config['rot'] == 'quat' else None,
            wh=output['wh'] if model_config['wh'] else None,
            mask=output['mask'],
        )
        det = det.numpy()[0]

        if config['nms']:
            det = nms(det, dist_th=config['nms_th'])

        if config['show']:
            img = cv2.imread('inputs/test_images/%s.jpg' %img_id)
            img_pred = visualize(img, det[det[:, 6] > config['score_th']])
            plt.imshow(img_pred[..., ::-1])
            plt.show()

        df.loc[i, 'PredictionString'] = convert_labels_to_str(det[det[:, 6] > config['score_th'], :7])

    df.to_csv('outputs/submissions/test/%s.csv' %config['name'], index=False)


if __name__ == '__main__':
    main()
