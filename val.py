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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None)
    parser.add_argument('--score_th', default=0.1, type=float)
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

    df = pd.read_csv('inputs/train.csv')
    img_paths = np.array('inputs/train_images/' + df['ImageId'].values + '.jpg')
    mask_paths = np.array('inputs/train_masks/' + df['ImageId'].values + '.jpg')
    labels = np.array([convert_str_to_labels(s) for s in df['PredictionString']])

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

    if config['wh']:
        heads['wh'] = 2

    # criterion = OrderedDict()
    # for head in heads.keys():
    #     criterion[head] = losses.__dict__[config[head + '_loss']]().cuda()

    pred_df = df.copy()
    pred_df['PredictionString'] = np.nan
    #
    # avg_meters = {'loss': AverageMeter()}
    # for head in heads.keys():
    #     avg_meters[head] = AverageMeter()

    kf = KFold(n_splits=config['n_splits'], shuffle=True, random_state=41)
    for fold, (train_idx, val_idx) in enumerate(kf.split(img_paths)):
        print('Fold [%d/%d]' %(fold + 1, config['n_splits']))

        train_img_paths, val_img_paths = img_paths[train_idx], img_paths[val_idx]
        train_mask_paths, val_mask_paths = mask_paths[train_idx], mask_paths[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        val_set = Dataset(
            val_img_paths,
            val_mask_paths,
            val_labels,
            input_w=config['input_w'],
            input_h=config['input_h'],
            transform=None)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            # pin_memory=True,
        )

        model = get_model(config['arch'], heads=heads)
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

                output = model(input)

                # loss = 0
                # losses = {}
                # for head in heads.keys():
                #     losses[head] = criterion[head](output[head], batch[head].cuda(),
                #                                    mask if head == 'hm' else reg_mask)
                #     loss += losses[head]
                # losses['loss'] = loss
                #
                # avg_meters['loss'].update(losses['loss'].item(), input.size(0))
                # postfix = OrderedDict([('loss', avg_meters['loss'].avg)])
                # for head in heads.keys():
                #     avg_meters[head].update(losses[head].item(), input.size(0))
                #     postfix[head + '_loss'] = avg_meters[head].avg
                # pbar.set_postfix(postfix)

                if config['rot'] == 'eular':
                    dets = decode(output['hm'], output['reg'], output['depth'], eular=output['eular'])
                elif config['rot'] == 'trig':
                    dets = decode(output['hm'], output['reg'], output['depth'], trig=output['trig'])
                elif config['rot'] == 'quat':
                    dets = decode(output['hm'], output['reg'], output['depth'], quat=output['quat'])
                dets = dets.detach().cpu().numpy()

                for k, det in enumerate(dets):
                    img_id = os.path.splitext(os.path.basename(batch['img_path'][k]))[0]
                    pred_df.loc[pred_df.ImageId == img_id, 'PredictionString'] = convert_labels_to_str(det[det[:, -1] > args.score_th])

                    if args.show:
                        gt = batch['gt'].numpy()[k]

                        img = cv2.imread(batch['img_path'][k])
                        img_gt = visualize(img, gt[gt[:, -1] > 0])
                        img_pred = visualize(img, det[det[:, -1] > args.score_th])

                        plt.subplot(121)
                        plt.imshow(img_gt[..., ::-1])
                        plt.subplot(122)
                        plt.imshow(img_pred[..., ::-1])
                        plt.show()

                pbar.update(1)
            pbar.close()

        torch.cuda.empty_cache()

        if not config['cv']:
            break

    # print('loss: %f' %avg_meters['loss'].avg)

    pred_df.to_csv('preds/%s_%.2f.csv' %(args.name, args.score_th), index=False)
    print(pred_df.head())


if __name__ == '__main__':
    main()
