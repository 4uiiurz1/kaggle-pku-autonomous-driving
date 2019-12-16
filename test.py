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
    parser.add_argument('--score_th', default=0.3, type=float)
    parser.add_argument('--nms', default=True, type=str2bool)
    parser.add_argument('--nms_th', default=0.1, type=float)
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
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
        input_w=config['input_w'],
        input_h=config['input_h'],
        transform=None,
        test=True,
        lhalf=config['lhalf'])
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config['batch_size'],
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

    if config['wh']:
        heads['wh'] = 2

    preds = []
    outputs = []
    for fold in range(config['n_splits']):
        print('Fold [%d/%d]' %(fold + 1, config['n_splits']))

        model = get_model(config['arch'], heads=heads)
        model = model.cuda()

        model_path = 'models/%s/model_%d.pth' % (config['name'], fold+1)
        if not os.path.exists(model_path):
            print('%s is not exists.' %model_path)
            continue
        model.load_state_dict(torch.load(model_path))

        model.eval()

        preds_fold = []
        outputs_fold = {}
        with torch.no_grad():
            pbar = tqdm(total=len(test_loader))
            for i, batch in enumerate(test_loader):
                input = batch['input'].cuda()
                mask = batch['mask'].cuda()

                output = model(input)
                # print(output)

                for b in range(len(batch['img_path'])):
                    img_id = os.path.splitext(os.path.basename(batch['img_path'][b]))[0]
                    outputs_fold[img_id] = {
                        'hm': output['hm'][b:b+1].detach().cpu(),
                        'reg': output['reg'][b:b+1].cpu(),
                        'depth': output['depth'][b:b+1].cpu(),
                        'eular': output['eular'][b:b+1].cpu() if config['rot'] == 'eular' else None,
                        'trig': output['trig'][b:b+1].cpu() if config['rot'] == 'trig' else None,
                        'quat': output['quat'][b:b+1].cpu() if config['rot'] == 'quat' else None,
                        'wh': output['wh'][b:b+1].cpu() if config['wh'] else None,
                        'mask': mask[b:b+1].cpu(),
                    }

                dets = decode(
                    config,
                    output['hm'],
                    output['reg'],
                    output['depth'],
                    eular=output['eular'] if config['rot'] == 'eular' else None,
                    trig=output['trig'] if config['rot'] == 'trig' else None,
                    quat=output['quat'] if config['rot'] == 'quat' else None,
                    wh=output['wh'] if config['wh'] else None,
                    mask=mask,
                )
                dets = dets.detach().cpu().numpy()

                for k, det in enumerate(dets):
                    if args.nms:
                        det = nms(det, dist_th=args.nms_th)
                    preds_fold.append(convert_labels_to_str(det[det[:, 6] > args.score_th, :7]))

                    if args.show:
                        img = cv2.imread(batch['img_path'][k])
                        img_pred = visualize(img, det[det[:, 6] > args.score_th])
                        plt.imshow(img_pred[..., ::-1])
                        plt.show()

                pbar.update(1)
            pbar.close()

        outputs.append(outputs_fold)

        torch.cuda.empty_cache()

        if not config['cv']:
            break

    # merge
    merged_outputs = {}
    for i in tqdm(range(len(df))):
        img_id = df.loc[i, 'ImageId']

        output = {
            'hm': 0,
            'reg': 0,
            'depth': 0,
            'eular': 0 if config['rot'] == 'eular' else None,
            'trig': 0 if config['rot'] == 'trig' else None,
            'quat': 0 if config['rot'] == 'quat' else None,
            'wh': 0 if config['wh'] else None,
        }

        for outputs_fold in outputs:
            output['hm'] += outputs_fold[img_id]['hm'] / config['n_splits']
            output['reg'] += outputs_fold[img_id]['reg'] / config['n_splits']
            output['depth'] += outputs_fold[img_id]['depth'] / config['n_splits']
            if config['rot'] == 'eular':
                output['eular'] += outputs_fold[img_id]['eular'] / config['n_splits']
            if config['rot'] == 'trig':
                output['trig'] += outputs_fold[img_id]['trig'] / config['n_splits']
            if config['rot'] == 'quat':
                output['quat'] += outputs_fold[img_id]['quat'] / config['n_splits']
            if config['wh']:
                output['wh'] += outputs_fold[img_id]['wh'] / config['n_splits']
        output['mask'] = outputs[0][img_id]['mask']

        merged_outputs[img_id] = output

    torch.save(merged_outputs, 'outputs/%s.pth' %args.name)

    # ensemble duplicate images
    dup_df = pd.read_csv('processed/test_image_hash.csv')
    dups = dup_df.hash.value_counts()
    dups = dups.loc[dups>1]

    for i in range(len(dups)):
        img_ids = dup_df[dup_df.hash == dups.index[i]].ImageId

        output = {
            'hm': 0,
            'reg': 0,
            'depth': 0,
            'eular': 0 if config['rot'] == 'eular' else None,
            'trig': 0 if config['rot'] == 'trig' else None,
            'quat': 0 if config['rot'] == 'quat' else None,
            'wh': 0 if config['wh'] else None,
            'mask': 0,
        }
        for img_id in img_ids:
            output['hm'] += merged_outputs[img_id]['hm'] / len(img_ids)
            output['reg'] += merged_outputs[img_id]['reg'] / len(img_ids)
            output['depth'] += merged_outputs[img_id]['depth'] / len(img_ids)
            if config['rot'] == 'eular':
                output['eular'] += merged_outputs[img_id]['eular'] / len(img_ids)
            if config['rot'] == 'trig':
                output['trig'] += merged_outputs[img_id]['trig'] / len(img_ids)
            if config['rot'] == 'quat':
                output['quat'] += merged_outputs[img_id]['quat'] / len(img_ids)
            if config['wh']:
                output['wh'] += merged_outputs[img_id]['wh'] / len(img_ids)
            output['mask'] += merged_outputs[img_id]['mask'] / len(img_ids)

        for img_id in img_ids:
            merged_outputs[img_id] = output

    # decode
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

        if args.nms:
            det = nms(det, dist_th=args.nms_th)

        df.loc[i, 'PredictionString'] = convert_labels_to_str(det[det[:, 6] > args.score_th, :7])


    name = '%s_%.2f' %(args.name, args.score_th)
    if args.nms:
        name += '_nms%.2f' %args.nms_th
    if args.ensemble_dup:
        name += '_ensemble_dup'
    df.to_csv('submissions/%s.csv' %name, index=False)


if __name__ == '__main__':
    main()
