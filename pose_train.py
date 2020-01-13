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

from lib.datasets import PoseDataset
from lib.utils.utils import *
from lib.models.model_factory import get_pose_model
from lib.optimizers import RAdam
from lib.decodes import decode


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        help='model architecture: (default: resnet18)')
    parser.add_argument('--input_w', default=224, type=int)
    parser.add_argument('--input_h', default=224, type=int)
    parser.add_argument('--freeze_bn', default=False, type=str2bool)
    parser.add_argument('--rot', default='trig', choices=['eular', 'trig', 'quat'])

    # loss
    parser.add_argument('--loss', default='L1Loss')

    # optimizer
    parser.add_argument('--optimizer', default='RAdam')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)

    # dataset
    parser.add_argument('--cv', default=False, type=str2bool)
    parser.add_argument('--n_splits', default=5, type=int)

    # augmentation
    parser.add_argument('--hflip', default=False, type=str2bool)
    parser.add_argument('--hflip_p', default=0.5, type=float)
    parser.add_argument('--shift', default=True, type=str2bool)
    parser.add_argument('--shift_p', default=0.5, type=float)
    parser.add_argument('--shift_limit', default=0.1, type=float)
    parser.add_argument('--scale', default=True, type=str2bool)
    parser.add_argument('--scale_p', default=0.5, type=float)
    parser.add_argument('--scale_limit', default=0.1, type=float)
    parser.add_argument('--hsv', default=True, type=str2bool)
    parser.add_argument('--hsv_p', default=0.5, type=float)
    parser.add_argument('--hue_limit', default=20, type=int)
    parser.add_argument('--sat_limit', default=0, type=int)
    parser.add_argument('--val_limit', default=0, type=int)
    parser.add_argument('--brightness', default=True, type=str2bool)
    parser.add_argument('--brightness_p', default=0.5, type=float)
    parser.add_argument('--brightness_limit', default=0.2, type=float)
    parser.add_argument('--contrast', default=True, type=str2bool)
    parser.add_argument('--contrast_p', default=0.5, type=float)
    parser.add_argument('--contrast_limit', default=0.2, type=float)
    parser.add_argument('--iso_noise', default=False, type=str2bool)
    parser.add_argument('--iso_noise_p', default=0.5, type=float)
    parser.add_argument('--clahe', default=False, type=str2bool)
    parser.add_argument('--clahe_p', default=0.5, type=float)

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--apex', action='store_true')

    args = parser.parse_args()

    return args


def train(config, train_loader, model, criterion, optimizer, epoch):
    avg_meter = AverageMeter()

    model.train()

    pbar = tqdm(total=len(train_loader))
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        output = model(input)

        loss = criterion(output, target.float())

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        if config['apex']:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        avg_meter.update(loss.item(), input.size(0))
        postfix = OrderedDict([('loss', avg_meter.avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return avg_meter.avg


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            output = model(input)

            loss = 0
            losses = {}
            loss = criterion(output, target.float())
            losses['loss'] = loss

            avg_meters['loss'].update(losses['loss'].item(), input.size(0))
            postfix = OrderedDict([('loss', avg_meters['loss'].avg)])
            pbar.set_postfix(postfix)
            pbar.update(1)

        pbar.close()

    return avg_meters['loss'].avg


def main():
    config = vars(parse_args())

    if config['name'] is None:
        config['name'] = '%s_%s' % (config['arch'], datetime.now().strftime('%m%d%H'))

    if not os.path.exists('models/pose/%s' % config['name']):
        os.makedirs('models/pose/%s' % config['name'])

    if config['resume']:
        with open('models/pose/%s/config.yml' % config['name'], 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['resume'] = True

    with open('models/pose/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    print('-'*20)
    for key in config.keys():
        print('- %s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    df = pd.read_csv('inputs/train.csv')
    img_ids = df['ImageId'].values
    pose_df = pd.read_csv('processed/pose_train.csv')
    pose_df['img_path'] = 'processed/pose_images/train/' + pose_df['img_path']

    if config['resume']:
        checkpoint = torch.load('models/pose/%s/checkpoint.pth.tar' % config['name'])

    if config['rot'] == 'eular':
        num_outputs = 3
    elif config['rot'] == 'trig':
        num_outputs = 6
    elif config['rot'] == 'quat':
        num_outputs = 4
    else:
        raise NotImplementedError

    if config['loss'] == 'L1Loss':
        criterion = nn.L1Loss().cuda()
    elif config['loss'] == 'MSELoss':
        criterion = nn.MSELoss().cuda()
    else:
        raise NotImplementedError

    train_transform = Compose([
        transforms.ShiftScaleRotate(
            shift_limit=config['shift_limit'],
            scale_limit=0,
            rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=config['shift_p']
        ) if config['shift'] else NoOp(),
        OneOf([
            transforms.HueSaturationValue(
                hue_shift_limit=config['hue_limit'],
                sat_shift_limit=config['sat_limit'],
                val_shift_limit=config['val_limit'],
                p=config['hsv_p']
            ) if config['hsv'] else NoOp(),
            transforms.RandomBrightness(
                limit=config['brightness_limit'],
                p=config['brightness_p'],
            ) if config['brightness'] else NoOp(),
            transforms.RandomContrast(
                limit=config['contrast_limit'],
                p=config['contrast_p'],
            ) if config['contrast'] else NoOp(),
        ], p=1),
        transforms.ISONoise(
            p=config['iso_noise_p'],
        ) if config['iso_noise'] else NoOp(),
        transforms.CLAHE(
            p=config['clahe_p'],
        ) if config['clahe'] else NoOp(),
        transforms.Resize(config['input_w'], config['input_h']),
        transforms.Normalize(),
        ToTensor(),
    ])

    val_transform = Compose([
        transforms.Resize(config['input_w'], config['input_h']),
        transforms.Normalize(),
        ToTensor(),
    ])

    folds = []
    best_losses = []

    kf = KFold(n_splits=config['n_splits'], shuffle=True, random_state=41)
    for fold, (train_idx, val_idx) in enumerate(kf.split(img_ids)):
        print('Fold [%d/%d]' %(fold + 1, config['n_splits']))

        if (config['resume'] and fold < checkpoint['fold'] - 1) or (not config['resume'] and os.path.exists('pose_models/%s/model_%d.pth' % (config['name'], fold+1))):
            log = pd.read_csv('models/pose/%s/log_%d.csv' %(config['name'], fold+1))
            best_loss = log.loc[log['val_loss'].values.argmin(), 'val_loss']
            # best_loss, best_score = log.loc[log['val_loss'].values.argmin(), ['val_loss', 'val_score']].values
            folds.append(str(fold + 1))
            best_losses.append(best_loss)
            # best_scores.append(best_score)
            continue

        train_img_ids, val_img_ids = img_ids[train_idx], img_ids[val_idx]

        train_img_paths = []
        train_labels = []
        for img_id in train_img_ids:
            tmp = pose_df.loc[pose_df.ImageId == img_id]

            img_path = tmp['img_path'].values
            train_img_paths.append(img_path)

            yaw = tmp['yaw'].values
            pitch = tmp['pitch'].values
            roll = tmp['roll'].values
            roll = rotate(roll, np.pi)

            if config['rot'] == 'eular':
                label = np.array([
                    yaw,
                    pitch,
                    roll
                ]).T
            elif config['rot'] == 'trig':
                label = np.array([
                    np.cos(yaw),
                    np.sin(yaw),
                    np.cos(pitch),
                    np.sin(pitch),
                    np.cos(roll),
                    np.sin(roll),
                ]).T
            elif config['rot'] == 'quat':
                raise NotImplementedError
            else:
                raise NotImplementedError

            train_labels.append(label)
        train_img_paths = np.hstack(train_img_paths)
        train_labels = np.vstack(train_labels)

        val_img_paths = []
        val_labels = []
        for img_id in val_img_ids:
            tmp = pose_df.loc[pose_df.ImageId == img_id]

            img_path = tmp['img_path'].values
            val_img_paths.append(img_path)

            yaw = tmp['yaw'].values
            pitch = tmp['pitch'].values
            roll = tmp['roll'].values
            roll = rotate(roll, np.pi)

            if config['rot'] == 'eular':
                label = np.array([
                    yaw,
                    pitch,
                    roll
                ]).T
            elif config['rot'] == 'trig':
                label = np.array([
                    np.cos(yaw),
                    np.sin(yaw),
                    np.cos(pitch),
                    np.sin(pitch),
                    np.cos(roll),
                    np.sin(roll),
                ]).T
            elif config['rot'] == 'quat':
                raise NotImplementedError
            else:
                raise NotImplementedError

            val_labels.append(label)
        val_img_paths = np.hstack(val_img_paths)
        val_labels = np.vstack(val_labels)

        # train
        train_set = PoseDataset(
            train_img_paths,
            train_labels,
            transform=train_transform,
        )
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            # pin_memory=True,
        )

        val_set = PoseDataset(
            val_img_paths,
            val_labels,
            transform=val_transform,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            # pin_memory=True,
        )

        # create model
        model = get_pose_model(config['arch'],
                          num_outputs=num_outputs,
                          freeze_bn=config['freeze_bn'])
        model = model.cuda()

        params = filter(lambda p: p.requires_grad, model.parameters())
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'RAdam':
            optimizer = RAdam(params, lr=config['lr'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                                  nesterov=config['nesterov'], weight_decay=config['weight_decay'])
        else:
            raise NotImplementedError

        if config['apex']:
            amp.initialize(model, optimizer, opt_level='O1')

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                       verbose=1, min_lr=config['min_lr'])
        elif config['scheduler'] == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
        else:
            raise NotImplementedError

        log = {
            'epoch': [],
            'loss': [],
            # 'score': [],
            'val_loss': [],
            # 'val_score': [],
        }

        best_loss = float('inf')
        # best_score = float('inf')

        start_epoch = 0

        if config['resume'] and fold == checkpoint['fold'] - 1:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            log = pd.read_csv('models/pose/%s/log_%d.csv' % (config['name'], fold+1)).to_dict(orient='list')
            best_loss = checkpoint['best_loss']

        for epoch in range(start_epoch, config['epochs']):
            print('Epoch [%d/%d]' % (epoch + 1, config['epochs']))

            # train for one epoch
            train_loss = train(config, train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            val_loss = validate(config, val_loader, model, criterion)

            if config['scheduler'] == 'CosineAnnealingLR':
                scheduler.step()
            elif config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_loss)

            print('loss %.4f - val_loss %.4f' % (train_loss, val_loss))
            # print('loss %.4f - score %.4f - val_loss %.4f - val_score %.4f'
            #       % (train_loss, train_score, val_loss, val_score))

            log['epoch'].append(epoch)
            log['loss'].append(train_loss)
            # log['score'].append(train_score)
            log['val_loss'].append(val_loss)
            # log['val_score'].append(val_score)

            pd.DataFrame(log).to_csv('models/pose/%s/log_%d.csv' % (config['name'], fold+1), index=False)

            if val_loss < best_loss:
                torch.save(model.state_dict(), 'models/pose/%s/model_%d.pth' % (config['name'], fold+1))
                best_loss = val_loss
                # best_score = val_score
                print("=> saved best model")

            state = {
                'fold': fold + 1,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(state, 'models/pose/%s/checkpoint.pth.tar' % config['name'])

        print('val_loss:  %f' % best_loss)
        # print('val_score: %f' % best_score)

        folds.append(str(fold + 1))
        best_losses.append(best_loss)
        # best_scores.append(best_score)

        results = pd.DataFrame({
            'fold': folds + ['mean'],
            'best_loss': best_losses + [np.mean(best_losses)],
            # 'best_score': best_scores + [np.mean(best_scores)],
        })

        print(results)
        results.to_csv('models/pose/%s/results.csv' % config['name'], index=False)

        del model
        torch.cuda.empty_cache()

        del train_set, train_loader
        del val_set, val_loader
        gc.collect()

        if not config['cv']:
            break


if __name__ == '__main__':
    main()
