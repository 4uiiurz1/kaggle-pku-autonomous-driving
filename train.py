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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 4)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18_fpn',
                        help='model architecture: (default: resnet18_fpn)')
    parser.add_argument('--input_w', default=1280, type=int)
    parser.add_argument('--input_h', default=1024, type=int)
    parser.add_argument('--freeze_bn', default=False, type=str2bool)
    parser.add_argument('--rot', default='trig', choices=['eular', 'trig', 'quat'])
    parser.add_argument('--wh', default=True, type=str2bool)
    parser.add_argument('--gn', default=True, type=str2bool)
    parser.add_argument('--ws', default=True, type=str2bool)
    parser.add_argument('--lhalf', default=True, type=str2bool)

    # loss
    parser.add_argument('--hm_loss', default='FocalLoss')
    parser.add_argument('--reg_loss', default='L1Loss')
    parser.add_argument('--wh_loss', default='L1Loss')
    parser.add_argument('--depth_loss', default='L1Loss')
    parser.add_argument('--eular_loss', default='L1Loss')
    parser.add_argument('--trig_loss', default='L1Loss')
    parser.add_argument('--quat_loss', default='L1Loss')

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
    parser.add_argument('--hflip', default=True, type=str2bool)
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


def train(config, heads, train_loader, model, criterion, optimizer, epoch):
    avg_meters = {'loss': AverageMeter()}
    for head in heads.keys():
        avg_meters[head] = AverageMeter()

    model.train()

    pbar = tqdm(total=len(train_loader))
    for i, batch in enumerate(train_loader):
        input = batch['input'].cuda()
        mask = batch['mask'].cuda()
        reg_mask = batch['reg_mask'].cuda()

        output = model(input)

        loss = 0
        losses = {}
        for head in heads.keys():
            losses[head] = criterion[head](output[head], batch[head].cuda(),
                                           mask if head == 'hm' else reg_mask)
            if head == 'wh':
                loss += 0.1 * losses[head]
            else:
                loss += losses[head]
        losses['loss'] = loss

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        if config['apex']:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        avg_meters['loss'].update(losses['loss'].item(), input.size(0))
        postfix = OrderedDict([('loss', avg_meters['loss'].avg)])
        for head in heads.keys():
            avg_meters[head].update(losses[head].item(), input.size(0))
            postfix[head + '_loss'] = avg_meters[head].avg
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return avg_meters['loss'].avg


def validate(config, heads, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter()}
    for head in heads.keys():
        avg_meters[head] = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for i, batch in enumerate(val_loader):
            input = batch['input'].cuda()
            mask = batch['mask'].cuda()
            reg_mask = batch['reg_mask'].cuda()

            output = model(input)

            loss = 0
            losses = {}
            for head in heads.keys():
                losses[head] = criterion[head](output[head], batch[head].cuda(),
                                               mask if head == 'hm' else reg_mask)
                if head == 'wh':
                    loss += 0.1 * losses[head]
                else:
                    loss += losses[head]
            losses['loss'] = loss

            avg_meters['loss'].update(losses['loss'].item(), input.size(0))
            postfix = OrderedDict([('loss', avg_meters['loss'].avg)])
            for head in heads.keys():
                avg_meters[head].update(losses[head].item(), input.size(0))
                postfix[head + '_loss'] = avg_meters[head].avg
            pbar.set_postfix(postfix)
            pbar.update(1)

            # dets = decode(
            #     config,
            #     output['hm'],
            #     output['reg'],
            #     output['depth'],
            #     eular=output['eular'] if config['rot'] == 'eular' else None,
            #     trig=output['trig'] if config['rot'] == 'trig' else None,
            #     quat=output['quat'] if config['rot'] == 'quat' else None,
            #     wh=output['wh'] if config['wh'] else None,
            # )
        pbar.close()

        # print(dets[0, 0])

    return avg_meters['loss'].avg


def main():
    config = vars(parse_args())

    if config['name'] is None:
        config['name'] = '%s_%s' % (config['arch'], datetime.now().strftime('%m%d%H'))

    if not os.path.exists('models/%s' % config['name']):
        os.makedirs('models/%s' % config['name'])

    if config['resume']:
        with open('models/%s/config.yml' % config['name'], 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['resume'] = True

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    print('-'*20)
    for key in config.keys():
        print('- %s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    df = pd.read_csv('inputs/train.csv')
    img_paths = np.array('inputs/train_images/' + df['ImageId'].values + '.jpg')
    mask_paths = np.array('inputs/train_masks/' + df['ImageId'].values + '.jpg')
    labels = np.array([convert_str_to_labels(s) for s in df['PredictionString']])

    if config['resume']:
        checkpoint = torch.load('models/%s/checkpoint.pth.tar' % config['name'])

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

    criterion = OrderedDict()
    for head in heads.keys():
        criterion[head] = losses.__dict__[config[head + '_loss']]().cuda()

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
    ], keypoint_params=KeypointParams(format='xy', remove_invisible=False))

    val_transform = None

    folds = []
    best_losses = []
    # best_scores = []

    kf = KFold(n_splits=config['n_splits'], shuffle=True, random_state=41)
    for fold, (train_idx, val_idx) in enumerate(kf.split(img_paths)):
        print('Fold [%d/%d]' %(fold + 1, config['n_splits']))

        if (config['resume'] and fold < checkpoint['fold'] - 1) or (not config['resume'] and os.path.exists('models/%s/model_%d.pth' % (config['name'], fold+1))):
            log = pd.read_csv('models/%s/log_%d.csv' %(config['name'], fold+1))
            best_loss = log.loc[log['val_loss'].values.argmin(), 'val_loss']
            # best_loss, best_score = log.loc[log['val_loss'].values.argmin(), ['val_loss', 'val_score']].values
            folds.append(str(fold + 1))
            best_losses.append(best_loss)
            # best_scores.append(best_score)
            continue

        train_img_paths, val_img_paths = img_paths[train_idx], img_paths[val_idx]
        train_mask_paths, val_mask_paths = mask_paths[train_idx], mask_paths[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        # train
        train_set = Dataset(
            train_img_paths,
            train_mask_paths,
            train_labels,
            input_w=config['input_w'],
            input_h=config['input_h'],
            transform=train_transform,
            lhalf=config['lhalf'],
            hflip=config['hflip_p'] if config['hflip'] else 0,
            scale=config['scale_p'] if config['scale'] else 0,
            scale_limit=config['scale_limit'])
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            # pin_memory=True,
        )

        val_set = Dataset(
            val_img_paths,
            val_mask_paths,
            val_labels,
            input_w=config['input_w'],
            input_h=config['input_h'],
            transform=val_transform,
            lhalf=config['lhalf'])
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            # pin_memory=True,
        )

        # create model
        model = get_model(config['arch'], heads=heads, gn=config['gn'],
                          ws=config['ws'], freeze_bn=config['freeze_bn'])
        model = model.cuda()
        # print(model)

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
            log = pd.read_csv('models/%s/log_%d.csv' % (config['name'], fold+1)).to_dict(orient='list')
            best_loss = checkpoint['best_loss']

        for epoch in range(start_epoch, config['epochs']):
            print('Epoch [%d/%d]' % (epoch + 1, config['epochs']))

            # train for one epoch
            train_loss = train(config, heads, train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            val_loss = validate(config, heads, val_loader, model, criterion)

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

            pd.DataFrame(log).to_csv('models/%s/log_%d.csv' % (config['name'], fold+1), index=False)

            if val_loss < best_loss:
                torch.save(model.state_dict(), 'models/%s/model_%d.pth' % (config['name'], fold+1))
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
            torch.save(state, 'models/%s/checkpoint.pth.tar' % config['name'])

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
        results.to_csv('models/%s/results.csv' % config['name'], index=False)

        torch.cuda.empty_cache()

        if not config['cv']:
            break


if __name__ == '__main__':
    main()
