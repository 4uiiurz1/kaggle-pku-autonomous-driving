import os
import math
import json

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

import torch

from .utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from .utils.image import draw_dense_reg
from .utils.utils import convert_2d_to_3d, convert_3d_to_2d, rotate


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, mask_paths, labels, input_w=640, input_h=512,
                 down_ratio=4, transform=None, test=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.input_w = input_w
        self.input_h = input_h
        self.down_ratio = down_ratio
        self.transform = transform
        self.output_w = self.input_w // self.down_ratio
        self.output_h = self.input_h // self.down_ratio
        self.max_objs = 100
        self.mean = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(1, 1, 3)
        self.test = test

    def __getitem__(self, index):
        img_path, mask_path, label = self.img_paths[index], self.mask_paths[index], self.labels[index]
        num_objs = len(label)

        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        img = cv2.resize(img, (self.input_w, self.input_h))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (self.output_w, self.output_h))
            mask = 1 - mask.astype('float32') / 255
        else:
            mask = np.ones((self.output_h, self.output_w), dtype='float32')

        if self.test:
            img = img.astype('float32') / 255
            img = (img - self.mean) / self.std
            img = img.transpose(2, 0, 1)

            mask = mask[None, ...]

            return {
                'img_path': img_path,
                'input': img,
                'mask': mask,
            }

        kpts = []
        for k in range(num_objs):
            ann = label[k]
            kpts.append([ann['x'], ann['y'], ann['z']])
        kpts = np.array(kpts)
        zs = kpts[:, 2]
        kpts = np.array(convert_3d_to_2d(kpts[:, 0], kpts[:, 1], kpts[:, 2])).T
        kpts[:, 0] *= self.input_w / width
        kpts[:, 1] *= self.input_h / height

        if self.transform is not None:
            data = self.transform(image=img, mask=mask, keypoints=kpts)
            img = data['image']
            mask = data['mask']
            kpts = data['keypoints']

        for k, (x, y) in enumerate(kpts):
            label[k]['x'] = x
            label[k]['y'] = y

        img = img.astype('float32') / 255
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)

        mask = mask[None, ...]

        hm = np.zeros((1, self.output_h, self.output_w), dtype=np.float32)
        reg_mask = np.zeros((1, self.output_h, self.output_w), dtype=np.float32)
        reg = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
        depth = np.zeros((1, self.output_h, self.output_w), dtype=np.float32)
        eular = np.zeros((3, self.output_h, self.output_w), dtype=np.float32)
        trig = np.zeros((6, self.output_h, self.output_w), dtype=np.float32)
        quat = np.zeros((4, self.output_h, self.output_w), dtype=np.float32)
        gt = np.zeros((self.max_objs, 7), dtype=np.float32)

        for k in range(num_objs):
            ann = label[k]
            x, y = ann['x'], ann['y']
            x *= self.output_w / self.input_w
            y *= self.output_h / self.input_h
            if x < 0 or y < 0 or x > self.output_w or y > self.output_h:
                continue

            radius = 3  # TODO: make variable according to object size

            ct = np.array([x, y], dtype=np.float32)
            ct_int = ct.astype(np.int32)

            draw_umich_gaussian(hm[0], ct_int, radius)

            reg_mask[0, ct_int[1], ct_int[0]] = 1
            reg[:, ct_int[1], ct_int[0]] = ct - ct_int
            depth[0, ct_int[1], ct_int[0]] = ann['z']

            yaw = ann['yaw']
            pitch = ann['pitch']
            roll = ann['roll']

            eular[0, ct_int[1], ct_int[0]] = yaw
            eular[1, ct_int[1], ct_int[0]] = pitch
            eular[2, ct_int[1], ct_int[0]] = rotate(roll, np.pi)

            trig[0, ct_int[1], ct_int[0]] = math.cos(yaw)
            trig[1, ct_int[1], ct_int[0]] = math.sin(yaw)
            trig[2, ct_int[1], ct_int[0]] = math.cos(pitch)
            trig[3, ct_int[1], ct_int[0]] = math.sin(pitch)
            trig[4, ct_int[1], ct_int[0]] = math.cos(rotate(roll, np.pi))
            trig[5, ct_int[1], ct_int[0]] = math.sin(rotate(roll, np.pi))

            qx, qy, qz, qw = (R.from_euler('xyz', [yaw, pitch, roll])).as_quat()
            quat[0, ct_int[1], ct_int[0]] = qx
            quat[1, ct_int[1], ct_int[0]] = qy
            quat[2, ct_int[1], ct_int[0]] = qz
            quat[3, ct_int[1], ct_int[0]] = qw

            gt[k, 0] = ann['yaw']
            gt[k, 1] = ann['pitch']
            gt[k, 2] = ann['roll']
            gt[k, 3:5] = convert_2d_to_3d(ann['x'] * width / self.input_w, ann['y'] * height / self.input_h, ann['z'])
            gt[k, 5] = ann['z']
            gt[k, 6] = 1

        ret = {
            'img_path': img_path,
            'input': img,
            'mask': mask,
            'label': label,
            'hm': hm,
            'reg_mask': reg_mask,
            'reg': reg,
            'depth': depth,
            'eular': eular,
            'trig': trig,
            'quat': quat,
            'gt': gt,
        }

        # import matplotlib.pyplot as plt
        # plt.subplot(121)
        # plt.imshow((img.transpose((1, 2, 0)) * self.std + self.mean))
        # plt.subplot(122)
        # plt.imshow(ret['hm'][0])
        # plt.colorbar()
        # plt.show()

        return ret

    def __len__(self):
        return len(self.img_paths)
