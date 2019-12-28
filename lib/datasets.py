import os
import math
import json

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import torch

from albumentations.augmentations import functional as F

from .utils.image import get_bbox, gaussian_radius
from .utils.image import draw_umich_gaussian, draw_msra_gaussian
from .utils.image import draw_dense_reg
from .utils.utils import convert_2d_to_3d, convert_3d_to_2d, rotate
from .utils.vis import visualize


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, mask_paths, labels, input_w=640, input_h=512,
                 down_ratio=4, transform=None, test=False, lhalf=False,
                 hflip=0, scale=0, scale_limit=0,
                 test_img_paths=None, test_mask_paths=None, test_outputs=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.test_img_paths = test_img_paths
        self.test_mask_paths = test_mask_paths
        self.test_outputs = test_outputs
        self.input_w = input_w
        self.input_h = input_h
        self.down_ratio = down_ratio
        self.transform = transform
        self.test = test
        self.lhalf = lhalf
        self.hflip = hflip
        self.scale = scale
        self.scale_limit = scale_limit
        self.output_w = self.input_w // self.down_ratio
        self.output_h = self.input_h // self.down_ratio
        self.max_objs = 100
        self.mean = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(1, 1, 3)

    def __getitem__(self, index):
        if index < len(self.img_paths):
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

                if self.lhalf:
                    img = img[:, self.input_h // 2:]
                    mask = mask[:, self.output_h // 2:]

                return {
                    'img_path': img_path,
                    'input': img,
                    'mask': mask,
                }

            kpts = []
            poses = []
            for k in range(num_objs):
                ann = label[k]
                kpts.append([ann['x'], ann['y'], ann['z']])
                poses.append([ann['yaw'], ann['pitch'], ann['roll']])
            kpts = np.array(kpts)
            poses = np.array(poses)

            if np.random.random() < self.hflip:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
                kpts[:, 0] *= -1
                poses[:, [0, 2]] *= -1

            if np.random.random() < self.scale:
                scale = np.random.uniform(-self.scale_limit, self.scale_limit) + 1.0
                img = F.shift_scale_rotate(img, angle=0, scale=scale, dx=0, dy=0)
                mask = F.shift_scale_rotate(mask, angle=0, scale=scale, dx=0, dy=0)
                kpts[:, 2] /= scale

            kpts = np.array(convert_3d_to_2d(kpts[:, 0], kpts[:, 1], kpts[:, 2])).T
            kpts[:, 0] *= self.input_w / width
            kpts[:, 1] *= self.input_h / height

            if self.transform is not None:
                data = self.transform(image=img, mask=mask, keypoints=kpts)
                img = data['image']
                mask = data['mask']
                kpts = data['keypoints']

            for k, ((x, y), (yaw, pitch, roll)) in enumerate(zip(kpts, poses)):
                label[k]['x'] = x
                label[k]['y'] = y
                label[k]['yaw'] = yaw
                label[k]['pitch'] = pitch
                label[k]['roll'] = roll

            img = img.astype('float32') / 255
            img = (img - self.mean) / self.std
            img = img.transpose(2, 0, 1)

            mask = mask[None, ...]

            hm = np.zeros((1, self.output_h, self.output_w), dtype=np.float32)
            reg_mask = np.zeros((1, self.output_h, self.output_w), dtype=np.float32)
            reg = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
            wh = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
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

                bbox = get_bbox(
                    ann['yaw'],
                    ann['pitch'],
                    ann['roll'],
                    *convert_2d_to_3d(ann['x'] * width / self.input_w, ann['y'] * height / self.input_h, ann['z']),
                    ann['z'],
                    width,
                    height,
                    self.output_w,
                    self.output_h)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))

                ct = np.array([x, y], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                draw_umich_gaussian(hm[0], ct_int, radius)

                reg_mask[0, ct_int[1], ct_int[0]] = 1
                reg[:, ct_int[1], ct_int[0]] = ct - ct_int
                wh[0, ct_int[1], ct_int[0]] = w
                wh[1, ct_int[1], ct_int[0]] = h
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

                gt[k, 0] = ann['pitch']
                gt[k, 1] = ann['yaw']
                gt[k, 2] = ann['roll']
                gt[k, 3:5] = convert_2d_to_3d(ann['x'] * width / self.input_w, ann['y'] * height / self.input_h, ann['z'])
                gt[k, 5] = ann['z']
                gt[k, 6] = 1

            if self.lhalf:
                img = img[:, self.input_h // 2:]
                mask = mask[:, self.output_h // 2:]
                hm = hm[:, self.output_h // 2:]
                reg_mask = reg_mask[:, self.output_h // 2:]
                reg = reg[:, self.output_h // 2:]
                wh = wh[:, self.output_h // 2:]
                depth = depth[:, self.output_h // 2:]
                eular = eular[:, self.output_h // 2:]
                trig = trig[:, self.output_h // 2:]
                quat = quat[:, self.output_h // 2:]

        else:
            index -= len(self.img_paths)

            img_path, mask_path = self.test_img_paths[index], self.test_mask_paths[index]
            output = self.test_outputs[os.path.splitext(os.path.basename(img_path))[0]]

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

                if self.lhalf:
                    img = img[:, self.input_h // 2:]
                    mask = mask[:, self.output_h // 2:]

                return {
                    'img_path': img_path,
                    'input': img,
                    'mask': mask,
                }

            img = img.astype('float32') / 255
            img = (img - self.mean) / self.std
            img = img.transpose(2, 0, 1)

            mask = mask[None, ...]

            hm = np.zeros((1, self.output_h // 2, self.output_w), dtype=np.float32)
            reg_mask = np.zeros((1, self.output_h // 2, self.output_w), dtype=np.float32)
            reg = np.zeros((2, self.output_h // 2, self.output_w), dtype=np.float32)
            wh = np.zeros((2, self.output_h // 2, self.output_w), dtype=np.float32)
            depth = np.zeros((1, self.output_h // 2, self.output_w), dtype=np.float32)
            eular = np.zeros((3, self.output_h // 2, self.output_w), dtype=np.float32)
            trig = np.zeros((6, self.output_h // 2, self.output_w), dtype=np.float32)
            quat = np.zeros((4, self.output_h // 2, self.output_w), dtype=np.float32)
            gt = np.zeros((self.max_objs, 7), dtype=np.float32)

            hm = torch.sigmoid(output['hm']).numpy()[0]
            reg_mask = hm
            reg = output['reg'].numpy()[0]
            wh = output['wh'].numpy()[0]
            depth = output['depth'].numpy()[0]
            eular = eular if output['eular'] is None else output['eular'].numpy()[0]
            trig = trig if output['trig'] is None else output['trig'].numpy()[0]
            quat = quat if output['quat'] is None else output['quat'].numpy()[0]

            if self.lhalf:
                img = img[:, self.input_h // 2:]
                mask = mask[:, self.output_h // 2:]

        ret = {
            'img_path': img_path,
            'input': img,
            'mask': mask,
            # 'label': label,
            'hm': hm,
            'reg_mask': reg_mask,
            'reg': reg,
            'wh': wh,
            'depth': depth,
            'eular': eular,
            'trig': trig,
            'quat': quat,
            'gt': gt,
        }

        # plt.imshow(ret['hm'][0])
        # plt.show()
        # img = visualize(((img.transpose(1, 2, 0) * self.std + self.mean) * 255).astype('uint8'),
        #                 gt[gt[:, -1] > 0], scale_w=self.input_w / width, scale_h=self.input_h / height)
        # plt.imshow(img)
        # plt.show()

        return ret

    def __len__(self):
        if self.test_img_paths is None:
            return len(self.img_paths)
        else:
            return len(self.img_paths) + len(self.test_img_paths)
