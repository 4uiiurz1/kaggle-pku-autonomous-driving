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
                 hflip=0, scale=0, scale_limit=0, car_spec_bbox=False,
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
        self.car_spec_bbox = car_spec_bbox
        self.car_sizes = [
            [0.9632981899999998, 0.7136288450000001, 1.9817771899999999],
            [1.107421, 0.8352809149999999, 2.19842186],
            [1.051964985, 0.7414249599999999, 2.134020005],
            [0.90897132, 0.738931525, 2.24773888],
            [0.942159235, 0.7322307299999999, 1.724695815],
            [0.9474815749999999, 0.89117324, 1.858441315],
            [1.00949707, 0.72244968, 2.19623657],
            [0.944831655, 0.718985865, 1.949793545],
            [0.9797444150000001, 0.825776405, 2.2137257399999997],
            [1.05457397, 0.9467736800000001, 2.396790775],
            [0.9306240100000001, 0.7779002550000002, 1.870461885],
            [0.96996384, 0.72374977, 1.79419418],
            [0.95508354, 0.7606044399999999, 1.8846455400000002],
            [1.00134636, 0.794314865, 2.041832045],
            [0.9405763250000001, 0.72577158, 1.9684198000000002],
            [0.86639252, 0.7552054050000001, 2.138031315],
            [0.9436828650000001, 0.6797149650000001, 2.15702667],
            [1.0505274800000002, 0.755187425, 2.613266945],
            [1.1874128, 0.83430786, 2.651320955],
            [1.074316405, 0.769780235, 2.494289855],
            [1.05809052, 0.7118933849999999, 2.487118835],
            [0.99840698, 0.71926383, 2.242857815],
            [1.04000145, 0.746722305, 2.40301361],
            [1.191761245, 0.8343393349999999, 2.779609375],
            [1.006286735, 0.72953824, 2.2549543799999996],
            [1.03298683, 0.745325545, 2.4731579549999996],
            [0.993705025, 0.73760435, 2.260355375],
            [0.983911745, 0.8010069300000001, 2.5674130249999996],
            [1.06183548, 0.7393771749999999, 2.486589355],
            [0.9123247449999999, 0.73765439, 2.29781015],
            [1.029463195, 0.7485418699999999, 2.273908155],
            [1.0438732899999998, 0.7366547800000001, 2.41565155],
            [1.20607063, 0.84150093, 2.7963800049999996],
            [1.03924639, 0.70758189, 2.409076415],
            [1.00701599, 0.7330542950000001, 2.38695633],
            [1.0113480400000001, 0.760608445, 2.4234931950000003],
            [1.02196514, 0.7210548449999998, 2.34112777],
            [1.04407776, 0.735242425, 2.498537445],
            [0.8905402800000001, 0.7124821149999999, 2.12523112],
            [0.959388005, 0.70339981, 2.45490692],
            [1.06404606, 0.710192455, 2.4109755699999997],
            [0.9869781850000001, 0.7448342199999999, 2.41504672],
            [0.9850992550000001, 0.763697435, 2.315551755],
            [1.012997245, 0.7285227599999999, 2.411222225],
            [0.8665149299999999, 0.70484667, 2.20135353],
            [1.13473343, 0.876485215, 2.40518509],
            [1.0678626249999998, 0.8982450900000001, 2.404490055],
            [1.13261795, 0.91300072, 2.642370605],
            [1.12635414, 0.88069916, 2.527064895],
            [0.9579540249999999, 0.8225663299999999, 2.082297365],
            [1.141249695, 0.871034545, 2.42393944],
            [1.1369170750000002, 0.9121536650000001, 2.67260696],
            [1.01158966, 0.84216837, 2.237540285],
            [1.00638557, 0.8449507900000001, 2.26701233],
            [1.08348011, 0.9254894300000001, 2.417293545],
            [0.9441433699999999, 0.9316225850000001, 2.524690245],
            [1.050550005, 0.8534145000000001, 2.257372895],
            [0.9437686149999999, 0.9316225850000001, 2.5256304949999997],
            [0.8889313799999999, 0.83441705, 2.179602125],
            [1.021995805, 0.8518500499999999, 2.23406461],
            [1.11875465, 0.89044697, 2.43236145],
            [1.04746582, 0.8500468449999999, 2.26505402],
            [1.0735009, 0.914662095, 2.326362305],
            [1.0004706250000002, 0.87646063, 2.211697665],
            [1.040797865, 0.87943635, 2.282441315],
            [1.048445665, 0.849516715, 2.32331192],
            [1.0835810449999999, 0.8120470399999999, 2.531510925],
            [0.9297848500000001, 0.78797222, 1.996817015],
            [0.999731295, 0.84199806, 2.141316985],
            [1.141052855, 0.9139129249999999, 2.5189808649999996],
            [1.091564025, 0.8702596650000001, 2.4774002050000004],
            [1.085014265, 0.879189415, 2.41231895],
            [1.053022615, 0.839972345, 2.366754305],
            [1.0007104150000001, 0.82414761, 2.08551117],
            [1.0420876350000001, 0.84661125, 2.2799122599999997],
            [1.065111465, 0.83470737, 2.244401475],
            [1.128580245, 0.869005665, 2.347976685],
            [0.93265957, 0.8717143150000001, 2.26743584],
            [1.0462940600000001, 0.908272345, 2.302528305]
        ]

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

                if self.car_spec_bbox:
                    bbox = get_bbox(
                        ann['yaw'],
                        ann['pitch'],
                        ann['roll'],
                        *convert_2d_to_3d(ann['x'] * width / self.input_w, ann['y'] * height / self.input_h, ann['z']),
                        ann['z'],
                        width,
                        height,
                        self.output_w,
                        self.output_h,
                        *self.car_sizes[ann['model_type']])
                else:
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
                norm = (qx**2 + qy**2 + qz**2 + qw**2)**(1 / 2)
                quat[0, ct_int[1], ct_int[0]] = qx / norm
                quat[1, ct_int[1], ct_int[0]] = qy / norm
                quat[2, ct_int[1], ct_int[0]] = qz / norm
                quat[3, ct_int[1], ct_int[0]] = qw / norm

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


class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform=None, masks=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.masks = masks

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]
        if self.masks is not None:
            mask = self.masks[index]

        img = cv2.imread(img_path)
        if img is None:
            print('%s does not exist' %img_path)
            img = np.zeros((224, 224, 3), 'uint8')

        if self.transform is not None:
            img = self.transform(image=img)['image']

        if self.masks is None:
            return img, label
        else:
            return img, label, mask

    def __len__(self):
        return len(self.img_paths)
