import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.utils import convert_2d_to_3d, convert_3d_to_2d
from .utils.utils import rotate


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def convert_quat_to_euler(qx, qy, qz, qw):
    t0 = 2.0 * (qw * qx + qy * qz)
    t1 = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(t0, t1)

    t2 = 2.0 * (qw * qy - qz * qx)
    t2 = torch.clamp(t2, -1, 1)
    pitch = torch.asin(t2)

    t3 = 2.0 * (qw * qz + qx * qy)
    t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(t3, t4)

    return yaw, pitch, roll


def decode(hm, reg, depth, eular=None, trig=None, quat=None, mask=None, K=100,
           org_width=3384, org_height=2710):
    batch, cat, height, width = hm.size()

    hm = nms(torch.sigmoid(hm))
    if mask is not None:
        hm *= mask

    depth = 1. / (torch.sigmoid(depth) + 1e-6) - 1.

    scores, inds, clses, ys, xs = _topk(hm, K=K)
    scores = scores.view(batch, K, 1)

    reg = _tranpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

    zs = _tranpose_and_gather_feat(depth, inds)
    zs = zs.view(batch, K, 1)

    xs /= width
    ys /= height
    xs *= org_width
    ys *= org_height
    xs, ys = convert_2d_to_3d(xs, ys, zs)

    if eular is not None:
        eular = _tranpose_and_gather_feat(eular, inds)
        yaw, pitch, roll = eular[..., 0:1], eular[..., 1:2], eular[..., 2:3]
        roll = rotate(roll, -np.pi)
    elif trig is not None:
        trig = _tranpose_and_gather_feat(trig, inds)
        yaw = torch.atan2(trig[..., 1:2], trig[..., 0:1])
        pitch = torch.atan2(trig[..., 3:4], trig[..., 2:3])
        roll = torch.atan2(trig[..., 5:6], trig[..., 4:5])
        roll = rotate(roll, -np.pi)
    elif quat is not None:
        quat = _tranpose_and_gather_feat(quat, inds)
        yaw, pitch, roll = convert_quat_to_euler(
            quat[..., 0:1], quat[..., 1:2], quat[..., 2:3], quat[..., 3:4])

    yaw = yaw.view(batch, K, 1)
    pitch = pitch.view(batch, K, 1)
    roll = roll.view(batch, K, 1)

    dets = torch.cat([scores, yaw, pitch, roll, xs, ys, zs], dim=2)

    return dets
