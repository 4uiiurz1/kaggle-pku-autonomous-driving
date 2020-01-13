import numpy as np
from .image import get_bbox


def bb_intersection_over_union(A, B):
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def calc_iou(det1, det2):
    bbox1 = get_bbox(*det1[:6], 1, 1, 1, 1)
    bbox2 = get_bbox(*det2[:6], 1, 1, 1, 1)
    print(bbox1, bbox2)

    return bb_intersection_over_union(bbox1, bbox2)


def get_weighted_det(dets, conf_type='avg'):
    det = np.zeros(7, dtype='float32')
    conf = 0
    conf_list = []
    for d in dets:
        det[:6] += (d[6] * d[:6])
        conf += d[6]
        conf_list.append(d[6])
    if conf_type == 'avg':
        det[6] = conf / len(dets)
    elif conf_type == 'max':
        det[6] = np.array(conf_list).max()
    det[:6] /= conf
    return det


def find_matching_det(dets, new_det, match_dist):
    best_iou = 0.30
    best_index = -1
    for i in range(len(dets)):
        det = dets[i]
        iou = calc_iou(det, new_det)
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def wpf(dets_list, weights=[0.3, 0.7], dist_th=1, skip_det_thr=0.0, conf_type='avg', allows_overflow=False):
    if weights is None:
        weights = np.ones(len(dets_list))
    if len(weights) != len(dets_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(dets_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    dets = []
    for i, weight in enumerate(weights):
        dets.append(dets_list[i].copy())
        dets[i] = dets[i][dets[i][:, 6] > skip_det_thr]
        dets[i][:, 6] *= weight
    dets = np.vstack(dets)
    dets = dets[dets[:, 6].argsort()[::-1]]
    if len(dets) == 0:
        return np.zeros((0, 7))

    new_dets = []
    weighted_dets = []

    # Clusterize dets
    for i in range(len(dets)):
        index, best_iou = find_matching_det(weighted_dets, dets[i], dist_th)
        if index != -1:
            new_dets[index].append(dets[i])
            weighted_dets[index] = get_weighted_det(new_dets[index], conf_type)
        else:
            new_dets.append([dets[i].copy()])
            weighted_dets.append(dets[i].copy())
    weighted_dets = np.array(weighted_dets)

    # Rescale confidence based on number of models and dets
    for i in range(len(new_dets)):
        if not allows_overflow:
            weighted_dets[i][6] = weighted_dets[i][6] * min(weights.sum(), len(new_dets[i])) / weights.sum()
        else:
            weighted_dets[i][6] = weighted_dets[i][6] * len(new_dets[i]) / weights.sum()

    weighted_dets = weighted_dets[weighted_dets[:, 6].argsort()[::-1]]

    return weighted_dets
