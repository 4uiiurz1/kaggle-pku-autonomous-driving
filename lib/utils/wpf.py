import numpy as np


def get_bbox(yaw, pitch, roll, x, y, z,
             car_hw=1.02, car_hh=0.80, car_hl=2.31):
    Rt = np.eye(4)
    t = np.array([x, y, z])
    Rt[:3, 3] = t
    Rt[:3, :3] = euler_to_Rot(-yaw, -pitch, -roll).T
    Rt = Rt[:3, :]
    P = np.array([
        [-car_hw,       0,       0, 1],
        [ car_hw,       0,       0, 1],
        [      0,  car_hh,       0, 1],
        [      0, -car_hh,       0, 1],
        [      0,       0,  car_hl, 1],
        [      0,       0, -car_hl, 1],
    ]).T
    P = Rt @  P
    P = P.T
    xs, ys = convert_3d_to_2d(P[:, 0], P[:, 1], P[:, 2])
    bbox = [xs.min(), ys.min(), xs.max(), ys.max()]

    return bbox


def calc_dist(d1, d2, norm=False):
    dx = d1[3] - d2[3]
    dy = d1[4] - d2[4]
    dz = d1[5] - d2[5]
    diff = (dx**2 + dy**2 + dz**2)**(1/2)
    return diff


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
    best_dist = match_dist
    best_index = -1
    for i in range(len(dets)):
        det = dets[i]
        dist = calc_dist(det, new_det, norm=False)
        if dist < best_dist:
            best_index = i
            best_dist = dist
    print(best_dist)

    return best_index, best_dist


def wpf(dets_list, weights=None, dist_th=1, skip_det_th=0.0, conf_type='avg', allows_overflow=False):
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
        dets[i] = dets[i][dets[i][:, 6] > skip_det_th]
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
