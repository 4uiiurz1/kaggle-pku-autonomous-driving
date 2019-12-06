import numpy as np


def dist(d1, d2, norm=False):
    dx = d1[3] - d2[3]
    dy = d1[4] - d2[4]
    dz = d1[5] - d2[5]
    diff = (dx**2 + dy**2 + dz**2)**(1/2)
    return diff


def nms(dets, dist_th=5, norm=False):
    B = dets.copy()
    print(B.shape)
    D = []
    while len(B) != 0:
        m = np.argmax(B[:, 6])
        M = B[m]
        D.append(M)
        B = np.delete(B, m, axis=0)
        rm_idx = []
        for i, b in enumerate(B):
            if dist(M, b) <= dist_th:
                rm_idx.append(i)
        B = np.delete(B, rm_idx, axis=0)

    return np.array(D)
