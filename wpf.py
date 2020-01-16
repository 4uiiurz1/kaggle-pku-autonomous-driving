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

from lib.utils.utils import *
from lib.utils.vis import visualize
from lib.utils.wpf import wpf


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None)
    parser.add_argument('--models', default=None)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--score_th', default=0.3, type=float)
    parser.add_argument('--dist_th', default=2.0, type=float)
    parser.add_argument('--skip_det_th', default=0, type=float)
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()

    return args


def main():
    config = vars(parse_args())

    if config['name'] is None:
        config['name'] = 'wpf_%s' % datetime.now().strftime('%m%d%H')

    config['models'] = config['models'].split(',')

    if config['weights'] is not None:
        config['weights'] = [float(s) for s in config['weights'].split(',')]

    if not os.path.exists('models/detection/%s' % config['name']):
        os.makedirs('models/detection/%s' % config['name'])

    with open('models/detection/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    df_list = [pd.read_csv('outputs/submissions/test/%s.csv' %p).fillna('') for p in config['models']]
    new_df = pd.read_csv('inputs/sample_submission.csv')
    img_paths = np.array('inputs/test_images/' + new_df['ImageId'].values + '.jpg')

    cnt = 0
    for i in tqdm(range(len(new_df))):
        dets_list = []
        for df in df_list:
            dets_list.append(np.array(df.loc[i, 'PredictionString'].split()).reshape([-1, 7]).astype('float'))
        dets = wpf(dets_list, dist_th=config['dist_th'], skip_det_th=config['skip_det_th'],
                   weights=config['weights'])
        dets = dets[dets[:, 6] > config['score_th']]
        cnt += len(dets)

        if config['show']:
            img = cv2.imread(img_paths[i])
            img_pred = visualize(img, dets)
            plt.imshow(img_pred[..., ::-1])
            plt.show()

        new_df.loc[i, 'PredictionString'] = convert_labels_to_str(dets)

    print('Number of cars: %d' %cnt)

    new_df.to_csv('outputs/submissions/test/%s.csv' %config['name'], index=False)


if __name__ == '__main__':
    main()
