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

from lib.utils.vis import visualize


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None)
    parser.add_argument('--uncropped', action='store_true')
    parser.add_argument('--write', action='store_true')


    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    df = pd.read_csv('outputs/submissions/test/%s.csv' %args.name).fillna('')
    img_ids = df['ImageId'].values
    img_paths = np.array('inputs/test_images/' + df['ImageId'].values + '.jpg')
    if args.uncropped:
        cropped_img_ids = pd.read_csv('inputs/testset_cropped_imageids.csv')['ImageId'].values
        for i, img_id in enumerate(img_ids):
            if img_id in cropped_img_ids:
                img_paths[i] = 'inputs/test_images_uncropped/' + img_id + '.jpg'

    os.makedirs(os.path.join('tmp', args.name), exist_ok=True)
    for i in tqdm(range(len(df))):
        dets = np.array(df.loc[i, 'PredictionString'].split()).reshape([-1, 7]).astype('float')

        img = cv2.imread(img_paths[i])
        img_pred = visualize(img, dets)
        if not args.write:
            plt.imshow(img_pred[..., ::-1])
            plt.show()
        else:
            cv2.imwrite(os.path.join('tmp', args.name, os.path.basename(img_paths[i])), img_pred)


if __name__ == '__main__':
    main()
