import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
from tqdm import tqdm


def calc_image_hash(img_path):
    with Image.open(img_path) as img:
        img_hash = imagehash.dhash(img)
        return img_hash


def main():
    df = pd.read_csv('inputs/train.csv')
    test_df = pd.read_csv('inputs/sample_submission.csv')

    df['hash'] = np.nan
    for i in tqdm(range(len(df))):
        df.loc[i, 'hash'] = calc_image_hash('inputs/train_images/' + df.loc[i, 'ImageId'] + '.jpg')

    test_df['hash'] = np.nan
    for i in tqdm(range(len(test_df))):
        test_df.loc[i, 'hash'] = calc_image_hash('inputs/test_images/' + test_df.loc[i, 'ImageId'] + '.jpg')

    df.to_csv('processed/train_image_hash.csv', index=False)
    test_df.to_csv('processed/test_image_hash.csv', index=False)


if __name__ == '__main__':
    main()
