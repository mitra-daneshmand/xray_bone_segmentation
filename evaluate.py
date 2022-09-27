import os
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataset import DatasetOAIfor
from preproc import custom, transforms
from evaluation_tools import predict_folds, merge_predictions


def generate_ds(df):
    df['path_image'] = df['path_mask'] = ''

    df['path_image'] = 'data/imgs/ID_side.png'
    df['path_image'] = df.apply(
        lambda df: df['path_image'].replace('ID', str(df['ID'])), axis=1)
    df['path_image'] = df.apply(
        lambda df: df['path_image'].replace('side', df['SIDE']), axis=1)

    df['path_mask'] = 'data/imgs/ID_side.png'
    df['path_mask'] = df.apply(
        lambda df: df['path_mask'].replace('ID', str(df['ID'])), axis=1)
    df['path_mask'] = df.apply(
        lambda df: df['path_mask'].replace('side', df['SIDE']), axis=1)

    return df



batch_size = 16
test_set = pd.read_csv('test_data.csv')
test_set = generate_ds(test_set)

fold_idcs = list(range(5))

mean, std = 8.32237651, 6.92832947


dataset_test = DatasetOAIfor(
        df_meta=test_set,
        transforms=[
            # custom.PercentileClippingAndToFloat(cut_min=10, cut_max=30),
            transforms.CenterCrop(height=355, width=215),
            custom.Normalize(mean=mean, std=std),
            transforms.ToTensor()
        ])
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8,
                                         pin_memory=torch.cuda.is_available())

folds_preds = True
merge_preds = False
with torch.no_grad():
    if folds_preds:
        predict_folds(loader=loader_test, fold_idcs=fold_idcs)

    if merge_preds:
        merge_predictions(loader=loader_test, save_plots=True, remove_foldw=False)

