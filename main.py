import logging
import warnings
from collections import defaultdict

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DatasetOAIfor
from preproc import custom, transform
from seed import seed
from trainer import ModelTrainer

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

logging.basicConfig()
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
seed.set_ultimate_seed()
warnings.filterwarnings('ignore')


def generate_ds(train_df, val_df):
    train_df['path_image'] = train_df['path_mask'] = val_df['path_image'] = val_df['path_mask'] = ''

    train_df['path_image'] = 'data/prep/ID_side.png'
    train_df['path_image'] = train_df.apply(
        lambda train_df: train_df['path_image'].replace('ID', str(train_df['ID'])), axis=1)
    train_df['path_image'] = train_df.apply(
        lambda train_df: train_df['path_image'].replace('side', train_df['SIDE']), axis=1)
    train_df['path_mask'] = 'data/masks/ID_side.png'
    train_df['path_mask'] = train_df.apply(
        lambda train_df: train_df['path_mask'].replace('ID', str(train_df['ID'])), axis=1)
    train_df['path_mask'] = train_df.apply(
        lambda train_df: train_df['path_mask'].replace('side', train_df['SIDE']), axis=1)

    val_df['path_image'] = 'data/prep/ID_side.png'
    val_df['path_image'] = val_df.apply(
        lambda val_df: val_df['path_image'].replace('ID', str(val_df['ID'])), axis=1)
    val_df['path_image'] = val_df.apply(
        lambda val_df: val_df['path_image'].replace('side', val_df['SIDE']), axis=1)
    val_df['path_mask'] = 'data/masks/ID_side.png'
    val_df['path_mask'] = val_df.apply(
        lambda val_df: val_df['path_mask'].replace('ID', str(val_df['ID'])), axis=1)
    val_df['path_mask'] = val_df.apply(
        lambda val_df: val_df['path_mask'].replace('side', val_df['SIDE']), axis=1)

    return train_df, val_df


def estimate_mean_std(dataset):
    mean_std_loader = DataLoader(dataset, batch_size=16, num_workers=16, pin_memory=torch.cuda.is_available())
    len_inputs = len(mean_std_loader.sampler)
    mean = 0
    std = 0
    for sample in tqdm(mean_std_loader, desc='Computing mean and std values:'):
        local_batch, local_labels = sample['xs'], sample['ys']

        for j in range(local_batch.shape[0]):
            mean += local_batch.float()[j, 0, :, :].mean()
            std += local_batch.float()[j, 0, :, :].std()

    mean /= len_inputs
    std /= len_inputs

    return mean, std


batch_size = 16
fold_scores = dict()
name_ds = 'OAI'
data = pd.read_csv('train_data.csv')
for fold_num in range(5):
    logger.info(f'Training fold {fold_num}')

    train_index = pd.read_csv('data/train{}.csv'.format(str(fold_num)))
    val_index = pd.read_csv('data/val{}.csv'.format(str(fold_num)))
    val_set, train_set = data.iloc[val_index.values.flatten()], data.iloc[train_index.values.flatten()]

    train_set, val_set = generate_ds(train_set, val_set)

    datasets = defaultdict(dict)

    mean = 90.3174
    std = 83.7574

    datasets[name_ds]['train'] = DatasetOAIfor(
        df_meta=train_set,
        transforms=[
            transform.HorizontalFlip(prob=.5),
            # transform.GammaCorrection(gamma_range=(0.5, 1.5), prob=.5),
            transform.OneOf([
                transform.DualCompose([
                    transform.Scale(ratio_range=(0.7, 0.8), prob=1.),
                    transform.Scale(ratio_range=(1.5, 1.6), prob=1.),
                ]),
                transform.NoTransform()
            ]),
            transform.Crop(output_size=(385, 245)),
            custom.Normalize(mean=mean, std=std),
            transform.ToTensor()]
        )
    datasets[name_ds]['val'] = DatasetOAIfor(
        df_meta=val_set,
        transforms=[
            transform.CenterCrop(height=385, width=245),
            custom.Normalize(mean=mean, std=std),
            transform.ToTensor()]
        )

    loaders = defaultdict(dict)
    loaders[name_ds]['train'] = DataLoader(datasets[name_ds]['train'], batch_size=batch_size, shuffle=False,
                                           num_workers=8, pin_memory=torch.cuda.is_available())
    loaders[name_ds]['val'] = DataLoader(datasets[name_ds]['val'], batch_size=batch_size, shuffle=False, num_workers=8,
                                         pin_memory=torch.cuda.is_available())

    trainer = ModelTrainer(fold_idx=fold_num)

    tmp = trainer.fit(loaders=loaders)
    metrics_train, fnames_train, metrics_val, fnames_val = tmp

    fold_scores[fold_num] = (metrics_val['datasetw'][f'{name_ds}__dice_score'],)

    trainer.tensorboard.close()
logger.info(f'Fold scores:\n{repr(fold_scores)}')


