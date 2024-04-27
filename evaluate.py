import torch
from torch.utils.data import DataLoader
import pandas as pd
from dataset import DatasetOAIfor
from preproc import custom, transform
from evaluation_tools import predict_folds, merge_predictions


def generate_ds(df):
    df['path_image'] = df['path_mask'] = ''

    df['path_image'] = 'data/prep/ID_side.png'
    df['path_image'] = df.apply(
        lambda df: df['path_image'].replace('ID', str(df['ID'])), axis=1)
    df['path_image'] = df.apply(
        lambda df: df['path_image'].replace('side', df['SIDE']), axis=1)

    df['path_mask'] = 'data/prep/ID_side.png'
    df['path_mask'] = df.apply(
        lambda df: df['path_mask'].replace('ID', str(df['ID'])), axis=1)
    df['path_mask'] = df.apply(
        lambda df: df['path_mask'].replace('side', df['SIDE']), axis=1)

    return df


batch_size = 4
test_set = pd.read_csv('test_data.csv')
test_set = generate_ds(test_set)

fold_idcs = list(range(5))

# mean, std = 0, 1
mean = 90.3174
std = 83.7574

dataset_test = DatasetOAIfor(
        df_meta=test_set,
        transforms=[
            custom.Normalize(mean=mean, std=std),
            transform.ToTensor()]
        )
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8,
                                         pin_memory=torch.cuda.is_available())

folds_preds = False
merge_preds = True
with torch.no_grad():
    if folds_preds:
        predict_folds(loader=loader_test, fold_idcs=fold_idcs)

    if merge_preds:
        merge_predictions(loader=loader_test, save_plots=True)

