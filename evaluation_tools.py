import os
import logging
from glob import glob

import numpy as np
from skimage.color import label2rgb
from skimage import img_as_ubyte
from tqdm import tqdm

import cv2
import tifffile
import torch
import torch.nn as nn

from model import UNetLext
from components import checkpoint
from seed import seed


torch.multiprocessing.set_sharing_strategy('file_system')

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

logging.basicConfig()
logger = logging.getLogger('eval')
logger.setLevel(logging.INFO)

seed.set_ultimate_seed()

if torch.cuda.is_available():
    maybe_gpu = 'cuda'
else:
    maybe_gpu = 'cpu'


def predict_folds(loader, fold_idcs):
    """Evaluate the model versus each fold
    """
    for fold_idx in fold_idcs:
        paths_weights_fold = dict()
        paths_weights_fold['segm'] = os.path.join('sessions', 'segm', f'fold_{fold_idx}')

        handlers_ckpt = dict()
        handlers_ckpt['segm'] = checkpoint.CheckpointHandler(paths_weights_fold['segm'])

        paths_ckpt_sel = dict()
        paths_ckpt_sel['segm'] = handlers_ckpt['segm'].get_last_ckpt()

        # Initialize and configure the model
        model = UNetLext(input_channels=1,
                         output_channels=3,
                         center_depth=1,
                         pretrained=True,
                         path_pretrained=paths_ckpt_sel['segm'],
                         restore_weights=True,
                         path_weights=paths_ckpt_sel['segm']
                         )
        model = nn.DataParallel(model).to(maybe_gpu)
        model.eval()

        with tqdm(total=len(loader), desc=f'Eval, fold {fold_idx}') as prog_bar:
            for i, data_batch in enumerate(loader):
                xs, ys_true = data_batch['xs'], data_batch['ys']
                xs, ys_true = xs.to(maybe_gpu), ys_true.to(maybe_gpu)

                ys_pred = model(xs)

                ys_pred_softmax = nn.Softmax(dim=1)(ys_pred)
                ys_pred_softmax_np = ys_pred_softmax.to('cpu').numpy()

                data_batch['pred_softmax'] = ys_pred_softmax_np

                # Rearrange the batch
                data_dicts = [{k: v[n] for k, v in data_batch.items()}
                              for n in range(len(data_batch['image']))]

                for k, data_dict in enumerate(data_dicts):
                    dir_base = os.path.join('predictions', str(data_dict['ID'])[-8:-1])
                    fname_base = os.path.splitext(os.path.basename(data_dict['path_image']))[0]

                    # Save the predictions
                    dir_predicts = os.path.join(dir_base, 'mask_folds')
                    if not os.path.exists(dir_predicts):
                        os.makedirs(dir_predicts)

                    fname_full = os.path.join(
                        dir_predicts,
                        f'{fname_base}_fold_{fold_idx}.tiff')

                    tmp = (data_dict['pred_softmax'] * 255).astype(np.uint8, casting='unsafe')
                    tifffile.imsave(fname_full, tmp)

                prog_bar.update(1)


def merge_predictions(loader, save_plots=False):
    """Merge the predictions over all folds
    """
    dir_source_root = ''
    df_meta = loader.dataset.df_meta

    with tqdm(total=len(df_meta), desc='Merge') as prog_bar:
        for i, row in df_meta.iterrows():
            dir_scan_predicts = os.path.join('predictions', str(row['ID']))
            dir_mask_folds = os.path.join(dir_scan_predicts, 'mask_folds')
            dir_mask_foldavg = os.path.join(dir_scan_predicts, 'mask_foldavg')
            dir_vis_foldavg = os.path.join(dir_scan_predicts, 'vis_foldavg')

            for p in (dir_mask_folds, dir_mask_foldavg, dir_vis_foldavg):
                if not os.path.exists(p):
                    os.makedirs(p)

            # Find the corresponding prediction files
            fname_base = os.path.splitext(os.path.basename(row['path_image']))[0]
            fnames_pred = glob(os.path.join(dir_mask_folds, f'{fname_base}_fold_*.*'))

            # Read the reference data
            image = cv2.imread(os.path.join(dir_source_root, row['path_image']), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (245, 385))

            # Read the fold-wise predictions
            yss_pred = [tifffile.imread(f) for f in fnames_pred]
            ys_pred = np.stack(yss_pred, axis=0).astype(np.float32) / 255
            ys_pred = torch.from_numpy(ys_pred).unsqueeze(dim=0)

            # Average the fold predictions
            ys_pred = torch.mean(ys_pred, dim=1, keepdim=False)
            ys_pred_softmax = ys_pred / torch.sum(ys_pred, dim=1, keepdim=True)
            ys_pred_softmax_np = ys_pred_softmax.squeeze().numpy()

            ys_pred_arg_np = ys_pred_softmax_np.argmax(axis=0)

            fname_meta = os.path.join('predictions', 'meta_dynamic.csv')
            if not os.path.exists(fname_meta):
                df_meta.to_csv(fname_meta, index=False)  # metainfo

            # Save ensemble prediction
            fname_full = os.path.join(dir_mask_foldavg, f'{fname_base}.png')
            cv2.imwrite(fname_full, ys_pred_arg_np)

            # Save ensemble visualizations
            if save_plots:
                fname_full = os.path.join(
                    dir_vis_foldavg, f"{fname_base}_overlay_pred.png")
                save_vis_overlay(image=image,
                                 mask=ys_pred_arg_np,
                                 fname=fname_full)
            prog_bar.update(1)


def save_vis_overlay(image, mask, fname):
    overlay = label2rgb(label=mask, image=image, bg_label=0,
                        colors=['lime', 'dodgerblue'])
    # Convert to uint8 to save space
    overlay = img_as_ubyte(overlay)
    # Save to file
    if overlay.ndim == 3:
        overlay = overlay[:, :, ::-1]
    cv2.imwrite(fname, overlay)



