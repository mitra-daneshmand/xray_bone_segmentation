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
from torch.utils.data.dataloader import DataLoader

from model import UNetLext
from components import checkpoint
from preproc import transforms
from seed import seed

# The fix is a workaround to PyTorch multiprocessing issue:
# "RuntimeError: received 0 items of ancdata"
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
        model = UNetLext()
        model = nn.DataParallel(model).to(maybe_gpu)
        model.eval()

        with tqdm(total=len(loader), desc=f'Eval, fold {fold_idx}') as prog_bar:
            for i, data_batch in enumerate(loader):
                xs, ys_true = data_batch['xs'], data_batch['ys']
                xs, ys_true = xs.to(maybe_gpu), ys_true.to(maybe_gpu)

                ys_pred = model(xs)

                ys_pred_softmax = nn.Softmax(dim=1)(ys_pred)
                ys_pred_softmax_np = ys_pred_softmax.detach().to('cpu').numpy()

                data_batch['pred_softmax'] = ys_pred_softmax_np

                # Rearrange the batch
                data_dicts = [{k: v[n] for k, v in data_batch.items()}
                              for n in range(len(data_batch['image']))]

                for k, data_dict in enumerate(data_dicts):
                    dir_base = os.path.join('predictions', str(data_dict['ID']))
                    fname_base = os.path.splitext(os.path.basename(data_dict['path_image']))[0]

                    # Save the predictions
                    dir_predicts = os.path.join(dir_base, 'mask_folds')
                    if not os.path.exists(dir_predicts):
                        os.makedirs(dir_predicts)

                    fname_full = os.path.join(dir_predicts, f'{fname_base}_fold_{fold_idx}.tiff')

                    tmp = (data_dict['pred_softmax'] * 255).astype(np.uint8, casting='unsafe')
                    tifffile.imwrite(fname_full, tmp)

                prog_bar.update(1)


def merge_predictions(loader, save_plots=False, remove_foldw=False):
    """Merge the predictions over all folds
    """
    dir_source_root = ''
    df_meta = loader.dataset.df_meta

    with tqdm(total=len(df_meta), desc='Merge') as prog_bar:
        for i, row in df_meta.iterrows():
            dir_scan_predicts = os.path.join('predictions', str(row['ID']))
            dir_image_prep = os.path.join(dir_scan_predicts, 'image_prep')
            dir_mask_prep = os.path.join(dir_scan_predicts, 'mask_prep')
            dir_mask_folds = os.path.join(dir_scan_predicts, 'mask_folds')
            dir_mask_foldavg = os.path.join(dir_scan_predicts, 'mask_foldavg')
            dir_vis_foldavg = os.path.join(dir_scan_predicts, 'vis_foldavg')

            for p in (dir_image_prep, dir_mask_prep, dir_mask_folds, dir_mask_foldavg,
                      dir_vis_foldavg):
                if not os.path.exists(p):
                    os.makedirs(p)

            # Find the corresponding prediction files
            fname_base = os.path.splitext(os.path.basename(row['path_image']))[0]

            fnames_pred = glob(os.path.join(dir_mask_folds, f'{fname_base}_fold_*.*'))

            # Read the reference data
            image = cv2.imread(os.path.join(dir_source_root, row['path_image']), cv2.IMREAD_GRAYSCALE)
            image = transforms.CenterCrop(height=355, width=215)(image[None, ])[0]  # dict_fns['crop'](image[None, ])[0]
            image = np.squeeze(image)
            if 'path_mask' in row.index:
                ys_true = loader.dataset.read_mask(
                    os.path.join(dir_source_root, row['path_mask']))
                if ys_true is not None:
                    ys_true = transforms.CenterCrop(height=355, width=215)(ys_true)[0]  # dict_fns['crop'](ys_true)[0]
            else:
                ys_true = None

            # Read the fold-wise predictions
            yss_pred = [tifffile.imread(f) for f in fnames_pred]
            ys_pred = np.stack(yss_pred, axis=0).astype(np.float32) / 255
            ys_pred = torch.from_numpy(ys_pred).unsqueeze(dim=0)

            # Average the fold predictions
            ys_pred = torch.mean(ys_pred, dim=1, keepdim=False)
            ys_pred_softmax = ys_pred / torch.sum(ys_pred, dim=1, keepdim=True)
            ys_pred_softmax_np = ys_pred_softmax.squeeze().numpy()

            ys_pred_arg_np = ys_pred_softmax_np.argmax(axis=0)

            # Save preprocessed input data
            fname_full = os.path.join(dir_image_prep, f'{fname_base}.png')
            cv2.imwrite(fname_full, image)  # image

            if ys_true is not None:
                ys_true = ys_true.astype(np.float32)
                ys_true = torch.from_numpy(ys_true).unsqueeze(dim=0)
                ys_true_arg_np = ys_true.numpy().squeeze().argmax(axis=0)
                fname_full = os.path.join(dir_mask_prep, f'{fname_base}.png')
                cv2.imwrite(fname_full, ys_true_arg_np)  # mask

            fname_meta = os.path.join('predictions', 'meta_dynamic.csv')
            if not os.path.exists(fname_meta):
                df_meta.to_csv(fname_meta, index=False)  # metainfo

            # Save ensemble prediction
            fname_full = os.path.join(dir_mask_foldavg, f'{fname_base}.png')
            cv2.imwrite(fname_full, ys_pred_arg_np)

            # Save ensemble visualizations
            if save_plots:
                if ys_true is not None:
                    fname_full = os.path.join(
                        dir_vis_foldavg, f"{fname_base}_overlay_mask.png")
                    save_vis_overlay(image=image,
                                     mask=ys_true_arg_np,
                                     num_classes=1,
                                     fname=fname_full)

                fname_full = os.path.join(
                    dir_vis_foldavg, f"{fname_base}_overlay_pred.png")
                save_vis_overlay(image=image,
                                 mask=ys_pred_arg_np,
                                 num_classes=1,
                                 fname=fname_full)

                if ys_true is not None:
                    fname_full = os.path.join(
                        dir_vis_foldavg, f"{fname_base}_overlay_diff.png")
                    save_vis_mask_diff(image=image,
                                       mask_true=ys_true_arg_np,
                                       mask_pred=ys_pred_arg_np,
                                       fname=fname_full)

            # Remove the fold predictions
            if remove_foldw:
                for f in fnames_pred:
                    try:
                        os.remove(f)
                    except OSError:
                        logger.error(f'Cannot remove {f}')
            prog_bar.update(1)


def save_vis_overlay(image, mask, num_classes, fname):
    # Add a sample of each class to have consistent class colors
    mask[0, :num_classes] = list(range(num_classes))
    overlay = label2rgb(label=mask, image=image, bg_label=0,
                        colors=['orangered', 'gold', 'lime', 'fuchsia'])
    # Convert to uint8 to save space
    overlay = img_as_ubyte(overlay)
    # Save to file
    if overlay.ndim == 3:
        overlay = overlay[:, :, ::-1]
    cv2.imwrite(fname, overlay)


def save_vis_mask_diff(image, mask_true, mask_pred, fname):
    diff = np.empty_like(mask_true)
    diff[(mask_true == mask_pred) & (mask_pred == 0)] = 0  # TN
    diff[(mask_true == mask_pred) & (mask_pred != 0)] = 0  # TP
    diff[(mask_true != mask_pred) & (mask_pred == 0)] = 2  # FP
    diff[(mask_true != mask_pred) & (mask_pred != 0)] = 3  # FN
    diff_colors = ('green', 'red', 'yellow')
    diff[0, :4] = [0, 1, 2, 3]
    overlay = label2rgb(label=diff, image=image, bg_label=0,
                        colors=diff_colors)
    # Convert to uint8 to save space
    overlay = img_as_ubyte(overlay)
    # Save to file
    if overlay.ndim == 3:
        overlay = overlay[:, :, ::-1]
    cv2.imwrite(fname, overlay)

