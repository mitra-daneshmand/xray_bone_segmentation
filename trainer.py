import os
import logging
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from seed import seed
from components import checkpoint, metrics, losses
from model import UNetLext


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

logging.basicConfig()
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
seed.set_ultimate_seed()

if torch.cuda.is_available():
    maybe_gpu = 'cuda'
else:
    maybe_gpu = 'cpu'


class ModelTrainer:
    def __init__(self, fold_idx=None):
        self.fold_idx = fold_idx

        self.paths_weights_fold = dict()
        self.paths_weights_fold['segm'] = os.path.join('sessions', 'segm', f'fold_{self.fold_idx}')
        os.makedirs(self.paths_weights_fold['segm'], exist_ok=True)

        self.path_logs_fold = os.path.join('sessions', f'fold_{self.fold_idx}')
        os.makedirs(self.path_logs_fold, exist_ok=True)

        self.handlers_ckpt = dict()
        self.handlers_ckpt['segm'] = checkpoint.CheckpointHandler(self.paths_weights_fold['segm'])

        paths_ckpt_sel = dict()
        paths_ckpt_sel['segm'] = self.handlers_ckpt['segm'].get_last_ckpt()

        # Initialize and configure the models
        self.models = dict()
        self.models['segm'] = UNetLext(input_channels=1,
                                       output_channels=3,
                                       center_depth=1,
                                       pretrained=False,
                                       path_pretrained='',
                                       restore_weights=False,
                                       path_weights=paths_ckpt_sel['segm']
                                       )
        self.models['segm'] = nn.DataParallel(self.models['segm'])
        self.models['segm'] = self.models['segm'].to(maybe_gpu)

        # Configure the training
        self.num_epoch = 200
        self.optimizers = dict()
        self.optimizers['segm'] = (optim.Adam(
            self.models['segm'].parameters(),
            lr=0.0001,
            weight_decay=5e-5))

        self.lr_update_rule = {30: 0.1}

        self.losses = dict()
        self.losses['segm'] = dict()
        self.losses['segm']['dice_loss'] = losses.GeneralizedDice(idc=[1, 2])
        self.losses['segm']['boundary_loss'] = losses.BoundaryLoss(idc=[1, 2])

        self.tensorboard = SummaryWriter(self.path_logs_fold)

    def run_one_epoch(self, epoch_idx, loaders):
        name_ds = list(loaders.keys())[0]

        fnames_acc = defaultdict(list)
        metrics_acc = dict()
        metrics_acc['samplew'] = defaultdict(list)
        metrics_acc['batchw'] = defaultdict(list)
        metrics_acc['datasetw'] = defaultdict(list)
        metrics_acc['datasetw'][f'{name_ds}__cm'] = np.zeros((3,) * 2, dtype=np.uint32)

        prog_bar_params = {'postfix': {'epoch': epoch_idx}, }

        if self.models['segm'].training:
            # ------------------------ Training regime ------------------------
            loader_ds = loaders[name_ds]['train']

            steps_ds = len(loader_ds)
            prog_bar_params.update({'total': steps_ds,
                                    'desc': f'Train, epoch {epoch_idx}'})

            loader_ds_iter = iter(loader_ds)
            alpha = 0.01

            with tqdm(**prog_bar_params) as prog_bar:
                for step_idx in range(steps_ds):
                    self.optimizers['segm'].zero_grad()

                    data_batch_ds = next(loader_ds_iter)

                    xs_ds, ys_true_ds = data_batch_ds['xs'], data_batch_ds['ys']
                    fnames_acc['oai'].extend(data_batch_ds['path_image'])

                    ys_true_arg_ds = torch.argmax(ys_true_ds.long(), dim=1)
                    xs_ds = xs_ds.to(maybe_gpu)

                    ys_pred_ds = self.models['segm'](xs_ds)

                    ys_pred_softmax_ds = nn.Softmax(dim=1)(ys_pred_ds)
                    gt_sdf = losses.compute_sdf(ys_true_ds.cpu().numpy(), ys_pred_ds.detach().to('cpu').numpy().shape)
                    gdl_loss = self.losses['segm']['dice_loss'](ys_pred_softmax_ds, ys_true_ds.to(maybe_gpu))
                    bl_loss = self.losses['segm']['boundary_loss'](ys_pred_softmax_ds, torch.from_numpy(gt_sdf).to(maybe_gpu))
                    loss_segm = gdl_loss + alpha * bl_loss

                    metrics_acc['batchw']['loss'].append(loss_segm.item())

                    loss_segm.backward()
                    self.optimizers['segm'].step()

                    prog_bar.update(1)
        else:
            # ----------------------- Validation regime -----------------------
            loader_ds = loaders[name_ds]['val']

            steps_ds = len(loader_ds)
            prog_bar_params.update({'total': steps_ds,
                                    'desc': f'Validate, epoch {epoch_idx}'})

            loader_ds_iter = iter(loader_ds)

            alpha = 0.01
            with torch.no_grad(), tqdm(**prog_bar_params) as prog_bar:
                for step_idx in range(steps_ds):
                    data_batch_ds = next(loader_ds_iter)

                    xs_ds, ys_true_ds = data_batch_ds['xs'], data_batch_ds['ys']
                    fnames_acc['oai'].extend(data_batch_ds['path_image'])

                    ys_true_arg_ds = torch.argmax(ys_true_ds.long(), dim=1)
                    xs_ds = xs_ds.to(maybe_gpu)
                    ys_true_arg_ds = ys_true_arg_ds.to(maybe_gpu)

                    ys_pred_ds = self.models['segm'](xs_ds)

                    # loss_segm = self.losses['segm'](input=ys_pred_ds,
                    #                                 target=ys_true_arg_ds)

                    ys_pred_softmax_ds = nn.Softmax(dim=1)(ys_pred_ds)
                    gt_sdf = losses.compute_sdf(ys_true_ds.cpu().numpy(), ys_pred_ds.detach().to('cpu').numpy().shape)
                    gdl_loss = self.losses['segm']['dice_loss'](ys_pred_softmax_ds, ys_true_ds.to(maybe_gpu))
                    bl_loss = self.losses['segm']['boundary_loss'](ys_pred_softmax_ds, torch.from_numpy(gt_sdf).to(maybe_gpu))
                    loss_segm = gdl_loss + alpha * bl_loss

                    metrics_acc['batchw']['loss'].append(loss_segm.item())

                    # Calculate metrics
                    ys_pred_softmax_ds = nn.Softmax(dim=1)(ys_pred_ds)
                    ys_pred_softmax_np_ds = ys_pred_softmax_ds.to('cpu').numpy()

                    ys_pred_arg_np_ds = ys_pred_softmax_np_ds.argmax(axis=1)
                    ys_true_arg_np_ds = ys_true_arg_ds.to('cpu').numpy()

                    metrics_acc['datasetw'][f'{name_ds}__cm'] += metrics.confusion_matrix(
                        ys_pred_arg_np_ds, ys_true_arg_np_ds, 3)

                    prog_bar.update(1)

        for k, v in metrics_acc['samplew'].items():
            metrics_acc['samplew'][k] = np.asarray(v)
        metrics_acc['datasetw'][f'{name_ds}__dice_score'] = np.asarray(
            metrics.dice_score_from_cm(metrics_acc['datasetw'][f'{name_ds}__cm']))

        return metrics_acc, fnames_acc

    def fit(self, loaders):
        epoch_idx_best = -1
        loss_best = float('inf')
        metrics_train_best = dict()
        fnames_train_best = []
        metrics_val_best = dict()
        fnames_val_best = []

        for epoch_idx in range(self.num_epoch):
            self.models = {n: m.train() for n, m in self.models.items()}
            metrics_train, fnames_train = self.run_one_epoch(epoch_idx, loaders)

            # Process the accumulated metrics
            for k, v in metrics_train['batchw'].items():
                if k.startswith('loss'):
                    metrics_train['datasetw'][k] = np.mean(np.asarray(v))
                else:
                    logger.warning(f'Non-processed batch-wise entry: {k}')

            self.models = {n: m.eval() for n, m in self.models.items()}
            metrics_val, fnames_val = \
                self.run_one_epoch(epoch_idx, loaders)

            # Process the accumulated metrics
            for k, v in metrics_val['batchw'].items():
                if k.startswith('loss'):
                    metrics_val['datasetw'][k] = np.mean(np.asarray(v))
                else:
                    logger.warning(f'Non-processed batch-wise entry: {k}')

            # Learning rate update
            for s, m in self.lr_update_rule.items():
                if epoch_idx == s:
                    for name, optim in self.optimizers.items():
                        # for param_group in optim.param_groups:
                        #     param_group['lr'] *= m

                        # Add console logging
                        logger.info(f'Epoch: {epoch_idx}')
                        for subset, metrics in (('train', metrics_train),
                                                ('val', metrics_val)):
                            logger.info(f'{subset} metrics:')
                            for k, v in metrics['datasetw'].items():
                                logger.info(f'{k}: \n{v}')

            # Add TensorBoard logging
            for subset, metrics in (('train', metrics_train),
                                    ('val', metrics_val)):
                # Log only dataset-reduced metrics
                for k, v in metrics['datasetw'].items():
                    if isinstance(v, np.ndarray):
                        self.tensorboard.add_scalars(
                            f'fold_{self.fold_idx}/{k}_{subset}',
                            {f'class{i}': e for i, e in enumerate(v.ravel().tolist())},
                            global_step=epoch_idx)
                    elif isinstance(v, (str, int, float)):
                        self.tensorboard.add_scalar(
                            f'fold_{self.fold_idx}/{k}_{subset}',
                            float(v),
                            global_step=epoch_idx)
                    else:
                        logger.warning(f'{k} is of unsupported dtype {v}')
            for name, optim in self.optimizers.items():
                for param_group in optim.param_groups:
                    self.tensorboard.add_scalar(
                        f'fold_{self.fold_idx}/learning_rate/{name}',
                        param_group['lr'],
                        global_step=epoch_idx)

            # Save the model
            loss_curr = metrics_val['datasetw']['loss']
            print('Loss=', loss_curr)

            if 0 < loss_curr < loss_best:
                loss_best = loss_curr
                epoch_idx_best = epoch_idx
                metrics_train_best = metrics_train
                metrics_val_best = metrics_val
                fnames_train_best = fnames_train
                fnames_val_best = fnames_val

                self.handlers_ckpt['segm'].save_new_ckpt(
                    model=self.models['segm'],
                    model_name='unet',
                    fold_idx=self.fold_idx,
                    epoch_idx=epoch_idx)

        msg = (f'Finished fold {self.fold_idx} '
               f'with the best loss {loss_best:.5f} '
               f'on epoch {epoch_idx_best}, '
               f'weights: ({self.paths_weights_fold})')

        logger.info(msg)
        return (metrics_train_best, fnames_train_best,
                metrics_val_best, fnames_val_best)



