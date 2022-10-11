import logging
from typing import List

import torch
from torch import nn, Tensor, einsum

logging.basicConfig()
logger = logging.getLogger('losses')
logger.setLevel(logging.DEBUG)


if torch.cuda.is_available():
    maybe_gpu = 'cuda'
else:
    maybe_gpu = 'cpu'


class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, batch_avg=True, batch_weight=None,
                 class_avg=True, class_weight=None, **kwargs):
        """
        Parameters
        ----------
        batch_avg:
            Whether to average over the batch dimension.
        batch_weight:
            Batch samples importance coefficients.
        class_avg:
            Whether to average over the class dimension.
        class_weight:
            Classes importance coefficients.
        """
        super().__init__()
        self.num_classes = num_classes
        self.batch_avg = batch_avg
        self.class_avg = class_avg
        self.batch_weight = batch_weight
        self.class_weight = class_weight
        logger.warning('Redundant loss function arguments:\n{}'
                       .format(repr(kwargs)))
        self.ce = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, input, target):
        """
        Parameters
        ----------
        input_: (b, ch, d0, d1) tensor
        target: (b, d0, d1) tensor
        Returns
        -------
        out: float tensor
        """
        return self.ce(input, target)


class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bkwh,bkwh->bk", pc, tc)
        union: Tensor = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

        divided: Tensor = 1 - 2 * (einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)

        loss = divided.mean()

        return loss.to(maybe_gpu)


class BoundaryLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss.to(maybe_gpu)

