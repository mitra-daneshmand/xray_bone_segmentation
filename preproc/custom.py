import logging
import numpy as np


logging.basicConfig()
logger = logging.getLogger('preprocessing_custom')
logger.setLevel(logging.DEBUG)


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask=None):
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std

        if mask is not None:
            mask = mask.astype(np.float32)
        return img, mask

