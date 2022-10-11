import cv2
import numpy as np
from torch.utils.data.dataset import Dataset


H = 385
W = 245


def read_image(path_file):
    image = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (W, H))
    image = image.reshape((1, *image.shape))
    return image


def read_mask(path_file):
    mask = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)

    if mask is not None:
        mask = cv2.resize(mask, (W, H))
        mask[(mask != 255) & (mask != 233)] = 0
        mask[mask == 255] = 1
        mask[mask == 233] = 2
        ret = np.empty((3, *mask.shape), dtype=mask.dtype)
        ret[0, :, :] = 0
        ret[1, :, :] = np.isin(mask, 1).astype(np.uint8)
        ret[2, :, :] = np.isin(mask, 2).astype(np.uint8)

        return ret
    else:
        return mask


class DatasetOAIfor(Dataset):
    def __init__(self, df_meta, name=None, transforms=None):

        self.df_meta = df_meta
        self.name = name
        self.transforms = transforms

    def __len__(self):
        return len(self.df_meta)

    def __getitem__(self, idx):
        image = read_image(self.df_meta['path_image'].iloc[idx])
        mask = read_mask(self.df_meta['path_mask'].iloc[idx])

        # Apply transformations
        if self.transforms is not None:
            for t in self.transforms:
                if hasattr(t, 'randomize'):
                    t.randomize()
                image, mask = t(image, mask)

        tmp = dict(self.df_meta.iloc[idx])
        tmp['image'] = image
        tmp['mask'] = mask

        if mask is None:
            tmp['ys'] = tmp['image']
        else:
            tmp['ys'] = tmp['mask']

        tmp['xs'] = tmp['image']

        return tmp

    def read_image(self, path_file):
        return read_image(path_file=path_file)

    def read_mask(self, path_file):
        return read_mask(path_file=path_file)
