import logging

import torch
import numpy as np
from scipy import ndimage
from ._surface_lut import neighbour_code_to_normals

logging.basicConfig()
logger = logging.getLogger('metrics')
logger.setLevel(logging.DEBUG)


def confusion_matrix(input_, target, num_classes):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
    Args:
        input_: (d0, ..., dn) ndarray or tensor
        target: (d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
    Returns:
        out: (num_classes, num_classes) ndarray
            Confusion matrix.
    """
    if torch.is_tensor(input_):
        input_ = input_.detach().to('cpu').numpy()
    if torch.is_tensor(target):
        target = target.detach().to('cpu').numpy()

    replace_indices = np.vstack((
        target.flatten(),
        input_.flatten())
    ).T
    cm, _ = np.histogramdd(
        replace_indices,
        bins=(num_classes, num_classes),
        range=[(0, num_classes - 1), (0, num_classes - 1)]
    )
    return cm.astype(np.uint32)


def dice_score_from_cm(cm):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
    Args:
        cm: (d, d) ndarray
            Confusion matrix.

    Returns:
        out: (d, ) list
            List of class Dice scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        false_positives = cm[:, index].sum() - true_positives
        false_negatives = cm[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            score = 0
        else:
            score = 2 * float(true_positives) / denom
        scores.append(score)
    return scores


# ----------------------------------------------------------------------------


def _template_score(func_score_from_cm, input_, target, num_classes,
                    batch_avg, batch_weight, class_avg, class_weight):
    """
    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        batch_avg: bool
            Whether to average over the batch dimension.
        batch_weight: (b,) iterable
            Batch samples importance coefficients.
        class_avg: bool
            Whether to average over the class dimension.
        class_weight: (c,) iterable
            Classes importance coefficients. Ignored when `class_avg` is False.
    Returns:
        out: scalar if `class_avg` is True, (num_classes,) list otherwise
    """
    if torch.is_tensor(input_):
        num_samples = tuple(input_.size())[0]
    else:
        num_samples = input_.shape[0]

    scores = np.zeros((num_samples, num_classes))
    for sample_idx in range(num_samples):
        cm = confusion_matrix(input_=input_[sample_idx],
                              target=target[sample_idx],
                              num_classes=num_classes)
        scores[sample_idx, :] = func_score_from_cm(cm)

    if batch_avg:
        scores = np.mean(scores, axis=0, keepdims=True)
    if class_avg:
        if class_weight is not None:
            scores = scores * np.reshape(class_weight, (1, -1))
        scores = np.mean(scores, axis=1, keepdims=True)
    return np.squeeze(scores)


def dice_score(input_, target, num_classes,
               batch_avg=True, batch_weight=None,
               class_avg=False, class_weight=None):
    """
    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        batch_avg: bool
            Whether to average over the batch dimension.
        batch_weight: (b,) iterable
            Batch samples importance coefficients.
        class_avg: bool
            Whether to average over the class dimension.
        class_weight: (c,) iterable
            Classes importance coefficients. Ignored when `class_avg` is False.
    Returns:
        out: scalar if `class_avg` is True, (num_classes,) list otherwise
    """
    return _template_score(
        dice_score_from_cm, input_, target, num_classes,
        batch_avg, batch_weight, class_avg, class_weight)


def _surf_dists(mask_gt, mask_pred, spacing_mm):
    """Compute closest distances from all surface points to the other surface.
    Finds all surface elements "surfels" in the ground truth mask `mask_gt` and
    the predicted mask `mask_pred`, computes their area in mm^2 and the distance
    to the closest point on the other surface. It returns two sorted lists of
    distances together with the corresponding surfel areas. If one of the masks
    is empty, the corresponding lists are empty and all distances in the other
    list are `inf`.
    Args:
        mask_gt: 3-dim Numpy array of type bool
            The ground truth mask.
        mask_pred: 3-dim Numpy array of type bool
            The predicted mask.
        spacing_mm: 3-element list-like structure
            Voxel spacing in x0, x1 and x2 direction.
    Returns:
        A dict with:
        "distances_gt_to_pred": 1-dim numpy array of type float
            The distances in mm from all ground truth surface elements to the
            predicted surface, sorted from smallest to largest.
        "distances_pred_to_gt": 1-dim numpy array of type float
            The distances in mm from all predicted surface elements to the
            ground truth surface, sorted from smallest to largest.
        "surfel_areas_gt": 1-dim numpy array of type float
            The area in mm^2 of the ground truth surface elements in the same order as
            distances_gt_to_pred.
        "surfel_areas_pred": 1-dim numpy array of type float
            The area in mm^2 of the predicted surface elements in the same order as
            distances_pred_to_gt.
    """

    # compute the area for all 256 possible surface elements
    # (given a 2x2x2 neighbourhood) according to the spacing_mm
    neighbour_code_to_surface_area = np.zeros([256])
    for code in range(256):
        normals = np.array(neighbour_code_to_normals[code])
        sum_area = 0
        for normal_idx in range(normals.shape[0]):
            # normal vector
            n = np.zeros([3])
            n[0] = normals[normal_idx, 0] * spacing_mm[1] * spacing_mm[2]
            n[1] = normals[normal_idx, 1] * spacing_mm[0] * spacing_mm[2]
            n[2] = normals[normal_idx, 2] * spacing_mm[0] * spacing_mm[1]
            area = np.linalg.norm(n)
            sum_area += area
        neighbour_code_to_surface_area[code] = sum_area

    # compute the bounding box of the masks to trim
    # the volume to the smallest possible processing subvolume
    mask_all = mask_gt | mask_pred
    bbox_min = np.zeros(3, np.int64)
    bbox_max = np.zeros(3, np.int64)

    # max projection to the x0-axis
    proj_0 = np.max(np.max(mask_all, axis=2), axis=1)
    idx_nonzero_0 = np.nonzero(proj_0)[0]
    if len(idx_nonzero_0) == 0:  # pylint: disable=g-explicit-length-test
        return {"distances_gt_to_pred": np.array([]),
                "distances_pred_to_gt": np.array([]),
                "surfel_areas_gt": np.array([]),
                "surfel_areas_pred": np.array([])}

    bbox_min[0] = np.min(idx_nonzero_0)
    bbox_max[0] = np.max(idx_nonzero_0)

    # max projection to the x1-axis
    proj_1 = np.max(np.max(mask_all, axis=2), axis=0)
    idx_nonzero_1 = np.nonzero(proj_1)[0]
    bbox_min[1] = np.min(idx_nonzero_1)
    bbox_max[1] = np.max(idx_nonzero_1)

    # max projection to the x2-axis
    proj_2 = np.max(np.max(mask_all, axis=1), axis=0)
    idx_nonzero_2 = np.nonzero(proj_2)[0]
    bbox_min[2] = np.min(idx_nonzero_2)
    bbox_max[2] = np.max(idx_nonzero_2)

    # crop the processing subvolume.
    # we need to zeropad the cropped region with 1 voxel at the lower,
    # the right and the back side. This is required to obtain the "full"
    # convolution result with the 2x2x2 kernel
    cropmask_gt = np.zeros((bbox_max - bbox_min) + 2, np.uint8)
    cropmask_pred = np.zeros((bbox_max - bbox_min) + 2, np.uint8)

    cropmask_gt[0:-1, 0:-1, 0:-1] = mask_gt[bbox_min[0]:bbox_max[0] + 1,
                                            bbox_min[1]:bbox_max[1] + 1,
                                            bbox_min[2]:bbox_max[2] + 1]

    cropmask_pred[0:-1, 0:-1, 0:-1] = mask_pred[bbox_min[0]:bbox_max[0] + 1,
                                                bbox_min[1]:bbox_max[1] + 1,
                                                bbox_min[2]:bbox_max[2] + 1]

    # compute the neighbour code for each voxel
    # the resulting arrays are spatially shifted by minus half a voxel in each axis.
    # i.e. the points are located at the corners of the original voxels
    kernel = np.array([[[128, 64],
                        [32, 16]],
                       [[8, 4],
                        [2, 1]]])
    neighbour_code_map_gt = ndimage.filters.correlate(
        cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbour_code_map_pred = ndimage.filters.correlate(
        cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0)

    # create masks with the surface voxels
    borders_gt = ((neighbour_code_map_gt != 0) & (neighbour_code_map_gt != 255))
    borders_pred = ((neighbour_code_map_pred != 0) &
                    (neighbour_code_map_pred != 255))

    # compute the distance transform
    # (closest distance of each voxel to the surface voxels)
    if borders_gt.any():
        distmap_gt = ndimage.morphology.distance_transform_edt(
            ~borders_gt, sampling=spacing_mm)
    else:
        distmap_gt = np.Inf * np.ones(borders_gt.shape)

    if borders_pred.any():
        distmap_pred = ndimage.morphology.distance_transform_edt(
            ~borders_pred, sampling=spacing_mm)
    else:
        distmap_pred = np.Inf * np.ones(borders_pred.shape)

    # compute the area of each surface element
    surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
    surface_area_map_pred = neighbour_code_to_surface_area[
        neighbour_code_map_pred]

    # create a list of all surface elements with distance and area
    distances_gt_to_pred = distmap_pred[borders_gt]
    distances_pred_to_gt = distmap_gt[borders_pred]
    surfel_areas_gt = surface_area_map_gt[borders_gt]
    surfel_areas_pred = surface_area_map_pred[borders_pred]

    # sort them by distance
    if distances_gt_to_pred.shape != (0,):
        sorted_surfels_gt = np.array(
            sorted(zip(distances_gt_to_pred, surfel_areas_gt)))
        distances_gt_to_pred = sorted_surfels_gt[:, 0]
        surfel_areas_gt = sorted_surfels_gt[:, 1]

    if distances_pred_to_gt.shape != (0,):
        sorted_surfels_pred = np.array(
            sorted(zip(distances_pred_to_gt, surfel_areas_pred)))
        distances_pred_to_gt = sorted_surfels_pred[:, 0]
        surfel_areas_pred = sorted_surfels_pred[:, 1]

    return {"distances_gt_to_pred": distances_gt_to_pred,
            "distances_pred_to_gt": distances_pred_to_gt,
            "surfel_areas_gt": surfel_areas_gt,
            "surfel_areas_pred": surfel_areas_pred}


def avg_surf_dist(input_, target, num_classes, spacing_mm,
                  batch_avg=True, batch_weight=None,
                  class_avg=False, class_weight=None):
    """Computes the average surface distance.
    Computes the average surface distances by correctly taking the area of each
    surface element into account.
    Args:
        input_: (b, d0, d1, d2) ndarray or tensor
        target: (b, d0, d1, d2) ndarray or tensor
        num_classes: int
            Total number of classes.
        spacing_mm: 3-tuple
            Pixel spacing in mm, one per each spatial dimension.
        batch_avg: bool
            Whether to average over the batch dimension.
        batch_weight: (b,) iterable
            Importance coefficients for batch samples.
        class_avg: bool
            Whether to average over the class dimension.
        class_weight:
            Importance coefficients for classes. Ignored when `class_avg` is False.
    Returns:
        out: ([b,] [c,] 2) ndarray
            The average distance (in mm) from the ground truth surface to the
            predicted surface [..., 0] and vice versa [..., 1].
    """

    def _asd(surface_distances):
        distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
        distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
        surfel_areas_gt = surface_distances["surfel_areas_gt"]
        surfel_areas_pred = surface_distances["surfel_areas_pred"]
        average_distance_gt_to_pred = (
                np.sum(distances_gt_to_pred * surfel_areas_gt) /
                np.sum(surfel_areas_gt))
        average_distance_pred_to_gt = (
                np.sum(distances_pred_to_gt * surfel_areas_pred) /
                np.sum(surfel_areas_pred))
        return average_distance_pred_to_gt, average_distance_gt_to_pred

    if torch.is_tensor(input_):
        input_ = input_.detach().to('cpu').numpy()
    if torch.is_tensor(target):
        target = target.detach().to('cpu').numpy()

    if input_.ndim != 4:
        raise ValueError(f"`input_` is expected to have 4 dims, got {input_.ndim}")
    if target.ndim != 4:
        raise ValueError(f"`target` is expected to have 4 dims, got {target.ndim}")
    if batch_weight is not None:
        raise NotImplementedError(f"Custom `batch_weight` is not supported")

    num_samples = input_.shape[0]

    scores = np.zeros((num_samples, num_classes, 2))
    for sample_idx in range(num_samples):
        for class_idx in range(num_classes):
            sel_input_ = input_[sample_idx] == class_idx
            sel_target = target[sample_idx] == class_idx

            if not np.any(sel_input_) or not np.any(sel_target):
                scores[sample_idx, class_idx, :] = np.nan
            else:
                tmp = _surf_dists(mask_gt=sel_input_,
                                  mask_pred=sel_target,
                                  spacing_mm=spacing_mm)
                scores[sample_idx, class_idx, :] = _asd(tmp)

    if batch_avg:
        scores = np.mean(scores, axis=0, keepdims=True)
    if class_avg:
        if class_weight is not None:
            scores = scores * np.reshape(class_weight, (1, -1))
        scores = np.mean(scores, axis=1, keepdims=True)
    return np.squeeze(scores)

