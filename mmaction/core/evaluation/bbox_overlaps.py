import numpy as np


def overlap2d(bboxes1, bboxes2):
    """Calculate the overlap between each bbox of bboxes1 and bboxes2.

    这个函数中，k 要么为 1，要么==n

    Args:
        bboxes1 (np.ndarray): shape (n, 4).
        bboxes2 (np.ndarray): shape (k, 4).

    Returns:
        np.ndarray: Overlap between the boxes pairs.
    """
    x_min = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
    y_min = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
    x_max = np.minimum(bboxes1[:, 2], bboxes2[:, 2])
    y_max = np.minimum(bboxes1[:, 3], bboxes2[:, 3])

    width = np.maximum(0, x_max - x_min)
    height = np.maximum(0, y_max - y_min)

    return width * height


def area2d(box):
    """Calculate bounding boxes area.

    Args:
        box (np.ndarray): Bounding boxes are in shape (n, 5) and
            [x1, y1, x2, y2] format.

    Returns:
        np.ndarray: Area for bounding boxes in shape (n,)
    """
    width = box[:, 2] - box[:, 0]
    height = box[:, 3] - box[:, 1]

    return width * height


# TODO: Maybe we can use BboxOverlaps2D in MMDet to implement iou2d & iou3d?
# However, BboxOverlaps2D use pytorch instead of numpy. Better double check
# this after related models are ready.
def iou2d(bboxes1, bboxes2):
    """Calculate the IoUs between each bbox of bboxes1 and bboxes2.

    注意，本函数中 bboxes2 只能有一个 bbox，即 k == 1

    Args:
        bboxes1 (np.ndarray): shape (n, 4).
        bboxes2 (np.ndarray): shape (k, 4).

    Returns:
        np.ndarray: IoU between the boxes pairs.
    """
    if bboxes1.ndim == 1:
        bboxes1 = bboxes1[None, :]
    if bboxes2.ndim == 1:
        bboxes2 = bboxes2[None, :]

    assert len(bboxes2) == 1

    # (n, )
    overlap = overlap2d(bboxes1, bboxes2)

    # (n, ) / ((n,) + scalar - (n,)) = (n, )
    return overlap / (area2d(bboxes1) + area2d(bboxes2) - overlap)


def iou3d(bboxes1, bboxes2):
    """Calculate the IoU3d regardless of temporal overlap between two pairs of
    bboxes.

    bboxes1 and bboxes2 share the same shape, (n, k), where k >= 5. The format
    is (frame_id, x1, y1, x2, y2, ...)

    Args:
        bboxes1 (np.ndarray): shape (n, k).
        bboxes2 (np.ndarray): shape (n, k).

    Returns:
        np.ndarray: IoU3d regardless of temporal overlap.
    """

    assert bboxes1.shape[0] == bboxes2.shape[0]
    assert np.all(bboxes1[:, 0] == bboxes2[:, 0])

    # cal average iou for all frames
    overlap = overlap2d(bboxes1[:, 1:5], bboxes2[:, 1:5])
    return np.mean(
        overlap /
        (area2d(bboxes1[:, 1:5]) + area2d(bboxes2[:, 1:5]) - overlap))


def spatio_temporal_iou3d(bboxes1, bboxes2, spatial_only=False):
    """Calculate the IoU3d between two pairs of bboxes.

    (frame_id, x1, y1, x2, y2, ...), k >= 5

    Args:
        bboxes1 (np.ndarray): shape (n, k).
        bboxes2 (np.ndarray): shape (m, k).
        spatial_only (bool): Whether to consider the temporal overlap.
            Default: False.

    Returns:
        np.ndarray: IoU3d for bboxes between two tubes.
    """
    # min/max frame_id
    tmin = max(bboxes1[0, 0], bboxes2[0, 0])
    tmax = min(bboxes1[-1, 0], bboxes2[-1, 0])

    # no temporal overlap
    if tmax < tmin:
        return 0.0

    temporal_inter = tmax - tmin + 1
    temporal_union = (
        max(bboxes1[-1, 0], bboxes2[-1, 0]) -
        min(bboxes1[0, 0], bboxes2[0, 0]) + 1)

    # chooose [tmin, tmax] from bboxes1 & bboxes2
    # tube1 & tube2 are in shape [tmax - tmin + 1, 5]
    tube1 = bboxes1[int(np.where(
        bboxes1[:,
                0] == tmin)[0]):int(np.where(bboxes1[:, 0] == tmax)[0]) + 1, :]
    tube2 = bboxes2[int(np.where(
        bboxes2[:,
                0] == tmin)[0]):int(np.where(bboxes2[:, 0] == tmax)[0]) + 1, :]

    return iou3d(tube1, tube2) * (1. if spatial_only else temporal_inter /
                                  temporal_union)


def spatio_temporal_nms3d(tubes, overlap=0.5):
    """NMS processing for tubes in spatio and temporal dimension.

    tubes is a list.
    each tube is a list of tuple with 2 elements (tube_info, tube_score).
    tube_score is a float and tube_info is (n, 6) ndarray

    Args:
        tubes (np.ndarray): Bounding boxes in tubes.
        overlap (float): Threshold of overlap for nms

    Returns:
        np.ndarray[int]: Index for Selected bboxes.
    """
    if not tubes:
        return np.array([], dtype=np.int32)

    indexes = np.argsort([tube[1] for tube in tubes])
    indices = np.zeros(indexes.size, dtype=np.int32)
    counter = 0

    while indexes.size > 0:
        i = indexes[-1]
        indices[counter] = i
        counter += 1
        ious = np.array([
            spatio_temporal_iou3d(tubes[index_list][0], tubes[i][0])
            for index_list in indexes[:-1]
        ])
        indexes = indexes[np.where(ious <= overlap)[0]]

    return indices[:counter]
