import os.path as osp
import pickle
from collections import defaultdict

import numpy as np

from .bbox_overlaps import iou2d, spatio_temporal_iou3d, spatio_temporal_nms3d

# TODO: try to reduce code duplication
#   Since all of frame_mean_ap, frame_mean_ap_error and video_mean_ap calculate
#   PR curve and ap, the structure of these functions are almost the same


def pr_to_ap(precision_recall):
    """Compute AP given precision-recall.

    Args:
        precision_recall (np.ndarray): precision_recall is an Nx2 array with
            first column being precision and second column being recall.

    Returns:
        np.ndarray: The result of average precision.
    """
    recall_diff = np.diff(precision_recall[:, 1])
    precision_sum = precision_recall[1:, 0] + precision_recall[:-1, 0]

    # Calculate the area of the polygon under PR curve, aka ap.
    # The polygon is composed of a set of right-angled trapezoids.
    # The height of each trapezoid is recall_diff[i] and the sum of bottom and
    # top is precision_sum[i]. And the area of each right-angled trapezoids
    # is `(top + bottom) * height / 2`
    return np.sum(recall_diff * precision_sum * 0.5)


def frame_mean_ap(det_results,
                  labels,
                  videos,
                  gt_tubes,
                  threshold=0.5,
                  return_error=False):
    """Calculate error information for frame mAP in tubes.

    Args:
        det_results (np.ndarray): Detection results for each frame. The result
            for each frame is a ndarray object with (N, 8) shape, where N is
            the number of predictions. And the format of each (8,) vector is
            `(video_id, frame_id, label_id, socre, x1, y1, x2, y2)`.
        labels (list): List of action labels.
        videos (list): List of video names.
        gt_tubes (dict): Ground truth tubes for each video. The format of
            ``gt_tubes`` is {video_name: {label: list[tube]}}, where tube is a
            np.ndarray with (N, 5) shape and each row contains frame index and
            a bounding box.
        threshold (float): Threshold for IoU. Default: 0.5.
        return_error (bool): Whether to calculate error information for frame
            mAP in tubes. The error information will contain ap_results,
            localization_error, classification_error, time_error, other_error,
            missing_detections.

    Returns:
        dict: Result dict containing frame mAP, localization_error,
            classification_error, time_error, other_error and
            missing_detections. If ``return_error`` is false, only frame mAP
            is included.
    """
    ap_results = []
    other_ap_results = [[], [], [], []]
    missing_detections = []

    for label_index, label in enumerate(labels):

        # filter results by label_index
        # TODO: better to first divide det_results by label, or you need to
        # enumerate det_results by n times instead of 1 time. Better to double
        # check this after tube related models and pipelines are ready.
        det_result = det_results[det_results[:, 2] == label_index, :]

        # save all kinds of ground truth
        # gt saves tubes with the current label_index
        # key -> (video_id, frame_id)
        # value -> (n, 4) ndarray
        gt = defaultdict(list)

        # other_gt saves tubes left
        # key -> (video_id, frame_id)
        # value -> (n, 4) ndarray
        other_gt = defaultdict(list)

        # saves all label_ids of current current video
        # key -> video_id
        # value -> list[int]
        label_dict = defaultdict(list)

        for video_id, video in enumerate(videos):
            # tubes is a dict with format `{label: list[tube]}`. where each
            # tube is a np.ndarray with (N, 5) shape. The format of (5,)
            # vector is (framd_ind, x1, y1, x2, y2)
            tubes = gt_tubes[video]
            label_dict[video_id] = list(tubes)

            for tube_label_index in tubes:
                for tube in tubes[tube_label_index]:
                    for t in tube:
                        key = (video_id, int(t[0]))
                        if tube_label_index == label_index:
                            gt[key].append(t[1:5])
                        else:
                            other_gt[key].append(t[1:5])
        for key in gt:
            gt[key] = np.array(gt[key].copy())
        for key in other_gt:
            other_gt[key] = np.array(other_gt[key].copy())

        original_key = list(gt)
        precision_recall = np.empty((det_result.shape[0] + 1, 6),
                                    dtype=np.float32)
        precision_recall[0, 0] = 1.0
        precision_recall[0, 1:] = 0.0
        fn = sum([item.shape[0] for item in gt.values()])
        (fp, tp, localization_error, classification_error, other_error,
         time_error) = (0, 0, 0, 0, 0, 0)

        # sort predictions with score in descending order
        for i, j in enumerate(np.argsort(-det_result[:, 3])):
            # key: (video_id, frame_id)
            key = (int(det_result[j, 0]), int(det_result[j, 1]))
            box = det_result[j, 4:8]
            is_positive = False

            if key in original_key:
                if key in gt:
                    ious = iou2d(gt[key], box)
                    max_idx = np.argmax(ious)

                    if ious[max_idx] >= threshold:
                        # match
                        is_positive = True
                        gt[key] = np.delete(gt[key], max_idx, 0)
                        if gt[key].size == 0:
                            del gt[key]
                    else:
                        # iou between current precition and remaining ground
                        # truth is not good enough
                        localization_error += 1
                else:
                    # no ground truth with the same class_id in the target
                    # frame/video
                    localization_error += 1

            elif key in other_gt:
                ious = iou2d(other_gt[key], box)
                if np.max(ious) >= threshold:
                    # tube spatio-temporal match, but class mismatch
                    classification_error += 1
                else:
                    other_error += 1

            elif label_index in label_dict[key[0]]:
                # current precition has no match gt in the current frame,
                # and there is gt with the same class_id in other frames from
                # the same video
                time_error += 1
            else:
                other_error += 1

            if is_positive:
                tp += 1
                fn -= 1
            else:
                fp += 1

            precision_recall[i + 1, 0] = tp / max(1, (tp + fp))  # precision
            precision_recall[i + 1, 1] = tp / max(1, (tp + fn))  # recall
            if return_error:
                precision_recall[i + 1,
                                 2] = localization_error / max(1, (tp + fp))
                precision_recall[i + 1,
                                 3] = classification_error / max(1, (tp + fp))
                precision_recall[i + 1, 4] = time_error / max(1, (tp + fp))
                precision_recall[i + 1, 5] = other_error / max(1, (tp + fp))

        ap_results.append(pr_to_ap(precision_recall[..., :2]))

        if return_error:
            for j in range(2, 6):
                other_ap_results[j - 2].append(
                    pr_to_ap(precision_recall[..., [j, 1]]))
            missing_detections.append(precision_recall[-1, 1])

    ap_results = np.array(ap_results) * 100

    if not return_error:
        return dict(ap_results=ap_results)

    other_ap_results = np.array(other_ap_results) * 100
    (localization_error, classification_error, time_error,
     other_error) = other_ap_results[:4]
    missing_detections = 100 - 100 * np.array(missing_detections)

    result = dict(
        ap_results=ap_results,
        localization_error=localization_error,
        classification_error=classification_error,
        time_error=time_error,
        other_error=other_error,
        missing_detections=missing_detections)

    result_str = ''
    for i, label in enumerate(labels):
        result_str += f'{label:20s}' + ' '.join(
            [f'{v[i]:8.2f}' for v in result.values()]) + '\n'
    result_str += '\n' + f"{'mean':20s}" + ' '.join(
        [f'{np.mean(v):8.2f}' for v in result.values()]) + '\n'

    msg = 'Error Analysis\n'
    msg += f"\n{'label':20s} {'   AP   ':8s} {'  Loc.  ':8s} {'  Cls.  ':8s} "
    msg += f"{'  Time  ':8s} {' Other ':8s} {' missed ':8s}\n"
    msg += f'\n{result_str}'
    result['msg'] = msg

    return result


def video_mean_ap(labels,
                  videos,
                  gt_tubes,
                  tube_dir,
                  threshold=0.5,
                  overlap=0.3):
    """Calculate video mAP for tubes.

    The format of each predicted tube pickle is `{label_id: [tube_info]}`.
    `tube_info` is a `tuple(tube_bboxes_info, tube_score)` where `tube_score`
    is a float and `tube_bboxes_info` is a ndarray in shape `(num_bboxes, 6)`.
    And the format of each bbox is `(frame_id, x1, y1, x2, y2, unknown)`.

    Args:
        labels (list[str]): List of action labels.
        videos (list[str]): List of video names.
        gt_tubes (dict): Ground truth tubes for each video. The format of
            ``gt_tubes`` is {video_name: {label: list[tube]}}, where tube is a
            np.ndarray with (N, 5) shape, each row contains frame index and a
            bbounding box.
        tube_dir (str): Directory of predicted tube pickle files.
        threshold (float): Threshold for IoU. Default: 0.5.
        overlap (float): Threshold of overlap for nms. Default: 0.3.

    Returns:
        float: The calculated video mAP.
    """

    num_labels = len(labels)

    # load pickle predictions, and det_results is a dict with format
    # `{label_id: [tuple(video, socre, tube_bboxes_info)]}'.
    # `tube_bboxes_info` is a ndarray in shape `(num_bboxes, 6)`
    det_results = defaultdict(list)
    for video in videos:
        tube_filepath = osp.join(tube_dir, video + '_tubes.pkl')
        if not osp.isfile(tube_filepath):
            raise FileNotFoundError(
                f'Extracted tubes {tube_filepath} is missing')
        with open(tube_filepath, 'rb') as f:
            # The format of tubes is {label: list[tube]}
            tubes = pickle.load(f)

        # Convert the format of predictions
        for label_index in range(num_labels):
            tube_list = tubes[label_index]
            index = spatio_temporal_nms3d(tube_list, overlap)
            det_results[label_index].extend([
                (video, tube_list[i][1], tube_list[i][0]) for i in index
            ])

    results = []
    for label_index in range(num_labels):
        # cal ap for each label

        # get predictions of label_index and the format is
        # `[tuple(video, socre, tube_bboxes_info)]` and tube_bboxes_info is
        # ndarray in shape (n, 6) and the format of each bbox is
        # `(frame_id, x1, y1, x2, y2, unknown)`
        det_result = np.array(det_results[label_index])

        # get gt for label_index in format `{video: [gt_tube]}`.
        # gt_tube is (n, 5) ndarray, where each bbox's format is
        # (frame_id, x1, y1, x2, y2)
        gt = defaultdict(list)
        for video in videos:
            tubes = gt_tubes[video]
            if label_index not in tubes:
                continue
            gt[video] = tubes[label_index].copy()
            if len(gt[video]) == 0:
                del gt[video]

        # prepare for calculate pr curve
        precision_recall = np.empty((len(det_result) + 1, 2), dtype=np.float32)
        precision_recall[0, 0] = 1.0
        precision_recall[0, 1] = 0.0
        fn, fp, tp = sum([len(item) for item in gt.values()]), 0, 0

        # sort by prediction score in descending order
        dets = -np.array(det_result[:, 1])
        for i, j in enumerate(np.argsort(dets)):
            key, _, tube = det_result[j]
            is_positive = False

            if key in gt:
                # calculate iou between current precition and the remaining gt
                # in the same video
                ious = [spatio_temporal_iou3d(g, tube) for g in gt[key]]
                max_index = np.argmax(ious)

                if ious[max_index] >= threshold:
                    # match
                    is_positive = True

                    # remove the used GT
                    del gt[key][max_index]
                    if len(gt[key]) == 0:
                        del gt[key]

            if is_positive:
                tp += 1
                fn -= 1
            else:
                fp += 1

            precision_recall[i + 1, 0] = tp / max(1, (tp + fp))
            precision_recall[i + 1, 1] = tp / max(1, (tp + fn))

        results.append(pr_to_ap(precision_recall))

    video_ap_result = np.mean(results * 100)
    return video_ap_result
