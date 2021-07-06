import random

import cv2
from mmcv import Config

from mmaction.datasets import build_dataset

# import mmaction
# import importlib
# importlib.reload(mmaction)

cfg = Config.fromfile('/home/ubuntu/mmaction2/configs/detection/tube/jhmdb.py')
dataset = build_dataset(cfg.data.train, dict(test_mode=False))


def _darknet_draw_bbox(bboxes,
                       labels,
                       img,
                       scores=None,
                       bboxes_color=(0, 255, 0),
                       bboxes_thickness=1,
                       text_color=(0, 255, 0),
                       text_thickness=2,
                       text_front_scale=0.5):
    """bbox的形式是 xyxy，取值范围是像素的值 labels是标签名称 scores是置信度，[0, 1]的浮点数."""
    img = img.copy()
    for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
        print(bbox)
        xmin, ymin, xmax, ymax = bbox
        pt1 = (int(xmin), int(ymin))  # 左下
        pt2 = (int(xmax), int(ymax))  # 右上

        # 画bbox
        cv2.rectangle(img, pt1, pt2, bboxes_color, bboxes_thickness)

        # 写上对应的文字
        cur_label = label
        if scores is not None:
            cur_label += ' [' + str(round(scores[idx] * 100, 2)) + ']'
        cv2.putText(
            img=img,
            text=cur_label,
            org=(pt1[0], pt1[1] + 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=.5,
            color=(0, 255, 0),
            thickness=2,
        )
    return img


def draw_imgs(imgs, gt_bboxes):
    res = []
    for label_id in gt_bboxes:
        for tube in gt_bboxes[label_id]:
            for bbox, img in zip(tube, imgs):
                res.append(
                    _darknet_draw_bbox([bbox], [dataset.labels[label_id]],
                                       img))
    return res


length = len(dataset)
while True:
    idx = 0
    idx = random.randint(0, length)
    r = dataset[idx]
    r['imgs'] = r['imgs'].transpose([0, 2, 3, 1])
    res = draw_imgs(r['imgs'], r['gt_bboxes'])
    for img in res:
        print(img.shape)
        cv2.imshow('demo', img[:, :, ::-1])
        cv2.waitKey(0)

    break
