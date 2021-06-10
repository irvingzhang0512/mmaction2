import copy

import mmcv
import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mmaction.datasets.pipelines import EntityBoxFlip, Flip, TubeFlip
from .base import check_flip


class TestFlip:

    def test_flip(self):
        with pytest.raises(ValueError):
            # direction must be in ['horizontal', 'vertical']
            Flip(direction='vertically')

        target_keys = ['imgs', 'flip_direction', 'modality']

        # do not flip imgs.
        imgs = list(np.random.rand(2, 64, 64, 3))
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB')
        flip = Flip(flip_ratio=0, direction='horizontal')
        flip_results = flip(results)
        assert assert_dict_has_keys(flip_results, target_keys)
        assert np.array_equal(imgs, results['imgs'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert np.shape(flip_results['imgs']) == np.shape(imgs)

        # always flip imgs horizontally.
        imgs = list(np.random.rand(2, 64, 64, 3))
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB')
        results['gt_bboxes'] = np.array([[0, 0, 60, 60]])
        results['proposals'] = np.array([[0, 0, 60, 60]])
        flip = Flip(flip_ratio=1, direction='horizontal')
        flip_results = flip(results)
        assert assert_dict_has_keys(flip_results, target_keys)
        if flip_results['flip'] is True:
            assert check_flip(imgs, flip_results['imgs'],
                              flip_results['flip_direction'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert np.shape(flip_results['imgs']) == np.shape(imgs)

        # flip flow images horizontally
        imgs = [
            np.arange(16).reshape(4, 4).astype(np.float32),
            np.arange(16, 32).reshape(4, 4).astype(np.float32)
        ]
        results = dict(imgs=copy.deepcopy(imgs), modality='Flow')
        flip = Flip(flip_ratio=1, direction='horizontal')
        flip_results = flip(results)
        assert assert_dict_has_keys(flip_results, target_keys)
        imgs = [x.reshape(4, 4, 1) for x in imgs]
        flip_results['imgs'] = [
            x.reshape(4, 4, 1) for x in flip_results['imgs']
        ]
        if flip_results['flip'] is True:
            assert check_flip([imgs[0]],
                              [mmcv.iminvert(flip_results['imgs'][0])],
                              flip_results['flip_direction'])
            assert check_flip([imgs[1]], [flip_results['imgs'][1]],
                              flip_results['flip_direction'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert np.shape(flip_results['imgs']) == np.shape(imgs)

        # always flip imgs vertivally.
        imgs = list(np.random.rand(2, 64, 64, 3))
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB')
        flip = Flip(flip_ratio=1, direction='vertical')
        flip_results = flip(results)
        assert assert_dict_has_keys(flip_results, target_keys)
        if flip_results['flip'] is True:
            assert check_flip(imgs, flip_results['imgs'],
                              flip_results['flip_direction'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert np.shape(flip_results['imgs']) == np.shape(imgs)

        assert repr(flip) == (f'{flip.__class__.__name__}'
                              f'(flip_ratio={1}, direction=vertical, '
                              f'flip_label_map={None}, lazy={False})')

        # transform label for the flipped image with the specific label.
        _flip_label_map = {4: 6}
        imgs = list(np.random.rand(2, 64, 64, 3))

        # the label should be mapped.
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB', label=4)
        flip = Flip(
            flip_ratio=1,
            direction='horizontal',
            flip_label_map=_flip_label_map)
        flip_results = flip(results)
        assert results['label'] == 6

        # the label should not be mapped.
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB', label=3)
        flip = Flip(
            flip_ratio=1,
            direction='horizontal',
            flip_label_map=_flip_label_map)
        flip_results = flip(results)
        assert results['label'] == 3

        # flip the keypoints
        results = dict(
            keypoint=np.array([[1, 1], [63, 63]]).reshape([1, 1, 2, 2]),
            modality='Pose',
            img_shape=(64, 64))
        flip = Flip(
            flip_ratio=1, direction='horizontal', left_kp=[0], right_kp=[1])
        flip_results = flip(results)
        assert_array_almost_equal(flip_results['keypoint'][0, 0],
                                  np.array([[1, 63], [63, 1]]))

        results = dict(
            keypoint=np.array([[1, 1], [63, 63]]).reshape([1, 1, 2, 2]),
            modality='Pose',
            img_shape=(64, 64))
        flip = Flip(
            flip_ratio=1, direction='horizontal', left_kp=[], right_kp=[])
        flip_results = flip(results)
        assert_array_almost_equal(flip_results['keypoint'][0, 0],
                                  np.array([[63, 1], [1, 63]]))

        with pytest.raises(AssertionError):
            results = dict(
                keypoint=np.array([[1, 1], [63, 63]]).reshape([1, 1, 2, 2]),
                modality='Pose',
                img_shape=(64, 64))
            flip = Flip(
                flip_ratio=1, direction='vertical', left_kp=[], right_kp=[])
            flip_results = flip(results)

    def test_tube_flip(self):
        target_keys = ['imgs', 'flip_direction', 'modality', 'flip']

        # do not flip imgs
        imgs = list(np.random.rand(2, 64, 64, 3))
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB', flip=False)
        tube_flip = TubeFlip()
        tube_flip_results = tube_flip(results)
        assert assert_dict_has_keys(tube_flip_results, target_keys)
        assert_array_equal(tube_flip_results['imgs'], imgs)
        assert id(tube_flip_results['imgs']) == id(results['imgs'])

        # always flip imgs horizontally
        imgs = list(np.random.rand(2, 64, 64, 3))
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB', flip=True)
        tube_flip = TubeFlip(direction='horizontal')
        tube_flip_results = tube_flip(results)
        assert assert_dict_has_keys(tube_flip_results, target_keys)
        if tube_flip_results['flip'] is True:
            assert self.check_flip(imgs, tube_flip_results['imgs'],
                                   tube_flip_results['flip_direction'])
        assert id(tube_flip_results['imgs']) == id(results['imgs'])

        # flip flow images horizontally
        imgs = list(np.random.rand(2, 64, 64, 3))
        results = dict(imgs=copy.deepcopy(imgs), modality='Flow', flip=True)
        with pytest.raises(AssertionError):
            TubeFlip(direction='vertical')(results)
        tube_flip = TubeFlip(direction='horizontal')
        assert assert_dict_has_keys(tube_flip_results, target_keys)
        tube_flip_results = tube_flip(results)
        assert id(tube_flip_results['imgs']) == id(results['imgs'])

        # always flip imgs vertivally.
        imgs = list(np.random.rand(2, 64, 64, 3))
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB', flip=True)
        tube_flip = TubeFlip(direction='vertical')
        assert assert_dict_has_keys(tube_flip_results, target_keys)
        tube_flip_results = tube_flip(results)
        if tube_flip_results['flip'] is True:
            assert self.check_flip(imgs, tube_flip_results['imgs'],
                                   results['flip_direction'])
        assert id(tube_flip_results['imgs']) == id(results['imgs'])

    def test_box_flip(self):
        target_keys = ['flip', 'flip_direction', 'resolution', 'gt_bboxes']

        gt_bboxes = {
            23: [
                np.array([[77., 0., 185., 168.], [78., 1., 185., 168.],
                          [78., 1., 186., 169.]],
                         dtype=np.float32)
            ]
        }
        results = dict(
            flip=True,
            flip_direction='horizontal',
            resolution=(240, 320),
            gt_bboxes=gt_bboxes)
        gt_tube = gt_bboxes[23][0]

        def check_box_flip(boxes, direction, img_shape):
            img_h, img_w = img_shape
            for label_index in boxes:
                for tube in boxes[label_index]:
                    if direction == 'horizontal':
                        assert_array_equal(tube[:, 1], gt_tube[:, 1])
                        assert_array_equal(tube[:, 3], gt_tube[:, 3])
                        assert_array_equal(gt_tube[:, 0], img_w - tube[:, 2])
                        assert_array_equal(gt_tube[:, 2], img_w - tube[:, 0])
                    else:
                        assert_array_equal(tube[:, 0], gt_tube[:, 0])
                        assert_array_equal(tube[:, 2], gt_tube[:, 2])
                        assert_array_equal(gt_tube[:, 1], img_h - tube[:, 3])
                        assert_array_equal(gt_tube[:, 3], img_h - tube[:, 1])

        # TODO: double chedk
        box_flip = EntityBoxFlip()

        # do not flip
        results_ = copy.deepcopy(results)
        results_['flip'] = False
        assert id(results_) == id(box_flip(results_))
        assert assert_dict_has_keys(results_, target_keys)

        # flip horizontally
        results_ = copy.deepcopy(results)
        results_['flip'] = True
        results_ = box_flip(results_)
        check_box_flip(results_['gt_bboxes'], results_['flip_direction'],
                       results_['resolution'])
        assert assert_dict_has_keys(results_, target_keys)

        # flip vertically
        results_ = copy.deepcopy(results)
        results_['flip'] = True
        results_['flip_direction'] = 'vertical'
        results_ = box_flip(results_)
        check_box_flip(results_['gt_bboxes'], results_['flip_direction'],
                       results_['resolution'])
        assert assert_dict_has_keys(results_, target_keys)
