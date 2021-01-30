import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys
from numpy.testing import assert_array_almost_equal

from mmaction.datasets.pipelines import CenterCrop, Albu
from .base import check_flip


class TestAugumentations:

    def test_albu(self):

        with pytest.raises(ValueError):
            # transforms only support one string, 'default'
            Albu(transforms='test')

        with pytest.raises(ValueError):
            # transforms only support string or list of dicts
            # or A.ReplayCompose object
            Albu(transforms=dict(type='Rotate'))

        with pytest.raises(AssertionError):
            # each dict must have a `type` key
            Albu(transforms=[dict(rotate=(-30, 30))])

        with pytest.raises(AttributeError):
            # `type` must be available in albu
            Albu(transforms=[dict(type='BlaBla')])

        with pytest.raises(TypeError):
            # `type` must be str or valid albu type
            Albu(transforms=[dict(type=CenterCrop)])

        import albumentations as albu

        # check default configs
        target_keys = ['imgs', 'img_shape', 'modality']
        imgs = list(np.random.randint(0, 255, (1, 64, 64, 3)).astype(np.uint8))
        results = dict(imgs=imgs, modality='RGB')
        default_albu = Albu(transforms='default')
        default_results = default_albu(results)
        assert_dict_has_keys(default_results, target_keys)
        assert default_results['img_shape'] == (64, 64)

        # check flip (both images and bboxes)
        target_keys = ['imgs', 'gt_bboxes', 'proposals', 'img_shape']
        imgs = list(np.random.rand(1, 64, 64, 3).astype(np.float32))
        results = dict(
            imgs=imgs,
            modality='RGB',
            proposals=np.array([[0, 0, 25, 35]]),
            img_shape=(64, 64),
            gt_bboxes=np.array([[0, 0, 25, 35]]))
        albu_flip = Albu(transforms=[dict(type='HorizontalFlip', p=1)])
        flip_results = albu_flip(results)
        assert assert_dict_has_keys(flip_results, target_keys)
        assert check_flip(imgs, flip_results['imgs'], 'horizontal')
        assert_array_almost_equal(flip_results['gt_bboxes'],
                                  np.array([[39, 0, 64, 35]]))
        assert_array_almost_equal(flip_results['proposals'],
                                  np.array([[39, 0, 64, 35]]))
        transforms = albu.ReplayCompose([albu.HorizontalFlip(p=1)],
                                        bbox_params=albu.BboxParams(
                                            format='pascal_voc'))
        assert repr(albu_flip) == f'Albu(transforms={transforms})'

        # check crop (both images and bboxes)
        target_keys = ['crop_bbox', 'gt_bboxes', 'imgs', 'img_shape']
        imgs = list(np.random.rand(1, 122, 122, 3))
        results = dict(
            imgs=imgs,
            modality='RGB',
            img_shape=(122, 122),
            gt_bboxes=np.array([[1.5, 2.5, 110, 64]]))
        albu_center_crop = Albu(
            transforms=[dict(type=albu.CenterCrop, width=100, height=100)])
        crop_results = albu_center_crop(results)
        assert_dict_has_keys(crop_results, target_keys)
        assert_array_almost_equal(crop_results['gt_bboxes'],
                                  np.array([[0., 0., 99., 53.]]))
        assert 'proposals' not in results
        transforms = albu.ReplayCompose(
            [albu.CenterCrop(width=100, height=100)],
            bbox_params=albu.BboxParams(format='pascal_voc'))
        assert repr(albu_center_crop) == f'Albu(transforms={transforms})'

        # check resize (images only)
        target_keys = ['imgs', 'img_shape']
        imgs = list(np.random.rand(1, 64, 64, 3))
        results = dict(imgs=imgs, modality='RGB')
        transforms = albu.ReplayCompose([albu.Resize(height=32, width=32)],
                                        bbox_params=albu.BboxParams(
                                            format='pascal_voc'))
        albu_resize = Albu(transforms=transforms)
        resize_results = albu_resize(results)
        assert_dict_has_keys(resize_results, target_keys)
        assert resize_results['img_shape'] == (32, 32)
        assert repr(albu_resize) == f'Albu(transforms={transforms})'
