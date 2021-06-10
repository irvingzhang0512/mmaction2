import copy

import numpy as np
from mmcv.utils import assert_dict_has_keys
from numpy.testing import assert_array_equal

from mmaction.datasets.pipelines import TubeDecode, TubeSampleFrames
from .base import BaseTestLoading


class TestTubeLoading(BaseTestLoading):

    def test_tube_sample_frames(self):
        target_keys = [
            'indice', 'total_frames', 'modality', 'tube_length', 'frame_inds'
        ]

        results = dict(
            indice=('video', 70),
            total_frames=100,
            modality='RGB',
            tube_length=7)

        tube_sample_frames = TubeSampleFrames()
        results = tube_sample_frames(results)
        assert assert_dict_has_keys(results, target_keys)
        assert_array_equal(results['frame_inds'], np.arange(70, 77))

        results['modality'] = 'Flow'
        results = tube_sample_frames(results)
        assert assert_dict_has_keys(results, target_keys)
        assert_array_equal(results['frame_inds'], np.arange(70, 81))

        results['modality'] = 'RGB'
        results['total_frames'] = 73
        results = tube_sample_frames(results)
        assert assert_dict_has_keys(results, target_keys)
        assert_array_equal(results['frame_inds'], [70, 71, 72, 73, 73, 73, 73])

    def test_tube_decode(self):
        target_keys = [
            'frame_dir', 'filename_tmpl', 'frame_inds', 'imgs',
            'original_shape', 'img_shape'
        ]

        # test frame selector with 2 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(0, self.total_frames, 2)[:,
                                                                  np.newaxis]
        # since the test images start with index 1, we plus 1 to frame_inds
        # in order to pass the CI
        inputs['frame_inds'] = inputs['frame_inds'] + 1
        tube_decode = TubeDecode(io_backend='disk')
        results = tube_decode(inputs)
        assert assert_dict_has_keys(results, target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)
        assert results['img_shape'] == (240, 320)
