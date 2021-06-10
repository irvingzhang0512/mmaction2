import os.path as osp
import tempfile

import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys
from numpy.testing import assert_array_equal

from mmaction.datasets import TubeDataset
from .base import BaseTestDataset


class TestTubeDataset(BaseTestDataset):

    def test_tube_dataset(self):
        target_keys = [
            'indice', 'video', 'frame_dir', 'total_frames', 'resolution',
            'gt_bboxes'
        ]

        with pytest.raises(ValueError):
            TubeDataset(
                self.tube_ann_file,
                self.tube_pipeline,
                data_prefix=self.data_prefix,
                save_preload=True)

        tube_dataset = TubeDataset(
            self.tube_ann_file,
            self.tube_pipeline,
            data_prefix=self.data_prefix)
        assert hasattr(tube_dataset, 'gt_tubes')
        assert hasattr(tube_dataset, 'labels')
        assert hasattr(tube_dataset, 'videos')

        tube_info = tube_dataset.video_infos[0]
        assert assert_dict_has_keys(tube_info, target_keys)

        assert tube_info['video'] == 'WalkingWithDog/v_WalkingWithDog_g05_c02'
        assert tube_info['indice'] == (tube_info['video'], 1)
        assert tube_info['frame_dir'] == osp.join(self.data_prefix,
                                                  tube_info['video'])
        assert tube_info['total_frames'] == 240
        assert tube_info['resolution'] == (240, 320)
        assert list(tube_info['gt_bboxes']) == [23]
        assert_array_equal(tube_info['gt_bboxes'][23], [
            np.array([[77., 0., 185., 168.], [77., 0., 185., 168.],
                      [77., 0., 185., 168.], [77., 0., 185., 168.],
                      [77., 0., 185., 168.], [77., 0., 185., 168.],
                      [77., 0., 185., 168.]],
                     dtype=np.float32)
        ])

        test_tube_dataset = TubeDataset(
            self.tube_ann_file,
            self.tube_pipeline,
            test_mode=True,
            data_prefix=self.data_prefix)
        test_tube_info = test_tube_dataset[0]
        assert assert_dict_has_keys(test_tube_info, target_keys)

        assert test_tube_info['video'] == 'SalsaSpin/v_SalsaSpin_g11_c02'
        assert test_tube_info['indice'] == (test_tube_info['video'], 7)
        assert test_tube_info['frame_dir'] == osp.join(self.data_prefix,
                                                       test_tube_info['video'])
        assert test_tube_info['total_frames'] == 167
        assert test_tube_info['resolution'] == (240, 320)
        assert list(test_tube_info['gt_bboxes']) == [14]
        assert_array_equal(test_tube_info['gt_bboxes'][14], [
            np.array([[125., 60., 192., 211.], [125., 60., 192., 211.],
                      [127., 61., 191., 212.], [127., 61., 188., 212.],
                      [134., 61., 195., 212.], [133., 61., 194., 212.],
                      [133., 61., 194., 212.]],
                     dtype=np.float32)
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            preload_video_infos = osp.join(tmpdir, 'tmp.pkl')
            TubeDataset(
                self.tube_ann_file,
                self.tube_pipeline,
                data_prefix=self.data_prefix,
                save_preload=True,
                preload_video_infos=preload_video_infos)
            assert osp.exists(preload_video_infos)

            TubeDataset(
                self.tube_ann_file,
                self.tube_pipeline,
                data_prefix=self.data_prefix,
                preload_video_infos=preload_video_infos)

    def test_tube_pipeline(self):
        target_keys = [
            'indice', 'video', 'frame_dir', 'total_frames', 'resolution',
            'gt_bboxes', 'filename_tmpl', 'modality', 'start_index',
            'tube_length', 'num_classes'
        ]

        tube_dataset = TubeDataset(
            self.tube_ann_file,
            self.tube_pipeline,
            data_prefix=self.data_prefix)
        tube_result = tube_dataset[0]
        assert assert_dict_has_keys(tube_result, target_keys)

        assert len(tube_dataset) == 450
        assert tube_result['filename_tmpl'] == '{:05}.jpg'
        assert tube_result['modality'] == 'RGB'
        assert tube_result['start_index'] == 1
        assert tube_result['tube_length'] == 7
        assert tube_result['num_classes'] == 24

        test_tube_dataset = TubeDataset(
            self.tube_ann_file,
            self.tube_pipeline,
            test_mode=True,
            data_prefix=self.data_prefix)
        test_tube_result = test_tube_dataset[0]
        assert assert_dict_has_keys(test_tube_result, target_keys)

        assert len(test_tube_result) == 11
        assert test_tube_result['filename_tmpl'] == '{:05}.jpg'
        assert test_tube_result['modality'] == 'RGB'
        assert test_tube_result['start_index'] == 1
        assert test_tube_result['tube_length'] == 7
        assert test_tube_result['num_classes'] == 24

    def test_tubelet_in_tube(self):
        video_tube = np.array([[4, 1], [5, 2], [6, 3]], dtype=np.int)
        assert TubeDataset.tubelet_in_tube(video_tube, 4, 3)
        assert not TubeDataset.tubelet_in_tube(video_tube, 1, 3)

        video_tube = np.array([[4, 1], [1, 2], [6, 3]], dtype=np.int)
        assert not TubeDataset.tubelet_in_tube(video_tube, 4, 3)
        assert not TubeDataset.tubelet_in_tube(video_tube, 1, 3)

    def test_tubelet_out_tube(self):
        video_tube = np.array([[4, 1], [5, 2], [6, 3]], dtype=np.int)
        assert TubeDataset.tubelet_out_tube(video_tube, 1, 3)
        assert not TubeDataset.tubelet_out_tube(video_tube, 4, 3)

        video_tube = np.array([[4, 1], [1, 2], [6, 3]], dtype=np.int)
        assert not TubeDataset.tubelet_out_tube(video_tube, 1, 3)
        assert not TubeDataset.tubelet_out_tube(video_tube, 4, 3)
