import copy
import os.path as osp
import pickle
from collections import defaultdict

from ..utils import get_root_logger
from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class TubeDataset(BaseDataset):
    """Tube dataset for frame-level spatial temporal detection.

    Based on original "UCF101-24", "JHMDB" and "MultiSports" annotation files,
    the dataset loads:
    1. labels ('labels'): a list of class names(strings)
    2. ground truth tubes ('gttubes'): a dict with format
        ``{frames_relative_path: {label_id: list[ndarray]}}``, where ndarray's
        shape is (n, 5) with format (frame_ind, x1, y1, x2, y2).
    3. frames number for each video ('nframes'): a dict with format
        ``{frames_relative_path: num_of_frames(int)}``
    4. train video file list ('train_videos'): a list of frames relative path
    5. test video file list ('test_videos'): a list of frames relative path
    6. resolution for each video ('resolution'): a list of tuple with format
        (width, height)

    This dataset applies specified transformations to return a dict containing
    frame tensors and other information.

    Specifically, it can save arranged information into a pickle file to
    accelerate loading.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        preload_video_infos (str | None): Path to a pickle file, which has
            saved arranged information. Default: None.
        save_preload (bool): Whether to save the arranged information to a
            file. Default: False.
        num_classes (int): Number class of the dataset. Default: 24.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        encoding (str): Encode style to load pickle file.
            Default: 'iso-8859-1'.
        filename_tmpl (str): Template for each filename. Default: '{:05}.jpg'.
        split (int): Index of split to indicate the data part for train and
            test videos. Default: 1.
        tube_length (int): Length of tube to form the tubelet. Default: 7.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 preload_video_infos=None,
                 save_preload=False,
                 num_classes=24,
                 data_prefix=None,
                 test_mode=False,
                 encoding='iso-8859-1',
                 filename_tmpl='{:05}.jpg',
                 split=1,
                 tube_length=7,
                 start_index=1,
                 modality='RGB'):
        self.preload_video_infos = preload_video_infos
        self.save_preload = save_preload
        self.filename_tmpl = filename_tmpl
        self.split = split
        self.tube_length = tube_length
        self.encoding = encoding
        self.logger = get_root_logger()
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode=test_mode,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality)

    @staticmethod
    def tubelet_in_tube(video_tube, frame_index, tube_length):
        return all([
            i in video_tube[:, 0]
            for i in range(frame_index, frame_index + tube_length)
        ])

    @staticmethod
    def tubelet_out_tube(video_tube, frame_index, tube_length):
        return all([
            i not in video_tube[:, 0]
            for i in range(frame_index, frame_index + tube_length)
        ])

    def check_tubelet(self, video_tubes, frame_index, tube_length):
        # 要求所有 tubelet，要么全部帧都属于 gt_tube，要么全部帧都不属于 gt_tube
        is_whole = all([
            self.tubelet_in_tube(video_tube, frame_index, tube_length)
            or self.tubelet_out_tube(video_tube, frame_index, tube_length)
            for video_tube in video_tubes
        ])
        has_gt = any([
            self.tubelet_in_tube(tube, frame_index, tube_length)
            for tube in video_tubes
        ])
        return is_whole and has_gt

    def load_annotations(self):
        # TODO: Double check whether to remove encoding and preprocess
        # annotation file during prprocessing
        pkl_data = pickle.load(
            open(self.ann_file, 'rb'), encoding=self.encoding)
        gt_tubes = pkl_data['gttubes']
        num_frames = pkl_data['nframes']
        train_videos = pkl_data['train_videos']
        test_videos = pkl_data['test_videos']
        resolution = pkl_data['resolution']

        self.gt_tubes = gt_tubes
        self.labels = pkl_data['labels']

        assert len(train_videos[self.split - 1]) + len(
            test_videos[self.split - 1]) == len(num_frames)
        videos = train_videos[
            self.split - 1] if not self.test_mode else test_videos[self.split -
                                                                   1]
        self.videos = videos

        if self.preload_video_infos is not None and osp.exists(
                self.preload_video_infos):
            # load preload_video_infos
            video_infos = pickle.load(open(self.preload_video_infos, 'rb'))
        else:
            # read video_infors from raw annotation file
            video_infos = []

            for video in videos:
                # 获取当前食品的所有 tubes
                video_tubes = sum(gt_tubes[video].values(), [])

                # 遍历当前视频所有长度为 tube_length 的小片段，即遍历所有 tubelet
                for i in range(1, num_frames[video] + 2 - self.tube_length):

                    if self.check_tubelet(video_tubes, i, self.tube_length):
                        # 如果 tubelet 符合要求，才会进行下一步
                        # 要求就是，tubelet 与所有 gt tubes 比较，
                        # tubelet 要么全部属于 gt tubes，要么全部不属于
                        # 不允许存在交叉情况，即部分属于、部分不属于
                        frame_dir = video
                        if self.data_prefix is not None:
                            frame_dir = osp.join(self.data_prefix, video)

                        # 构建当前 tubelet 的 gt
                        # 即遍历当前视频所有 gt tubes，如果当前 tubelet 的起始帧（即 i）
                        # 存在于某个gt tube中，则将所有在当前 tubelet 范围内的bboxes保存
                        # gt_bboxes 的格式就是 {label_index: list[ndarray]}，其中
                        # ndarray 的shape是 [tube_length, 4]
                        gt_bboxes = defaultdict(list)
                        for label_index, tubes in gt_tubes[video].items():
                            for tube in tubes:
                                if i not in tube[:, 0]:
                                    continue
                                # * 是按位与
                                # 功能就是获取所有满足两个条件的所有帧编号，并提取bboxes保存
                                boxes = tube[
                                    (tube[:, 0] >= i) *
                                    (tube[:, 0] < i + self.tube_length), 1:5]
                                gt_bboxes[label_index].append(boxes)
                        video_info = {}
                        video_info['indice'] = (video, i)
                        video_info['video'] = video
                        video_info['frame_dir'] = frame_dir
                        video_info['total_frames'] = num_frames[video]
                        video_info['resolution'] = resolution[video]
                        video_info['gt_bboxes'] = gt_bboxes
                        video_infos.append(video_info)

            if self.save_preload:
                # TODO: Seems different tube length may cause some difference
                # to the videos infos, can we do the other parts of work during
                # preprocessing, and let load_annotation focus on processing
                # with different tube_length?
                if self.preload_video_infos is None:
                    raise ValueError('preload annotation file should be '
                                     'assigned for saving')
                self.logger.info(
                    f'Save tube info to {self.preload_video_infos}')
                pickle.dump(video_infos, open(self.preload_video_infos, 'wb'))

        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['tube_length'] = self.tube_length
        results['num_classes'] = self.num_classes

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['tube_length'] = self.tube_length
        results['num_classes'] = self.num_classes

        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metrics,
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        # TODO: Add evluataiton codes for tube dataset
        # waiting for a model to predict tubelet
        raise NotImplementedError
