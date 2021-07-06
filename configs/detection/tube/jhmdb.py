dataset_type = 'TubeDataset'
data_root = '/home/ubuntu/data/JHMDB/Frames'
anno_root = '/home/ubuntu/data/JHMDB'

ann_file_train = f'{anno_root}/JHMDB-GT.pkl'
ann_file_val = ann_file_train

train_pipeline = [
    dict(type='TubeSampleFrames'),
    dict(type='TubeDecode'),
    dict(
        type='CuboidCrop',
        cuboid_settings=[
            dict(
                max_trials=5,
                max_sample=5,
                sampler=dict(
                    min_scale=0.8, max_scale=1.2, min_aspect=1.,
                    max_aspcet=1.),
                constraints=dict(
                    min_jaccard_overlap=0.4, max_jaccard_overlap=0.4))
        ]),
    dict(type='TubePad', expand_ratio=0.5, max_expand_ratio=4),
    dict(type='TubeResize', resize_scale=(120, 160)),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatTubeShape')
]

val_pipeline = [
    dict(type='TubeSampleFrames'),
    dict(type='TubeDecode'),
    dict(
        type='CuboidCrop',
        cuboid_settings=[
            dict(
                max_trials=5,
                max_sample=5,
                sampler=dict(
                    min_scale=1., max_scale=1., min_aspect=1., max_aspcet=1.),
                constraints=dict(min_jaccard_overlap=0, max_jaccard_overlap=0))
        ]),
    dict(type='TubePad', expand_ratio=0.5, max_expand_ratio=4),
    dict(type='TubeResize', resize_scale=(256, 320)),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatTubeShape')
]

data = dict(
    videos_per_gpu=9,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root,
        preload_video_infos=None,
        save_preload=False,
        num_classes=24,
        test_mode=False,
        encoding='iso-8859-1',
        filename_tmpl='{:05}.png',
        split=1,
        tube_length=7,
        start_index=1,
        modality='RGB'),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root,
        preload_video_infos=None,
        save_preload=False,
        num_classes=24,
        test_mode=False,
        encoding='iso-8859-1',
        filename_tmpl='{:05}.png',
        split=1,
        tube_length=7,
        start_index=1,
        modality='RGB'))
data['test'] = data['val']
