# TPN

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{yang2020tpn,
  title={Temporal Pyramid Network for Action Recognition},
  author={Yang, Ceyuan and Xu, Yinghao and Shi, Jianping and Dai, Bo and Zhou, Bolei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020},
}
```

## 模型库

### Kinetics-400

|配置文件 | 分辨率 | GPU 数量 | 主干网络 | 预训练 | top1 准确率 | top5 准确率 | 参考代码的 top1 准确率 | 参考代码的 top5 准确率 | 推理时间 (video/s) | GPU 显存占用 (M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tpn_slowonly_r50_8x8x1_150e_kinetics_rgb](/configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py)|短边 320|8x4| ResNet50 | ImageNet | 73.10 | 91.03 | x | x | x | 6916 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb_20200910-b796d7a0.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb/20200910_134330.log) | [json](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb/20200910_134330.log.json) |
|[tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb](/configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.py)|短边 320|8x4| ResNet50 | ImageNet | 76.20 | 92.44 | [75.49](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md) | [92.05](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md) | x | 6916 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb_20200923-52629684.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/20200923_151919.log) | [json](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/20200923_151919.log.json) |

### Something-Something V1

|配置文件 | GPU 数量 | 主干网络 | 预训练 | top1 准确率| top5 准确率 | GPU 显存占用 (M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tpn_tsm_r50_1x1x8_150e_sthv1_rgb](/configs/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb.py)|height 100|8x6| ResNet50 | TSM | 50.80 | 79.05 | 8828 |[ckpt](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb/tpn_tsm_r50_1x1x8_150e_sthv1_rgb_20210311-28de4cd5.pth) |[log](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb/20210311_162636.log)|[json](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb/20210311_162636.log.json)|

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 这里的 **推理时间** 是根据 [基准测试脚本](/tools/analysis/benchmark.py) 获得的，采用测试时的采帧策略，且只考虑模型的推理时间，
   并不包括 IO 时间以及预处理时间。对于每个配置，MMAction2 使用 1 块 GPU 并设置批大小（每块 GPU 处理的视频个数）为 1 来计算推理时间。
3. 参考代码的结果是通过使用相同的模型配置在原来的代码库上训练得到的。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 TPN 模型在 Kinetics-400 数据集上的训练。

```shell
python tools/train.py configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py \
    --work-dir work_dirs/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb [--validate --seed 0 --deterministic]
```

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 Kinetics-400 数据集上测试 TPN 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips prob
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。