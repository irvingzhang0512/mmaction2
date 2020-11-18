from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x, labels.squeeze())
            x = x.squeeze(2)
            num_segs = 1

        cls_score = self.cls_head(x, num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        test_crops = self.test_cfg.get('test_crops', None)
        twice_sample = self.test_cfg.get('twice_sample', False)

        batches = imgs.shape[0]

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x)
            x = x.squeeze(2)
            losses.update(loss_aux)
            num_segs = 1

        # 这个结果不一定是 [batch_size, num_classes]
        # 对于TSN，不管训练还是预测，都是固定是 [batch_size, num_classes]
        # 对于TSM的训练阶段，如果参数没有设置错，那就是 `[batch_size, num_classes]`
        # 对于TSM的测试阶段
        #   如果参数类似于 1x1x8，那输出的是 [batch_size*num_crops, num_classes]
        #   如果参数类似于 8x8x1，那输出的是 [batch_size*num_crops*num_clips, num_classes]
        cls_score = self.cls_head(x, num_segs)

        # 为了确保将输出结果转换为 [batch_size, num_classes]，
        # 这里就需要通过参数来设置，即，根据上面TSM的情况，要求满足
        # test_crops == num_crops 或 test_crops == num_crops * num_clips
        if test_crops is not None:
            if twice_sample:
                test_crops = test_crops * 2
            cls_score = self.average_clip(cls_score, test_crops)

        return cls_score.cpu().numpy()

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        outs = (self.cls_head(x, num_segs), )
        return outs

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        test_crops = self.test_cfg.get('test_crops', None)
        twice_sample = self.test_cfg.get('twice_sample', False)

        batches = imgs.shape[0]

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x)
            x = x.squeeze(2)
            losses.update(loss_aux)
            num_segs = 1

        cls_score = self.cls_head(x, num_segs)
        if test_crops is not None:
            if twice_sample:
                test_crops = test_crops * 2
            cls_score = self.average_clip(cls_score, test_crops)

        return cls_score
