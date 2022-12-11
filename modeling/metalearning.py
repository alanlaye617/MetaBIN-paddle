import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .resnet import build_resnet_backbone
from .heads import MetalearningHead

class Metalearning(nn.Layer):
    def __init__(self, pixel_mean, pixel_std):
        self.register_buffer('pixel_mean', paddle.Tensor(pixel_mean)).view((1, -1, 1, 1))
        self.register_buffer('pixel_mean', paddle.Tensor(pixel_std)).view((1, -1, 1, 1))
        self.backbone = build_resnet_backbone() # resnet-50
        self.heads = MetalearningHead()

    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, opt=None):
        if self.training:
            outs = dict()
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            outs['targets'] = batched_inputs["targets"].long().to(self.device)
            if 'others' in batched_inputs.keys():
                assert "others" in batched_inputs, "View ID annotation are missing in training!"
                assert "domains" in batched_inputs['others'], "View ID annotation are missing in training!"
                outs['domains'] = batched_inputs['others']['domains'].long().to(self.device)
            if outs['targets'].sum() < 0: outs['targets'].zero_()

            features = self.backbone(images, opt)
            result = self.heads(features, outs['targets'], opt)

            outs['outputs'] = result

            return outs
        else:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images, opt)
            return self.heads(features)

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        images = batched_inputs["images"].to(self.device)
        # images = batched_inputs
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

pixel_mean = [123.675, 116.28, 103.53]
pixel_std = [58.395, 57.120000000000005, 57.375]