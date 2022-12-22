import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .ops import meta_bn, meta_linear, meta_conv2d

class MetalearningHead(nn.Layer):
    def __init__(self, num_classes, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        norm_opt = dict()
        norm_opt['BN_AFFINE'] = True
        norm_opt['BN_RUNNING'] = True
        norm_opt['IN_AFFINE'] = True
        norm_opt['IN_RUNNING'] = False

        norm_opt['BN_W_FREEZE'] = False
        norm_opt['BN_B_FREEZE'] = False
        norm_opt['IN_W_FREEZE'] = False
        norm_opt['IN_B_FREEZE'] = False

        norm_opt['BIN_INIT'] = 'one'
        norm_opt['IN_FC_MULTIPLY'] = 0.0
        # pool_type = 'avgpool'
        self.pool_layer = nn.AdaptiveAvgPool2D(1)
        in_feat = 2048
       # self.classifier_norm = meta_norm('BN', in_feat, norm_opt=norm_opt, bias_freeze=True)
        self.classifier_norm = meta_bn(in_feat, norm_opt=norm_opt, bias_freeze=True)

        num_classes = num_classes
        # cls_type = linear
        self.classifier_fc = meta_linear(in_feat, num_classes, weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(0, 0.001)), bias_attr=False)

    def forward(self, features, targets=None, opt=None):
        global_feat = self.pool_layer(features)
        bn_feat = self.classifier_norm(global_feat, opt)
        if len(bn_feat.shape) == 4:
            bn_feat = bn_feat.flatten(1)
        if not self.training: return bn_feat
        cls_outputs = self.classifier_fc(bn_feat, opt)
        pred_class_logits = F.linear(bn_feat, self.classifier_fc.weight) # compute accuracy
        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": pred_class_logits,
            "pooled_features": global_feat.flatten(1),
            "bn_features": bn_feat,
        }

if __name__ == '__main__':
    inputs = paddle.rand((5, 2048, 16, 8))
    model = MetalearningHead()
    for k, v in model(inputs).items():
        print(k+ ":", v.shape)
