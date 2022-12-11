import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import param_init
import math
from .ops import meta_norm, meta_conv2d


class BasicBlock(nn.Layer):
    expansion = 1
    def __init__(self, in_channels, hidden_channels, bn_norm, norm_opt, stride=1, 
                downsample=None, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.conv1 = meta_conv2d(in_channels, hidden_channels, kernel_size = 3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = meta_norm(bn_norm, hidden_channels, norm_opt)
        self.conv2 = meta_conv2d(hidden_channels, hidden_channels, kernel_size = 3, stride=1, padding=1, bias_attr=False)
        self.bn2 = meta_norm(bn_norm, hidden_channels, norm_opt)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, opt = None):
        identity = x
        out = self.conv1(x, opt)
        out = self.bn1(out, opt)
        out = self.relu(out)
        out = self.conv2(out, opt)
        out = self.bn2(out, opt)
        if self.downsample is not None:
            identity = self.downsample(x, opt)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Layer):
    expansion = 4
    def __init__(self, in_channels, hidden_channels, bn_norm, norm_opt, stride=1,
                downsample=None, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.conv1 = meta_conv2d(in_channels, hidden_channels, kernel_size=1, bias_attr=False)
        self.bn1 = meta_norm(bn_norm, hidden_channels, norm_opt)
        self.conv2 = meta_conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn2 = meta_norm(bn_norm, hidden_channels, norm_opt)
        self.conv3 = meta_conv2d(hidden_channels, hidden_channels * self.expansion, kernel_size=1, bias_attr=False)
        self.bn3 = meta_norm(bn_norm, hidden_channels * self.expansion, norm_opt)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, opt = None):
        identity = x
        out = self.conv1(x, opt)
        out = self.bn1(out, opt)
        out = self.relu(out)
        out = self.conv2(out, opt)
        out = self.bn2(out, opt)
        out = self.relu(out)
        out = self.conv3(out, opt)
        out = self.bn3(out, opt)

        if self.downsample is not None:
            identity = self.downsample(x, opt)
        out += identity
        out = self.relu(out)
        return out


class Downsample(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, enable_bias, bn_norm, norm_opt, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.conv = meta_conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias_attr=enable_bias)
        self.bn = meta_norm(bn_norm, out_channels, norm_opt)

    def forward(self, x, opt = None):
        x = self.conv(x, opt)
        out = self.bn(x, opt)
        return out


class ResNet(nn.Layer):
    def __init__(self, last_stride, bn_norm, norm_opt, block, layers, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.in_channels = 64
        self.conv1 = meta_conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = meta_norm(bn_norm, 64, norm_opt)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, norm_opt)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, norm_opt)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, norm_opt)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, norm_opt)
        self.random_init()
    
    def _make_layer(self, block, hidden_channels, n_repeats, stride, bn_norm, norm_opt):
        downsample = None
        if stride != 1 or self.in_channels != hidden_channels * block.expansion:
            downsample = Downsample(self.in_channels, hidden_channels * block.expansion, 1, stride, False, bn_norm, norm_opt)
        layers = []
        layers.append(block(self.in_channels, hidden_channels, bn_norm, norm_opt, stride, downsample))
        self.in_channels = hidden_channels * block.expansion
        for _ in range(1, n_repeats):
            layers.append(block(self.in_channels, hidden_channels, bn_norm, norm_opt))
        return nn.Sequential(*layers)

    def forward(self, x, opt = None):
        x = self.conv1(x, opt)
        x = self.bn1(x, opt)
        x = self.relu(x)
        x = self.maxpool(x)
        for layer in self.layer1._sub_layers.values():
            x = layer(x, opt)
        for layer in self.layer2._sub_layers.values():
            x = layer(x, opt)
        for layer in self.layer3._sub_layers.values():
            x = layer(x, opt)
        for layer in self.layer4._sub_layers.values():
            x = layer(x, opt)
        return x


    def random_init(self):
        for _, m in self.named_children():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                param_init.normal_init(m.weight, mean=0.0, std=math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2D):
                param_init.constant_init(m.weight, 1)
                param_init.constant_init(m.bias, 0)

def build_resnet_backbone():
    # M-resnet50
    pretrain = True     # cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = ''  # cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride = 1     # cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm = 'BIN_gate2'      # cfg.MODEL.NORM.TYPE_BACKBONE
    norm_opt = dict()
    norm_opt['BN_AFFINE'] = True    # cfg.MODEL.NORM.BN_AFFINE
    norm_opt['BN_RUNNING'] = True   # cfg.MODEL.NORM.BN_RUNNING
    norm_opt['BN_W_FREEZE'] = False # cfg.MODEL.NORM.BN_W_FREEZE
    norm_opt['BN_B_FREEZE'] = False # cfg.MODEL.NORM.BN_B_FREEZE

    norm_opt['IN_AFFINE'] = True    # cfg.MODEL.NORM.IN_AFFINE
    norm_opt['IN_RUNNING'] = False  # cfg.MODEL.NORM.IN_RUNNING
    norm_opt['IN_W_FREEZE'] = False # cfg.MODEL.NORM.IN_W_FREEZE
    norm_opt['IN_B_FREEZE'] = False # cfg.MODEL.NORM.IN_B_FREEZE

    norm_opt['BIN_INIT'] = 'one'    # cfg.MODEL.NORM.BIN_INIT
    norm_opt['IN_FC_MULTIPLY'] = 0.0    # cfg.MODEL.NORM.IN_FC_MULTIPLY
    num_splits = 1      # cfg.MODEL.BACKBONE.NORM_SPLIT
    #with_ibn = False    # cfg.MODEL.BACKBONE.WITH_IBN
    #with_se = False     # cfg.MODEL.BACKBONE.WITH_SE
    #with_nl = False     # cfg.MODEL.BACKBONE.WITH_NL
    depth = 50      # cfg.MODEL.BACKBONE.DEPTH

    num_blocks_per_stage = {18: [2,2,2,2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], }[depth]
    block = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152:Bottleneck,}[depth]
    model = ResNet(last_stride, bn_norm, norm_opt, block, num_blocks_per_stage)
    return model


if __name__ == '__main__':
    inputs = paddle.rand((5, 3, 256, 128))
    model = build_resnet_backbone()
    print(model(inputs).shape)
    print()
