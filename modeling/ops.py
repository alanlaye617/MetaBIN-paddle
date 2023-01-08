import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math


class meta_linear(nn.Linear):
    def __init__(self, in_features, out_features, weight_attr=None, bias_attr=None, compute_meta_params=False, name=None):
        if weight_attr is None:
            weight_attr=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity='leaky_relu')
        super().__init__(in_features, out_features, weight_attr, bias_attr, name)
        self.compute_meta_params = compute_meta_params

    def forward(self, input, opt = None):
        param_update = (opt != None) and opt.get('param_update', False) and self.compute_meta_params and (self.weight is not None)

        if param_update:
            if (self.weight is not None) and (not self.weight.stop_gradient) and (self.weight.grad is not None):
                lr_w = opt['meta_ratio'] * opt['main_lr'] * self.weight.optimize_attr.get('learning_rate', 1.0)
                updated_weight = self.weight - lr_w * self.weight.grad
            else:
                updated_weight = self.weight
            if (self.bias is not None) and (not self.bias.stop_gradient) and (self.bias.grad is not None):
                lr_b = opt['meta_ratio'] * opt['main_lr'] * self.bias.optimize_attr.get('learning_rate', 1.0)
                updated_bias = self.bias - lr_b * self.bias.grad
            else:
                updated_bias = self.bias
            return F.linear(input, updated_weight, updated_bias)
        else:
            return F.linear(input, self.weight, self.bias)


class meta_conv2d(nn.Conv2D):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, padding_mode='zeros',
                weight_attr=None, bias_attr=None, compute_meta_params=False, data_format="NCHW"):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, padding_mode, 
                        weight_attr, bias_attr, data_format)
        self.compute_meta_params = compute_meta_params

    def forward(self, inputs, opt = None):
        param_update = (opt != None) and opt.get('param_update', False) and self.compute_meta_params and (self.weight is not None)

        if param_update:
            if (self.weight is not None) and (not self.weight.stop_gradient) and (self.weight.grad is not None):
                lr_w = opt['meta_ratio'] * opt['main_lr'] * self.weight.optimize_attr.get('learning_rate', 1.0)
                updated_weight = self.weight - lr_w * self.weight.grad
            else:
                updated_weight = self.weight
            if (self.bias is not None) and (not self.bias.stop_gradient) and (self.bias.grad is not None):
                lr_b = opt['meta_ratio'] * opt['main_lr'] * self.bias.optimize_attr.get('learning_rate', 1.0)
                updated_bias = self.bias - lr_b * self.bias.grad
            else:
                updated_bias = self.bias
            return F.conv2d(inputs, updated_weight, updated_bias, self._stride, self._padding, self._dilation, self._groups)
        else:
            return F.conv2d(inputs, self.weight, self.bias, self._stride, self._padding, self._dilation, self._groups)


class meta_bin(nn.Layer):
    def __init__(self, num_features, norm_opt=None, compute_meta_gates=True, **kwargs):
        super().__init__()
        self.bat_n = meta_bn(num_features, norm_opt, **kwargs)
        self.ins_n = meta_in(num_features, norm_opt, **kwargs)
        if norm_opt['BIN_INIT'] == 'one':
            gate_init = nn.initializer.Constant(1)
        elif norm_opt['BIN_INIT'] == 'zero':
            gate_init = nn.initializer.Constant(0)
        elif norm_opt['BIN_INIT'] == 'half':
            gate_init = nn.initializer.Constant(0.5)
        elif norm_opt['BIN_INIT'] == 'random':
            gate_init = nn.initializer.Uniform(0, 1)
        self.gate = paddle.create_parameter([num_features], dtype='float32', default_initializer=gate_init)
        setattr(self.gate, 'bin_gate', True)
        self.compute_meta_gates = compute_meta_gates

    def forward(self, inputs, opt=None):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))
        param_update = (opt != None) and opt.get('param_update', False) and self.compute_meta_gates

        if param_update and (self.gate is not None) and (self.gate.grad is not None) and (not self.gate.stop_gradient):
            lr_g = opt['meta_ratio'] * opt['norm_lr'] * self.gate.optimize_attr.get('learning_rate', 1.0)
            update_gate = self.gate - lr_g * self.gate.grad
            if opt['inner_clamp']:
                update_gate.clip_(min=0, max=1)
        else:
            update_gate = self.gate
        out_bn = self.bat_n(inputs, opt)
        out_in = self.ins_n(inputs, opt)
        update_gate = update_gate.unsqueeze([0, -1, -1]).astype(out_bn.dtype)
        out = out_bn * update_gate + out_in * (1 - update_gate)
        return out


class meta_bn(nn.BatchNorm2D):
    def __init__(self, num_features, norm_opt, momentum=0.9, epsilon=1e-05,
                weight_freeze = False, bias_freeze = False, compute_meta_params=False,
                weight_attr=None, bias_attr=None, data_format='NCHW', use_global_stats=None, name=None):
        if not weight_freeze:
            weight_freeze = norm_opt['BN_W_FREEZE']
        if not bias_freeze:
            bias_freeze = norm_opt['BN_B_FREEZE']
        use_global_stats = norm_opt['BN_RUNNING']
        self.affine = True
        super().__init__(num_features, momentum, epsilon, weight_attr, bias_attr, data_format, use_global_stats, name)
        self.weight.stop_gradient = weight_freeze
        self.bias.stop_gradient = bias_freeze
        self.compute_meta_params = compute_meta_params

    def forward(self, inputs, opt:dict=None):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))
        param_update = (opt != None) and opt.get('param_update', False) and self.compute_meta_params

        if self.training:
            norm_type = opt['type_running_stats']
        else:
            norm_type = "eval"

        if param_update and self.affine:
            if (self.weight is not None) and (not self.weight.stop_gradient) and (self.weight.grad is not None):
                lr_w = opt['meta_ratio'] * opt['main_lr'] * self.weight.optimize_attr.get('learning_rate', 1.0)
                updated_weight = self.weight - lr_w * self.weight.grad
            else:
                updated_weight = self.weight
            if (self.bias is not None) and (not self.bias.stop_gradient) and (self.bias.grad is not None):
                lr_b = opt['meta_ratio'] * opt['main_lr'] * self.bias.optimize_attr.get('learning_rate', 1.0)
                updated_bias = self.bias - lr_b * self.bias.grad
            else:
                updated_bias = self.bias
        else:
            updated_weight = self.weight
            updated_bias = self.bias

        if norm_type == "general": # update, but not apply running_mean/var
            result = F.batch_norm(inputs, self._mean, self._variance,
                                    updated_weight, updated_bias,
                                    self.training, self._momentum, self._epsilon)
        elif norm_type == "hold": # not update, not apply running_mean/var
                #result = F.batch_norm(inputs, None, None,
            result = F.batch_norm(inputs, paddle.mean(inputs, axis=(0, 2, 3)), paddle.var(inputs, axis=(0, 2, 3)),
                                    updated_weight, updated_bias,
                                    self.training, self._momentum, self._epsilon)
        elif norm_type == "eval": # fix and apply running_mean/var,
            if self._mean is None:
                    #result = F.batch_norm(inputs, None, None,
                result = F.batch_norm(inputs, paddle.mean(inputs, axis=(0, 2, 3)), paddle.var(inputs, axis=(0, 2, 3)),
                                        updated_weight, updated_bias,
                                        True, self._momentum, self._epsilon)
            else:
                result = F.batch_norm(inputs, self._mean, self._variance,
                                        updated_weight, updated_bias,
                                        False, self._momentum, self._epsilon)
        return result

class meta_in(nn.InstanceNorm2D):
    def __init__(self, num_features, norm_opt, epsilon=1e-05, momentum=0.9, 
                weight_freeze = False, bias_freeze = False, compute_meta_params=False,
                weight_attr=None, bias_attr=None, data_format="NCHW", name=None):
        if not weight_freeze:
            weight_freeze = norm_opt['IN_W_FREEZE']
        if not bias_freeze:
            bias_freeze = norm_opt['IN_B_FREEZE']
        self.affine = True
        use_global_stats = norm_opt['IN_RUNNING']
        super().__init__(num_features, epsilon, momentum, weight_attr, bias_attr, data_format, name)
        self.in_fc_multiply = norm_opt['IN_FC_MULTIPLY']
        self._momentum = momentum
        self.compute_meta_params = compute_meta_params

    def forward(self, inputs, opt = None):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))

        if (inputs.shape[2] == 1) and (inputs.shape[2] == 1): # fc layers
            inputs[:] *= self.in_fc_multiply
            return inputs
        else:
            param_update = (opt != None) and opt.get('param_update', False) and self.compute_meta_params and (self.scale is not None)

            if self.training:
                norm_type = opt['type_running_stats']
            else:
                norm_type = "eval"

            if param_update and self.affine:
                # if opt['zero_grad']: self.zero_grad()
                if (self.scale is not None) and (not self.scale.stop_gradient) and (self.scale.grad is not None):
                    lr_w = opt['meta_ratio'] * opt['main_lr'] * self.scale.optimize_attr.get('learning_rate', 1.0)
                    updated_scale = self.scale - lr_w * self.scale.grad
                else:
                    updated_scale = self.scale
                if (self.bias is not None) and (not self.bias.stop_gradient) and (self.bias.grad is not None):
                    lr_b = opt['meta_ratio'] * opt['main_lr'] * self.bias.optimize_attr.get('learning_rate', 1.0)
                    updated_bias = self.bias - lr_b * self.bias.grad
                else:
                    updated_bias = self.bias
                # print('meta_bn is computed')
            else:
                updated_scale = self.scale
                updated_bias = self.bias


            if norm_type == "general":
                return F.instance_norm(inputs, None, None,
                                       updated_scale, updated_bias,
                                       self.training, self._momentum, self._epsilon)
            elif norm_type == "hold":
                return F.instance_norm(inputs, None, None,
                                       updated_scale, updated_bias,
                                       self.training, self._momentum, self._epsilon)
            elif norm_type == "eval":
                if self._mean is None:
                    return F.instance_norm(inputs, None, None,
                                           updated_scale, updated_bias,
                                           True, self._momentum, self._epsilon)
                else:
                    return F.instance_norm(inputs, None, None,
                                           updated_scale, updated_bias,
                                           False, self._momentum, self._epsilon)    
    