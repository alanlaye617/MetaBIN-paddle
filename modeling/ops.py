import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.autograd as autograd
import math
import copy

def update_parameter(param, step_size, opt = None):
    """
    TODO debug
    """
    loss = opt['meta_loss']
    use_second_order = opt['use_second_order']
    allow_unused = opt['allow_unused']
    stop_gradient = opt['stop_gradient']
    flag_update = False
    if step_size is not None:
        if not stop_gradient:
            if param is not None:
                if opt['auto_grad_outside']:
                    if opt['grad_params'][0] == None:
                        del opt['grad_params'][0]
                        updated_param = param
                    else:
                        # print("[GRAD]{} [PARAM]{}".format(opt['grad_params'][0].data.shape, param.data.shape))
                        # outer
                        updated_param = param - step_size * opt['grad_params'][0]
                        del opt['grad_params'][0]
                else:
                    # inner
                    grad = autograd.grad(loss, param, create_graph=use_second_order, allow_unused=allow_unused)[0]
                    updated_param = param - step_size * grad
                # outer update
                # updated_param = opt['grad_params'][0]
                # del opt['grad_params'][0]
                flag_update = True
        else:
            if param is not None:

                if opt['auto_grad_outside']:
                    if opt['grad_params'][0] == None:
                        del opt['grad_params'][0]
                        updated_param = param
                    else:
                        # print("[GRAD]{} [PARAM]{}".format(opt['grad_params'][0].data.shape, param.data.shape))
                        # outer
                        updated_param = param - step_size * opt['grad_params'][0]
                        del opt['grad_params'][0]
                else:
                    # inner
                    # grad = Variable(autograd.grad(loss, param, create_graph=use_second_order, allow_unused=allow_unused)[0].data, requires_grad=False)
                    grad = paddle.to_tensor(autograd.grad(loss, param, create_graph=use_second_order, allow_unused=allow_unused)[0].data, stop_gradient=True)
                    updated_param = param - step_size * grad
                # outer update
                # updated_param = opt['grad_params'][0]
                # del opt['grad_params'][0]
                flag_update = True
    if not flag_update:
        return param
    return updated_param


class meta_linear(nn.Linear):
    def __init__(self, in_features, out_features, weight_attr=None, bias_attr=None, name=None):
        super().__init__(in_features, out_features, weight_attr, bias_attr, name)

    def forward(self, input, opt = None):
        if opt != None:
            use_meta_learning = False
            if opt['param_update']:
                if self.weight is not None:
                    if self.compute_meta_params:
                        use_meta_learning = True
        else:
            use_meta_learning = False
        if use_meta_learning:
            updated_weight = update_parameter(self.weight, self.w_step_size, opt)
            updated_bias = update_parameter(self.bias, self.b_step_size, opt)
            return F.linear(input, updated_weight, updated_bias)
        else:
            return F.linear(input, self.weight, self.bias)


class meta_conv2d(nn.Conv2D):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', weight_attr=None, bias_attr=None, data_format="NCHW"):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, padding_mode, weight_attr, bias_attr, data_format)
    
    def forward(self, inputs, opt = None):
        if opt != None:
            use_meta_learning = False
            if opt['param_update']:
                if self.weight is not None:
                    if self.compute_meta_params:
                        use_meta_learning = True
        else:
            use_meta_learning = False
        if use_meta_learning:
            updated_weight = update_parameter(self.weight, self.w_step_size, opt)
            updated_bias = update_parameter(self.bias, self.b_step_size, opt)
            # print('meta_conv is computed')
            return F.conv2d(inputs, updated_weight, updated_bias, self._stride, self._padding, self._dilation, self._groups)
        else:
            return F.conv2d(inputs, self.weight, self.bias, self._stride, self._padding, self._dilation, self._groups)


def meta_norm(norm, out_channels, norm_opt, **kwargs):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": Meta_bn_norm(out_channels, norm_opt, **kwargs),
            "IN": Meta_in_norm(out_channels, norm_opt, **kwargs),
            "BIN_gate2": Meta_bin_gate_ver2(out_channels, norm_opt, **kwargs),
        }[norm]
    return norm


class Meta_bin_gate_ver2(nn.Layer):
    def __init__(self, num_features, norm_opt = None, **kwargs):
        super().__init__()
        self.bat_n = Meta_bn_norm(num_features, norm_opt, **kwargs)
        self.ins_n = Meta_in_norm(num_features, norm_opt, **kwargs)
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

    def forward(self, inputs, opt=None):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))
        if opt != None:
            use_meta_learning_gates = False
            if opt['param_update']:
                if self.compute_meta_gates:
                    use_meta_learning_gates = True
        else:
            use_meta_learning_gates = False

        if use_meta_learning_gates:
            update_gate = update_parameter(self.gate, self.g_step_size, opt)
            if opt['inner_clamp']:
                # update_gate.data.clamp_(min=0, max=1)
                update_gate = update_gate.clip(min=0, max=1)
            # print(update_gate[0].data.cpu())
        else:
            update_gate = self.gate
        update_gate = update_gate.unsqueeze([0, -1, -1])
        out_bn = self.bat_n(inputs)#, opt)
        out_in = self.ins_n(inputs)#, opt)
        out = out_bn * update_gate + out_in * (1-update_gate)
        return out


class Meta_bn_norm(nn.BatchNorm2D):
    def __init__(self, num_features, norm_opt, momentum=0.9, epsilon=0.00001,
                weight_freeze = False, bias_freeze = False, weight_init = 1.0, bias_init = 0.0,
                data_format='NCHW', use_global_stats=None, name=None):
        if not weight_freeze:
            weight_freeze = norm_opt['BN_W_FREEZE']
        if not bias_freeze:
            bias_freeze = norm_opt['BN_B_FREEZE']
        lr = (0.0 if norm_opt['BN_AFFINE'] else 1.0)
        use_global_stats = norm_opt['BN_RUNNING']
        weight_attr = paddle.ParamAttr(
            initializer=(nn.initializer.Constant(weight_init) if weight_init is not None else None), 
            learning_rate=lr)
        bias_attr = paddle.ParamAttr(
            initializer=(nn.initializer.Constant(bias_init) if bias_init is not None else None), 
            learning_rate=lr)
        super().__init__(num_features, momentum, epsilon, weight_attr, bias_attr, data_format, use_global_stats, name)
        self.weight.stop_gradient = weight_freeze
        self.bias.stop_gradient = bias_freeze


class Meta_in_norm(nn.InstanceNorm2D):
    def __init__(self, num_features, norm_opt, epsilon=0.00001, momentum=0.9, 
                weight_freeze = False, bias_freeze = False, weight_init = 1.0, bias_init = 0.0,
                data_format="NCHW", name=None):
        if not weight_freeze:
            weight_freeze = norm_opt['IN_W_FREEZE']
        if not bias_freeze:
            bias_freeze = norm_opt['IN_B_FREEZE']
        lr = (0.0 if norm_opt['IN_AFFINE'] else 1.0)
        use_global_stats = norm_opt['IN_RUNNING']
        weight_attr = paddle.ParamAttr(
            initializer=(nn.initializer.Constant(weight_init) if weight_init is not None else None), 
            learning_rate=lr)
        bias_attr = paddle.ParamAttr(
            initializer=(nn.initializer.Constant(bias_init) if bias_init is not None else None), 
            learning_rate=lr)
        super().__init__(num_features, epsilon, momentum, weight_attr, bias_attr, data_format, name)
        self.in_fc_multiply = norm_opt['IN_FC_MULTIPLY']
    

"""
TODO
class Meta_bn_norm(nn.BatchNorm2d):
    def __init__(self, num_features, norm_opt = None, eps=1e-05,
                 momentum=0.1, weight_freeze = False, 
                 weight_init = 1.0, bias_init = 0.0):

        if not weight_freeze:
            weight_freeze = norm_opt['BN_W_FREEZE']
        if not bias_freeze:
            bias_freeze = norm_opt['BN_B_FREEZE']

        affine = True if norm_opt['BN_AFFINE'] else False
        track_running_stats = True if norm_opt['BN_RUNNING'] else False
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

        if weight_init is not None: self.weight.data.fill_(weight_init)
        if bias_init is not None: self.bias.data.fill_(bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


    def forward(self, inputs, opt = None):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))
        if opt != None:
            use_meta_learning = False
            if opt['param_update']:
                if self.weight is not None:
                    if self.compute_meta_params:
                        use_meta_learning = True
        else:
            use_meta_learning = False

        if self.training:
            norm_type = opt['type_running_stats']
        else:
            norm_type = "eval"

        if use_meta_learning and self.affine:
            # if opt['zero_grad']: self.zero_grad()
            updated_weight = update_parameter(self.weight, self.w_step_size, opt)
            updated_bias = update_parameter(self.bias, self.b_step_size, opt)
            # print('meta_bn is computed')
        else:
            updated_weight = self.weight
            updated_bias = self.bias

        if opt == None:
            compute_each_batch = False
        else:
            try:
                if opt['each_domain']:
                    compute_each_batch = True
                else:
                    compute_each_batch = False
            except: # if opt['each_domain'] does not exist
                compute_each_batch = False
        if norm_type == "eval":
            compute_each_batch = False

        if compute_each_batch:
            domain_idx = opt['domains']
            unique_domain_idx = [int(x) for x in torch.unique(domain_idx).cpu()]
            cnt = 0
            for j in unique_domain_idx:
                t_logical_domain = domain_idx == j

                if norm_type == "general":  # update, but not apply running_mean/var
                    result_local = F.batch_norm(inputs[t_logical_domain], self.running_mean, self.running_var,
                                          updated_weight, updated_bias,
                                          self.training, self.momentum, self.eps)
                elif norm_type == "hold":  # not update, not apply running_mean/var
                    result_local = F.batch_norm(inputs[t_logical_domain], None, None,
                                          updated_weight, updated_bias,
                                          self.training, self.momentum, self.eps)
                elif norm_type == "eval":  # fix and apply running_mean/var,
                    if self.running_mean is None:
                        result_local = F.batch_norm(inputs[t_logical_domain], None, None,
                                              updated_weight, updated_bias,
                                              True, self.momentum, self.eps)
                    else:
                        result_local = F.batch_norm(inputs[t_logical_domain], self.running_mean, self.running_var,
                                              updated_weight, updated_bias,
                                              False, self.momentum, self.eps)

                if cnt == 0:
                    result = copy.copy(result_local)
                else:
                    result = paddle.cat((result, result_local), 0)
                cnt += 1

        else:
            if norm_type == "general": # update, but not apply running_mean/var
                result = F.batch_norm(inputs, self.running_mean, self.running_var,
                                      updated_weight, updated_bias,
                                      self.training, self.momentum, self.eps)
            elif norm_type == "hold": # not update, not apply running_mean/var
                result = F.batch_norm(inputs, None, None,
                                      updated_weight, updated_bias,
                                      self.training, self.momentum, self.eps)
            elif norm_type == "eval": # fix and apply running_mean/var,
                if self.running_mean is None:
                    result = F.batch_norm(inputs, None, None,
                                          updated_weight, updated_bias,
                                          True, self.momentum, self.eps)
                else:
                    result = F.batch_norm(inputs, self.running_mean, self.running_var,
                                          updated_weight, updated_bias,
                                          False, self.momentum, self.eps)
        return result


class Meta_in_norm(nn.InstanceNorm2D):
    def __init__(self, num_features, epsilon=0.00001, momentum=0.9, weight_attr=None, bias_attr=None, data_format="NCHW", name=None):
        super().__init__(num_features, epsilon, momentum, weight_attr, bias_attr, data_format, name)
    def __init__(self, num_features, norm_opt = None, eps=1e-05,
                 momentum=0.1, weight_freeze = False, bias_freeze = False,
                 weight_init = 1.0, bias_init = 0.0):

        if not weight_freeze:
            weight_freeze = norm_opt['IN_W_FREEZE']
        if not bias_freeze:
            bias_freeze = norm_opt['IN_B_FREEZE']

        affine = True if norm_opt['IN_AFFINE'] else False
        track_running_stats = True if norm_opt['IN_RUNNING'] else False
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

        if self.weight is not None:
            if weight_init is not None: self.weight.data.fill_(weight_init)
            self.weight.requires_grad_(not weight_freeze)
        if self.bias is not None:
            if bias_init is not None: self.bias.data.fill_(bias_init)
            self.bias.requires_grad_(not bias_freeze)
        self.in_fc_multiply = norm_opt['IN_FC_MULTIPLY']

    def forward(self, inputs, opt = None):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))

        if (inputs.shape[2] == 1) and (inputs.shape[2] == 1): # fc layers
            inputs[:] *= self.in_fc_multiply
            return inputs
        else:
            if opt != None:
                use_meta_learning = False
                if opt['param_update']:
                    if self.weight is not None:
                        if self.compute_meta_params:
                            use_meta_learning = True
            else:
                use_meta_learning = False

            if self.training:
                norm_type = opt['type_running_stats']
            else:
                norm_type = "eval"

            if use_meta_learning and self.affine:
                # if opt['zero_grad']: self.zero_grad()
                updated_weight = update_parameter(self.weight, self.w_step_size, opt)
                updated_bias = update_parameter(self.bias, self.b_step_size, opt)
                # print('meta_bn is computed')
            else:
                updated_weight = self.weight
                updated_bias = self.bias


            if norm_type == "general":
                return F.instance_norm(inputs, self.running_mean, self.running_var,
                                       updated_weight, updated_bias,
                                       self.training, self.momentum, self.eps)
            elif norm_type == "hold":
                return F.instance_norm(inputs, None, None,
                                       updated_weight, updated_bias,
                                       self.training, self.momentum, self.eps)
            elif norm_type == "eval":
                if self.running_mean is None:
                    return F.instance_norm(inputs, None, None,
                                           updated_weight, updated_bias,
                                           True, self.momentum, self.eps)
                else:
                    return F.instance_norm(inputs, self.running_mean, self.running_var,
                                           updated_weight, updated_bias,
                                           False, self.momentum, self.eps)

"""