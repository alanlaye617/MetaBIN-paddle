import paddle
from paddle import optimizer
from paddle.optimizer import Optimizer


def build_optimizer(model, learning_rate, lr_scheduler, momentum, flag = None):
    params = []
    assert flag in ['main', 'norm'], NameError('Unknown flag'+str(flag))
    for key, value in model.named_parameters():
        if isinstance(value, list):
            print('.')
        if value.stop_gradient:
            continue
        lr = learning_rate
        weight_decay = 0.0005

        if "bias" in key:
            lr *= 2.0       # cfg.SOLVER.BIAS_LR_FACTOR
        if "gate" in key:
            lr *= 20.0      # cfg.META.SOLVER.LR_FACTOR.GATE

        if flag == 'main' and "gate" not in key:
                params += [{
                    "learning_rate": lr,
                    "name": key,
                    "params": [value], 
                    "weight_decay": weight_decay, 
                    }]
            
        elif flag == 'norm' and "gate" in key:
                params += [{
                    "learning_rate": lr,
                    "name": key, 
                    "params": [value], 
                    "weight_decay": 0.0,
                    }] 

    opt_fns = optimizer.Momentum(
        learning_rate=lr_scheduler,
        parameters=params, 
        momentum=momentum)
    return opt_fns