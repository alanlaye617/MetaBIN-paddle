import paddle
from paddle import optimizer
from paddle.optimizer import Optimizer


def build_optimizer(model, lr_scheduler, momentum, flag = None):
    params = []
    assert flag in ['main', 'norm'], NameError('Unknown flag'+str(flag))
    for key, value in model.named_parameters():
        if isinstance(value, list):
            print('.')
        if value.stop_gradient:
            continue
        weight_decay = 0.0005
        lr = 1
        if "backbone" in key:
            lr *= 1.0       # cfg.SOLVER.BACKBONE_LR_FACTOR
        if "heads" in key:
            lr *= 1.0       # cfg.SOLVER.HEADS_LR_FACTOR
        if "bias" in key:
            lr *= 2.0       # cfg.SOLVER.BIAS_LR_FACTOR
        if "gate" in key:
            lr *= 20.0      # cfg.META.SOLVER.LR_FACTOR.GATE

        if flag == 'main':
            if "gate" not in key:
                params += [{
                    "learning_rate": lr,
                    "name": key,
                    "params": [value], 
                    "weight_decay": weight_decay, 
                    "freeze": False
                    }]
            """
            else:
                params += [{
                    "learning_rate": 0,
                    "name": key,
                    "params": [value], 
                    "weight_decay": weight_decay, 
                    "freeze": False
                    }]
            """
        elif flag == 'norm':
            if "gate" in key:
                params += [{
                    "learning_rate": lr,
                    "name": key, 
                    "params": [value], 
                    "weight_decay": 0.0,
                    "freeze": False
                    }]
            """
            else:
                params += [{
                    "learning_rate": 0,
                    "name": key, 
                    "params": [value], 
                    "weight_decay": 0.0,
                    "freeze": False
                    }]                
            """
    name2optimier = {
        'Momentum': optimizer.Momentum,
    }
    
    opt_fns = optimizer.Momentum(learning_rate=lr_scheduler, parameters=params, momentum=momentum)
    return opt_fns
