import paddle
from paddle import optimizer
from paddle.optimizer import Optimizer


def build_optimizer(model, lr_scheduler, solver_opt, momentum, flag = None):
    params = []
    for key, value in model.named_parameters():
        weight_decay = 0.0005

        if "bias" in key:
            weight_decay = 0.0005  #cfg.SOLVER.WEIGHT_DECAY_BIAS
        if (flag == 'main') and ("gate" not in key):
            params += [{
                "name": key,
                "parameters": [value], 
                "lr": lr, 
                "weight_decay": weight_decay, 
                "freeze": False
                }]

        elif (flag == 'norm') and ("gate" in key):
            params += [{
                "name": key, 
                "parameters": [value], 
                "lr": lr, 
                "weight_decay": 0.0,
                "freeze": False
                }]

        # params += [{"name": key, "params": [value], "weight_decay": weight_decay, "freeze": False}]
    name2optimier = {
        'Momentum': optimizer.Momentum(),
    }
    
    if solver_opt in name2optimier:
        if solver_opt == "Momentum":
            opt_fns = name2optimier[solver_opt](learning_rate=lr_scheduler, parameters=params, momentum=momentum)
        else:
            opt_fns = name2optimier[solver_opt](params)
    else:
        raise NameError("optimizer {} not support".format(solver_opt))
    return opt_fns
