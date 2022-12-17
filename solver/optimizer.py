def build_optimizer(cfg, model, solver_opt, momentum, flag = None):
    params = []
    for key, value in model.named_parameters():
        # print(key)
        if isinstance(value, list):
            print('.')
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "backbone" in key:
            lr *= cfg.SOLVER.BACKBONE_LR_FACTOR
        if "heads" in key:
            lr *= cfg.SOLVER.HEADS_LR_FACTOR
        if "bias" in key:
            lr *= cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if "gate" in key:
            print(key, value.shape)
            lr *= cfg.META.SOLVER.LR_FACTOR.GATE

        if (flag == 'main') and ("gate" not in key):
            params += [{"name": key, "params": [value], "lr": lr, "weight_decay": weight_decay, "freeze": False}]
        elif (flag == 'norm') and ("gate" in key):
            params += [{"name": key, "params": [value], "lr": lr, "weight_decay": cfg.SOLVER.WEIGHT_DECAY_NORM, "freeze": False}]

        # params += [{"name": key, "params": [value], "lr": lr, "weight_decay": weight_decay, "freeze": False}]

    if hasattr(optim, solver_opt):
        if solver_opt == "SGD":
            opt_fns = getattr(optim, solver_opt)(params, momentum=momentum)
        else:
            opt_fns = getattr(optim, solver_opt)(params)
    else:
        raise NameError("optimizer {} not support".format(solver_opt))
    return opt_fns
