def build_lr_scheduler(optimizer,
                       scheduler_method,
                       warmup_factor,
                       warmup_iters,
                       warmup_method,
                       milestones,
                       gamma,
                       max_iters,
                       delay_iters,
                       eta_min_lr):
    scheduler_args = {
        "optimizer": optimizer,

        # warmup options
        "warmup_factor": warmup_factor,
        "warmup_iters": warmup_iters,
        "warmup_method": warmup_method,

        # multi-step lr scheduler options
        "milestones": milestones,
        "gamma": gamma,

        # cosine annealing lr scheduler options
        "max_iters": max_iters,
        "delay_iters": delay_iters,
        "eta_min_lr": eta_min_lr,

    }
    return getattr(lr_scheduler, scheduler_method)(**scheduler_args)
