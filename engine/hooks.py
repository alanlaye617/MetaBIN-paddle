from collections import Counter

class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    .. code-block:: python
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        hook.after_train()
    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).
        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.
           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.
    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class Logger(HookBase):
    def __init__(self) -> None:
        super().__init__()

class PeriodicEval(HookBase):
    def __init__(self, period, dataset, model, batch_size) -> None:
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.period = period
        self.batch_size = batch_size
        self.count = 0

    def after_step(self):
        self.count += 1
        if self.count and self.count % self.period == 0:
            self.trainer.test(dataset_name=self.dataset, model=self.model, batch_size=self.batch_size)

class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer, scheduler, optimizer2 = None, scheduler2 = None):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim._LRScheduler)
        """
        self._optimizer = optimizer
        self._scheduler = scheduler
        if scheduler2 is not None:
            self._scheduler2 = scheduler2
        else:
            self._scheduler2 = None

        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        '''
        largest_group = max(len(g["params"]) for g in optimizer._param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer._param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer._param_groups):
                if g["lr"] == lr:
                    self._best_param_group_id = i
                    break
        else:
            for i, g in enumerate(optimizer._param_groups):
                if len(g["params"]) == largest_group:
                    self._best_param_group_id = i
                    break
        '''
    def after_step(self):
        #!lr = self._optimizer._param_groups[self._best_param_group_id]["lr"]
        #!self.trainer.storage.put_scalar("main_lr", lr, smoothing_hint=False)
        self._scheduler.step()
        if self._scheduler2 is not None:
            self._scheduler2.step()