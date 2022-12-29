import yaml
import os
from paddle.optimizer.lr import LRScheduler

def get_cfg(mode, root='./configs'):
        path = os.path.join(root, mode+".yaml")
        with open(path, encoding="UTF-8") as cfg_file:
            cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
        return cfg


def get_curr_lr(optimizer, param_idx):
    if isinstance(optimizer._learning_rate, LRScheduler):
        lr = optimizer._learning_rate.last_lr * optimizer._param_groups[param_idx]['learning_rate']
    else:
        lr = optimizer._learning_rate * optimizer._param_groups[param_idx]['learning_rate']
    return lr