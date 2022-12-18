import paddle 
import torch
import copy

def translate_inputs(inputs_pad, if_train=True):
    inputs_ref = {}
    inputs_ref['images'] = torch.tensor(inputs_pad['images'].numpy(), dtype=torch.float32)
    inputs_ref['targets'] = torch.tensor(inputs_pad['targets'].numpy(), dtype=torch.int64)
    inputs_ref['camid'] = torch.tensor(inputs_pad['camid'].numpy(), dtype=torch.int64)
    inputs_ref['img_path'] = copy.deepcopy(inputs_pad['img_path'])
    if if_train:
        inputs_ref['others'] = {'domains': torch.tensor(inputs_pad['domains'].numpy(), dtype=torch.int64)}
    else:
        inputs_ref['others'] = {'domains': inputs_pad['domains']}
    return inputs_ref