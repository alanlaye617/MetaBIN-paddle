import paddle 
import torch
import copy

def translate_inputs(inputs_pad):
    inputs_ref = {}
    inputs_ref['images'] = torch.tensor(inputs_pad['images'].numpy(), dtype=torch.float32)
    inputs_ref['targets'] = torch.tensor(inputs_pad['targets'].numpy(), dtype=torch.int64)
    inputs_ref['camid'] = torch.tensor(inputs_pad['camid'].numpy(), dtype=torch.int64)
    inputs_ref['img_path'] = copy.deepcopy(inputs_pad['img_path'])
    inputs_ref['others'] = {'domains': torch.tensor(inputs_pad['domains'].numpy(), dtype=torch.int64)}
    return inputs_ref