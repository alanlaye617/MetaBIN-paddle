import paddle 
import torch
import copy

def translate_inputs_p2t(inputs_pad, if_train=True):
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

def translate_inputs_t2p(inputs_ref, if_train=True):
    inputs_pad = {}
    inputs_pad['images'] = paddle.to_tensor(inputs_ref['images'].numpy(), dtype=paddle.float32)
    inputs_pad['targets'] = paddle.to_tensor(inputs_ref['targets'].numpy(), dtype=paddle.int64)
    inputs_pad['camid'] = paddle.to_tensor(inputs_ref['camid'].numpy(), dtype=paddle.int64)
    inputs_pad['img_path'] = copy.deepcopy(inputs_ref['img_path'])
    if if_train:
        inputs_pad['domains'] = {'domains': paddle.to_tensor(inputs_ref['others']['domains'].numpy(), dtype=paddle.int64)}
    else:
        inputs_pad['domains'] = {'domains': inputs_ref['others']['domains']}
    return inputs_pad