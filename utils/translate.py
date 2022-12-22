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
        inputs_pad['domains'] = paddle.to_tensor(inputs_ref['others']['domains'].numpy(), dtype=paddle.int64)
    else:
        inputs_pad['domains'] = inputs_ref['others']['domains']
    return inputs_pad


def translate_weight(torch_path, paddle_path):
    torch_state_dict = torch.load(torch_path)
    fc_names = ["classifier_fc"]
    in_names = 'ins_n'
    paddle_state_dict = {}
    for k in torch_state_dict:
        if "num_batches_tracked" in k:
            continue
        v = torch_state_dict[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k: # ignore bias
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}")
            v = v.transpose(new_shape)
        if 'ins_n.weight' in k:
            k = k.replace("weight", "scale")
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        # if k not in model_state_dict:
        if False:
            print(k)
        else:
            paddle_state_dict[k] = v
    paddle.save(paddle_state_dict, paddle_path)


def translate_params_name_t2p(name_torch):
    if 'ins_n.weight' in name_torch:
        name_torch = name_torch.replace("weight", "scale")
    name_torch = name_torch.replace("running_var", "_variance")
    name_torch = name_torch.replace("running_mean", "_mean")        
    name_paddle = name_torch
    return name_paddle

