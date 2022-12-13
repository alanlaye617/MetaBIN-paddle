import numpy as np
import torch
import paddle

def torch2paddle(torch_path, paddle_path):
    torch_state_dict = torch.load(torch_path)
    fc_names = ["classifier"]
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

if __name__ == '__main__':
    torch_path = "./data/backbone.pth"
    paddle_path = "./data/backbone.pdparams"
    torch2paddle(torch_path, paddle_path)