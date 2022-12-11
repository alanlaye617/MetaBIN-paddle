import sys
import paddle
from paddle import nn
import paddle.nn.functional as F
import torch
import numpy as np
from modeling import build_resnet_backbone
from utils.weight_translate import torch2paddle
from utils.build_ref_model import build_ref_model
from reprod_log import ReprodLogger, ReprodDiffHelper
import random

paddle.device.set_device('gpu:0')

seed = 2022
np.random.seed = seed
random.seed = seed

reprod_log_ref = ReprodLogger()
reprod_log_pad = ReprodLogger()

torch_path = "./data/backbone.pth"
paddle_path = "./data/backbone.pdparams"

ref_model = build_ref_model().backbone.cuda()
torch.save(ref_model.state_dict(), torch_path)
ref_model.eval()

torch2paddle(torch_path, paddle_path)

pad_model = build_resnet_backbone()
pad_model.set_state_dict(paddle.load(paddle_path))
pad_model.eval()

for i in range(5):
    inputs = np.random.rand(16, 3, 256, 128)

    ref_inputs = torch.tensor(inputs, dtype=torch.float32).cuda()
    ref_outputs = ref_model(ref_inputs)
    reprod_log_ref.add("forwards_test_%d"%(i), ref_outputs.cpu().detach().numpy())
    del ref_outputs, ref_inputs

    pad_inputs = paddle.to_tensor(inputs, dtype=paddle.float32)
    pad_outputs = pad_model(pad_inputs)
    reprod_log_pad.add("forwards_test_%d"%(i), pad_outputs.detach().numpy())
    del pad_outputs, pad_inputs

reprod_log_ref.save('result/forward_ref.npy')
reprod_log_pad.save('result/forward_paddle.npy')

diff_helper = ReprodDiffHelper()

info1 = diff_helper.load_info("./result/forward_paddle.npy")
info2 = diff_helper.load_info("./result/forward_ref.npy")

diff_helper.compare_info(info1, info2)

diff_helper.report(
    diff_method="mean", diff_threshold=1e-6, path="./result/log/forward_diff.log")