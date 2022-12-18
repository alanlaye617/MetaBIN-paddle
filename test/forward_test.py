import sys
import os
import paddle
from paddle import nn
import paddle.nn.functional as F
import torch
import numpy as np
from reprod_log import ReprodLogger, ReprodDiffHelper
import random
sys.path.append('.')
from utils import translate_weight, build_ref_model
from modeling import build_resnet_backbone

def forward_test():
    paddle.device.set_device('gpu:0')

    seed = 2022
    np.random.seed = seed
    random.seed = seed

    reprod_log_ref = ReprodLogger()
    reprod_log_pad = ReprodLogger()

    torch_path = "./model_weights/backbone.pth"
    paddle_path = "./model_weights/backbone.pdparams"

    model_ref = build_ref_model(num_classes=1).backbone.cuda()
    torch.save(model_ref.state_dict(), torch_path)
    model_ref.eval()

    translate_weight(torch_path, paddle_path)

    model_pad = build_resnet_backbone()
    model_pad.set_state_dict(paddle.load(paddle_path))
    model_pad.eval()

    for i in range(5):
        inputs = np.random.rand(16, 3, 256, 128)

        inputs_ref = torch.tensor(inputs, dtype=torch.float32).cuda()
        outputs_ref = model_ref(inputs_ref)
        reprod_log_ref.add("forwards_logits_%d"%(i), outputs_ref.cpu().detach().numpy())
        del outputs_ref, inputs_ref

        inputs_pad = paddle.to_tensor(inputs, dtype=paddle.float32)
        outputs_pad = model_pad(inputs_pad)
        reprod_log_pad.add("forwards_logits_%d"%(i), outputs_pad.detach().numpy())
        del outputs_pad, inputs_pad

    reprod_log_ref.save('./result/forward_ref.npy')
    reprod_log_pad.save('./result/forward_paddle.npy')

    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info("./result/forward_paddle.npy")
    info2 = diff_helper.load_info("./result/forward_ref.npy")

    diff_helper.compare_info(info1, info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./result/log/forward_diff.log")

if __name__ == '__main__':
    forward_test()