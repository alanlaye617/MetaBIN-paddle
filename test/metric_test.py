import paddle
from paddle import nn
import paddle.nn.functional as F
import torch
import numpy as np
from reprod_log import ReprodLogger, ReprodDiffHelper
import random

from utils.weight_translate import weight_translate
from utils.build_ref_model import build_ref_model
from modeling import build_resnet_backbone, Metalearning
from data import build_reid_test_loader


def metric_test():
    paddle.device.set_device('gpu:0')

    seed = 2022
    np.random.seed = seed
    random.seed = seed

    reprod_log_ref = ReprodLogger()
    reprod_log_pad = ReprodLogger()

    torch_path = "./data/backbone.pth"
    paddle_path = "./data/backbone.pdparams"

    model_ref = build_ref_model(4).cuda()
#    model_ref = build_ref_model().backbone.cuda()
    torch.save(model_ref.state_dict(), torch_path)
    model_ref.eval()

    weight_translate(torch_path, paddle_path)

#    model_pad = build_resnet_backbone()
    model_pad = Metalearning(num_classes=4)
    model_pad.set_state_dict(paddle.load(paddle_path))
    model_pad.eval()

    test_loader_lite, num_query_lite= build_reid_test_loader('LiteData', 20, num_workers=0)
    for x in test_loader_lite:
        break
    inputs = x['images'].numpy()
    inputs_ref = torch.tensor(inputs, dtype=torch.float32).cuda()
    outputs_ref = model_ref(inputs_ref)
    reprod_log_ref.add("metric", outputs_ref.cpu().detach().numpy())
    del outputs_ref, inputs_ref

    inputs_pad = paddle.to_tensor(inputs, dtype=paddle.float32)
    outputs_pad = model_pad(inputs_pad)
    reprod_log_pad.add("metric", outputs_pad.detach().numpy())
    del outputs_pad, inputs_pad

    reprod_log_ref.save('./result/metric_ref.npy')
    reprod_log_pad.save('./result/metric_paddle.npy')

    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info("./result/metric_paddle.npy")
    info2 = diff_helper.load_info("./result/metric_ref.npy")

    diff_helper.compare_info(info1, info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./result/log/metric_diff.log")

if __name__ == '__main__':
    metric_test()
