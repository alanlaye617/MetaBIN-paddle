import paddle
import torch
import numpy as np
from reprod_log import ReprodLogger, ReprodDiffHelper
import random
import sys
sys.path.append('.')
from utils import translate_weight, build_ref_model, build_ref_evaluator, translate_inputs_p2t
from modeling import Metalearning
from data import build_reid_test_loader
from evaluation import ReidEvaluator

def metric_test():
    paddle.device.set_device('gpu:0')

    seed = 2022
    np.random.seed(seed)
    random.seed(seed)

    reprod_log_ref = ReprodLogger()
    reprod_log_pad = ReprodLogger()

    torch_path = "./model_weights/model.pth"
    paddle_path = "./model_weights/model.pdparams"

    model_ref = build_ref_model(4).cuda()
#    model_ref = build_ref_model().backbone.cuda()
    torch.save(model_ref.state_dict(), torch_path)
    model_ref.eval()

    translate_weight(torch_path, paddle_path)

#    model_pad = build_resnet_backbone()
    model_pad = Metalearning(num_classes=4)
    model_pad.set_state_dict(paddle.load(paddle_path))
    model_pad.eval()

    test_loader, num_query= build_reid_test_loader('LiteData', 20, num_workers=0)
    inputs = next(test_loader.__iter__())

    inputs_ref = translate_inputs_p2t(inputs, if_train=False)
    outputs_ref = model_ref(inputs_ref)
    evaluator_ref = build_ref_evaluator(num_query)
    evaluator_ref.process(inputs_ref, outputs_ref.cpu().detach())
    metric_ref = evaluator_ref.evaluate()
    reprod_log_ref.add("Rank-1", np.array([metric_ref['Rank-1']]))
    reprod_log_ref.add("Rank-5", np.array([metric_ref['Rank-5']]))
    reprod_log_ref.add("Rank-10", np.array([metric_ref['Rank-10']]))
    reprod_log_ref.add("mAP", np.array([metric_ref['mAP']]))
    reprod_log_ref.add("mINP", np.array([metric_ref['mINP']]))
    
    evaluator_pad = ReidEvaluator(num_query)
    inputs_pad = inputs
    outputs_pad = model_pad(inputs_pad)
    evaluator_pad.process(inputs_pad, outputs_pad.cpu().detach())
    metric_pad = evaluator_pad.evaluate()
    reprod_log_pad.add("Rank-1", np.array([metric_pad['Rank-1']]))
    reprod_log_pad.add("Rank-5", np.array([metric_pad['Rank-5']]))
    reprod_log_pad.add("Rank-10", np.array([metric_pad['Rank-10']]))
    reprod_log_pad.add("mAP",  np.array([metric_pad['mAP']]))
    reprod_log_pad.add("mINP",  np.array([metric_pad['mINP']]))

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
