import paddle
import torch
import numpy as np
from reprod_log import ReprodLogger, ReprodDiffHelper
import random
import sys
sys.path.append('.')
from utils import translate_weight, build_ref_trainer, build_ref_evaluator, translate_inputs_p2t
from modeling import Metalearning
from data import build_reid_test_loader
from evaluation import ReidEvaluator 
from evaluation import inference_on_dataset as inference_on_dataset_pad
from refs.fastreid.evaluation.evaluator import inference_on_dataset as inference_on_dataset_ref


def metric_test():
    paddle.device.set_device('gpu:0')

    seed = 2022
    np.random.seed(seed)
    random.seed(seed)

    reprod_log_ref = ReprodLogger()
    reprod_log_pad = ReprodLogger()  

    torch_path = "./model_weights/model.pth"
    paddle_path = "./model_weights/model.pdparams"

    batch_size=128

    trainer_ref = build_ref_trainer(batch_size=batch_size, test_dataset=['DukeMTMC'], resume=True)
    model_ref = trainer_ref.model.cuda()
    dataloader_ref, num_query = trainer_ref.build_test_loader(trainer_ref.cfg, 'DukeMTMC')
    evaluator_ref = trainer_ref.build_evaluator(trainer_ref.cfg, num_query)
    model_ref.eval()
    torch.save(model_ref.state_dict(), torch_path)
    translate_weight(torch_path, paddle_path)

    metric_ref = inference_on_dataset_ref(model_ref, dataloader_ref, evaluator_ref)

    reprod_log_ref.add("Rank-1", np.array([metric_ref['Rank-1']]))
    reprod_log_ref.add("Rank-5", np.array([metric_ref['Rank-5']]))
    reprod_log_ref.add("Rank-10", np.array([metric_ref['Rank-10']]))
    reprod_log_ref.add("mAP", np.array([metric_ref['mAP']]))
    reprod_log_ref.add("mINP", np.array([metric_ref['mINP']]))
    print(metric_ref)
    del metric_ref, model_ref
    model_pad = Metalearning(num_classes=751)
    model_pad.set_state_dict(paddle.load(paddle_path))
    model_pad.eval()

    dataloader_pad, num_query= build_reid_test_loader('DukeMTMC', batch_size, num_workers=2, flag_test=True)
    evaluator_pad = ReidEvaluator(num_query=num_query)
    metric_pad = inference_on_dataset_pad(model_pad, dataloader_pad, evaluator_pad)


    reprod_log_pad.add("Rank-1", np.array([metric_pad['Rank-1']]))
    reprod_log_pad.add("Rank-5", np.array([metric_pad['Rank-5']]))
    reprod_log_pad.add("Rank-10", np.array([metric_pad['Rank-10']]))
    reprod_log_pad.add("mAP",  np.array([metric_pad['mAP']]))
    reprod_log_pad.add("mINP",  np.array([metric_pad['mINP']]))
    print(metric_pad)

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
