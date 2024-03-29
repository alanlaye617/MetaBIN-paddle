import sys
import paddle
import torch
import numpy as np
from reprod_log import ReprodLogger, ReprodDiffHelper
import random
sys.path.append('.')
from utils import translate_weight, translate_inputs_p2t
from utils.build_ref import build_ref_trainer
from data import build_train_loader_for_m_resnet
from engine import Trainer

def forward_test():
    paddle.device.set_device('gpu:0')

    seed = 2022
    np.random.seed(seed)
    random.seed(seed)

    reprod_log_ref = ReprodLogger()
    reprod_log_pad = ReprodLogger()

    torch_path = "./model_weights/model.pth"
    paddle_path = "./model_weights/model.pdparams"

    #model_ref = build_ref_model(num_classes=751, resume=False, pretrain=False, eval_only=False).cuda()
    model_ref = build_ref_trainer(batch_size=16, resume=False, pretrain=False).model.cuda()
    #model_pad = Metalearning(751)
    model_pad = Trainer(train_batch_size=16).model
    
    torch.save(model_ref.state_dict(), torch_path)
    translate_weight(torch_path, paddle_path)
    model_pad.set_state_dict(paddle.load(paddle_path))

#    model_ref.eval()
#    model_pad.eval()
    train_loader, mtrain_loader, mtest_loader, num_domains = build_train_loader_for_m_resnet(['Market1501'], batch_size=16, num_workers=0)
    inputs = next(train_loader.__iter__())
    inputs_ref = translate_inputs_p2t(inputs)
    outputs_ref = model_ref(inputs_ref, {'param_update': False, 'loss': ('CrossEntropyLoss', 'TripletLoss'), 'type_running_stats': 'general', 'each_domain': False})
    reprod_log_ref.add("pred_class_logits", outputs_ref['outputs']['pred_class_logits'].cpu().detach().numpy())
    reprod_log_ref.add("cls_outputs", outputs_ref['outputs']['cls_outputs'].cpu().detach().numpy())
    reprod_log_ref.add("pooled_features", outputs_ref['outputs']['pooled_features'].cpu().detach().numpy())
    reprod_log_ref.add("bn_features", outputs_ref['outputs']['bn_features'].cpu().detach().numpy())

    inputs_pad = inputs
    outputs_pad = model_pad(inputs_pad, {'param_update': False, 'loss': ('CrossEntropyLoss', 'TripletLoss'), 'type_running_stats': 'general', 'each_domain': False})
    reprod_log_pad.add("pred_class_logits", outputs_pad['outputs']['pred_class_logits'].cpu().detach().numpy())
    reprod_log_pad.add("cls_outputs", outputs_pad['outputs']['cls_outputs'].cpu().detach().numpy())
    reprod_log_pad.add("pooled_features", outputs_pad['outputs']['pooled_features'].cpu().detach().numpy())
    reprod_log_pad.add("bn_features", outputs_pad['outputs']['bn_features'].cpu().detach().numpy())

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