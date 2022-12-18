import paddle
from paddle import nn
import paddle.nn.functional as F
import torch
import numpy as np
from reprod_log import ReprodLogger, ReprodDiffHelper
import random
import sys
sys.path.append('.')
from utils import translate_weight, build_ref_model, translate_inputs
from modeling import build_resnet_backbone, Metalearning
from data import build_train_loader_for_m_resnet

def loss_test():
    paddle.device.set_device('gpu:0')

    seed = 2022
    np.random.seed = seed
    random.seed = seed

    reprod_log_ref = ReprodLogger()
    reprod_log_pad = ReprodLogger()

    torch_path = "./model_weights/model.pth"
    paddle_path = "./model_weights/model.pdparams"

    model_ref = build_ref_model(num_classes=4).cuda()
 #   model_ref.eval()

    torch.save(model_ref.state_dict(), torch_path)
#
    translate_weight(torch_path, paddle_path)
    model_pad = Metalearning(num_classes=4)
    model_pad.set_state_dict(paddle.load(paddle_path))
 #   model_pad.eval()

    train_loader, mtrain_loader, mtest_loader, num_domains = build_train_loader_for_m_resnet(['LiteData'], batch_size=16, num_workers=0)

    inputs = next(train_loader.__iter__())

    inputs_ref = translate_inputs(inputs)
    output_ref = model_ref(inputs_ref, {'param_update': False, 'loss': ('CrossEntropyLoss', 'TripletLoss'), 'type_running_stats': 'general', 'each_domain': False})
    losses_ref = model_ref.losses(output_ref, opt={'loss':['CrossEntropyLoss', "TripletLoss"]})

#    reprod_log_ref.add("pred_class_logits", output_ref['outputs']['pred_class_logits'].cpu().detach().numpy())
#    reprod_log_ref.add("cls_outputs", output_ref['outputs']['cls_outputs'].cpu().detach().numpy())
#    reprod_log_ref.add("pooled_features", output_ref['outputs']['pooled_features'].cpu().detach().numpy())
#    reprod_log_ref.add("bn_features", output_ref['outputs']['bn_features'].cpu().detach().numpy())

    reprod_log_ref.add("CEloss", losses_ref['loss_cls'].cpu().detach().numpy())
    reprod_log_ref.add("Tripletloss", losses_ref['loss_triplet'].cpu().detach().numpy())

    inputs_pad = inputs
    outputs_pad = model_pad(inputs_pad, {'param_update': False, 'loss': ('CrossEntropyLoss', 'TripletLoss'), 'type_running_stats': 'general', 'each_domain': False})
    losses_pad = model_pad.losses(outputs_pad, opt={'loss':['CrossEntropyLoss', "TripletLoss"]})

#    reprod_log_pad.add("pred_class_logits", outputs_pad['outputs']['pred_class_logits'].cpu().detach().numpy())
#    reprod_log_pad.add("cls_outputs", outputs_pad['outputs']['cls_outputs'].cpu().detach().numpy())
#    reprod_log_pad.add("pooled_features", outputs_pad['outputs']['pooled_features'].cpu().detach().numpy())
#    reprod_log_pad.add("bn_features", outputs_pad['outputs']['bn_features'].cpu().detach().numpy())

    reprod_log_pad.add("CEloss", losses_pad['loss_cls'].cpu().detach().numpy())
    reprod_log_pad.add("Tripletloss", losses_pad['loss_triplet'].cpu().detach().numpy())

    reprod_log_ref.save('./result/loss_ref.npy')
    reprod_log_pad.save('./result/loss_paddle.npy')

    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info("./result/loss_paddle.npy")
    info2 = diff_helper.load_info("./result/loss_ref.npy")

    diff_helper.compare_info(info1, info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./result/log/loss_diff.log")

if __name__ == '__main__':
    loss_test()
