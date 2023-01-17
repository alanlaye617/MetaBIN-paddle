import sys
sys.path.append('.')
import paddle
import torch
from engine import Trainer
from utils.build_ref import build_ref_trainer
from reprod_log import ReprodLogger, ReprodDiffHelper
from utils import translate_weight, translate_inputs_p2t, translate_params_name_t2p
from utils.build_ref import build_ref_model
from arch import Metalearning
from data import build_train_loader_for_m_resnet
from tqdm import tqdm
from paddle.optimizer import Momentum
from refs.fastreid.solver.optim.sgd import SGD
import numpy as np
import torch.nn as nn

reprod_log_ref = ReprodLogger()
reprod_log_pad = ReprodLogger()
torch_path = "./model_weights/model.pth"
paddle_path = "./model_weights/model.pdparams"

trainer_ref = build_ref_trainer(batch_size=48, resume=True)
trainer_pad = Trainer(train_batch_size=48)
model_ref = trainer_ref.model
model_pad = trainer_pad.model

assert trainer_pad.optimizer_main._default_dict['momentum'] == trainer_ref.optimizer_main.defaults['momentum'], 'The momentum of main optimizer is different.'
assert trainer_pad.optimizer_norm._default_dict['momentum'] == trainer_ref.optimizer_norm.defaults['momentum'], 'The momentum of norm optimizer is different.'

#model_ref = build_ref_model(751)
#model_pad = Metalearning(751)

torch.save(model_ref.state_dict(), torch_path)
translate_weight(torch_path, paddle_path)
model_pad.set_state_dict(paddle.load(paddle_path))

#optimizer_pad = Momentum(learning_rate=1e-4, momentum=0.9, parameters=model_pad.parameters())
#optimizer_ref = SGD(params=model_ref.parameters(), lr=1e-4, momentum=0.9)

train_loader, mtrain_loader, mtest_loader, num_domains = build_train_loader_for_m_resnet(['LiteData'], batch_size=16, num_workers=0)
grad_value = 1.0
opt = {'param_update': False, 'loss': ('CrossEntropyLoss', 'TripletLoss'), 'type_running_stats': 'general', 'each_domain': False}
for i in range(2):
    named_parameters_pad = {k:v for k, v in model_pad.named_parameters()}
    for name_ref, param_ref in model_ref.named_parameters():
        name_pad = translate_params_name_t2p(name_ref)
        param_pad = named_parameters_pad[name_pad]
        assert param_ref.requires_grad != param_pad.stop_gradient
        if param_ref.requires_grad and name_pad != 'heads.classifier_norm.weight':
            param_ref.grad = torch.full_like(param_ref, grad_value)
            param_pad.grad = paddle.full_like(param_pad, grad_value)

    trainer_ref.optimizer_main.step()
    trainer_ref.optimizer_norm.step()
    trainer_ref.scheduler_main.step()
    trainer_ref.scheduler_norm.step()

    trainer_pad.optimizer_main.step()
    trainer_pad.optimizer_norm.step()
    trainer_pad.scheduler_main.step()
    trainer_pad.scheduler_norm.step()

    inputs_pad = next(train_loader.__iter__())
    inputs_ref = translate_inputs_p2t(inputs_pad)

    outputs_ref = model_ref(inputs_ref, opt)

    reprod_log_ref.add("pred_class_logits_%d"%(i), outputs_ref['outputs']['pred_class_logits'].cpu().detach().numpy())
    reprod_log_ref.add("cls_outputs_%d"%(i), outputs_ref['outputs']['cls_outputs'].cpu().detach().numpy())
    reprod_log_ref.add("pooled_features_%d"%(i), outputs_ref['outputs']['pooled_features'].cpu().detach().numpy())
    reprod_log_ref.add("bn_features_%d"%(i), outputs_ref['outputs']['bn_features'].cpu().detach().numpy())


    outputs_pad = model_pad(inputs_pad, opt)
    reprod_log_pad.add("pred_class_logits_%d"%(i), outputs_pad['outputs']['pred_class_logits'].cpu().detach().numpy())
    reprod_log_pad.add("cls_outputs_%d"%(i), outputs_pad['outputs']['cls_outputs'].cpu().detach().numpy())
    reprod_log_pad.add("pooled_features_%d"%(i), outputs_pad['outputs']['pooled_features'].cpu().detach().numpy())
    reprod_log_pad.add("bn_features_%d"%(i), outputs_pad['outputs']['bn_features'].cpu().detach().numpy())

reprod_log_ref.save('./result/optimizer_ref.npy')
reprod_log_pad.save('./result/optimizer_paddle.npy')

diff_helper = ReprodDiffHelper()

info1 = diff_helper.load_info("./result/optimizer_paddle.npy")
info2 = diff_helper.load_info("./result/optimizer_ref.npy")

diff_helper.compare_info(info1, info2)

diff_helper.report(
    diff_method="mean", diff_threshold=1e-6, path="./result/log/optimizer_diff.log")
