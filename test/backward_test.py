import sys
import paddle
import torch
import numpy as np
from reprod_log import ReprodLogger, ReprodDiffHelper
import random
sys.path.append('.')
from utils import translate_weight, build_ref_trainer, translate_inputs_t2p
from modeling import Metalearning
from solver import build_lr_scheduler, build_optimizer

def backward_test():
    paddle.device.set_device('gpu:0')

    seed = 2022
    np.random.seed(seed)
    random.seed(seed)

    reprod_log_ref = ReprodLogger()
    reprod_log_pad = ReprodLogger()

    torch_path = "./model_weights/model.pth"
    paddle_path = "./model_weights/model.pdparams"

    trainer_ref = build_ref_trainer(num_classes=751, batch_size=16)
    train_loader = trainer_ref.data_loader
    model_ref = trainer_ref.model.cuda()
    optimizer_ref = trainer_ref.optimizer_main
    scheduler_ref = trainer_ref.scheduler_main
    torch.save(model_ref.state_dict(), torch_path)
#    model_ref.eval()

    translate_weight(torch_path, paddle_path)

    model_pad = Metalearning(751)
    model_pad.set_state_dict(paddle.load(paddle_path))
#    model_pad.eval()
    scheduler_pad = build_lr_scheduler(
        learning_rate=0.01, 
        milestones=scheduler_ref.milestones,
        gamma=scheduler_ref.gamma,
        warmup_factor=scheduler_ref.warmup_factor,
        warmup_iters=scheduler_ref.warmup_iters,
        warmup_method=scheduler_ref.warmup_method,
        last_epoch=-1,
        verbose=False
        )

    optimizer_pad = build_optimizer(model=model_pad, lr_scheduler=scheduler_pad, momentum=0.9, flag='main')
    for i in range(2):
        inputs_ref = next(train_loader.__iter__())
        outputs_ref = model_ref(inputs_ref, {'param_update': False, 'type_running_stats': 'general', 'each_domain': False})
        losses_ref = model_ref.losses(outputs_ref, opt={'loss':['CrossEntropyLoss', 'TripletLoss']})

        reprod_log_ref.add("CEloss_%d"%(i), losses_ref['loss_cls'].cpu().detach().numpy())
        reprod_log_ref.add("Tripletloss_%d"%(i), losses_ref['loss_triplet'].cpu().detach().numpy())

        losses_ref['loss_cls'].backward()
        optimizer_ref.step()
        optimizer_ref.zero_grad()
        scheduler_ref.step()

        inputs_pad = translate_inputs_t2p(inputs_ref)
        outputs_pad = model_pad(inputs_pad, {'param_update': False,  'type_running_stats': 'general', 'each_domain': False})
        losses_pad = model_pad.losses(outputs_pad, opt={'loss':['CrossEntropyLoss', 'TripletLoss']})

        reprod_log_pad.add("CEloss_%d"%(i), losses_pad['loss_cls'].cpu().detach().numpy())
        reprod_log_pad.add("Tripletloss_%d"%(i), losses_pad['loss_triplet'].cpu().detach().numpy())

        losses_pad['loss_cls'].backward()
        optimizer_pad.step()
        optimizer_pad.clear_grad()
        scheduler_pad.step()

    reprod_log_ref.save('./result/backward_ref.npy')
    reprod_log_pad.save('./result/backward_paddle.npy')

    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info("./result/backward_paddle.npy")
    info2 = diff_helper.load_info("./result/backward_ref.npy")

    diff_helper.compare_info(info1, info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./result/log/backward_diff.log")
    

if __name__ == '__main__':
    backward_test()

