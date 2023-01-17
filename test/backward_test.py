import sys
import paddle
import torch
import numpy as np
from reprod_log import ReprodLogger, ReprodDiffHelper
import random
sys.path.append('.')
from utils import translate_weight, translate_inputs_t2p
from utils.build_ref import build_ref_trainer
from optim import build_lr_scheduler, build_optimizer
from tqdm import tqdm
from engine import Trainer


def backward_test():
    paddle.device.set_device('gpu:0')

    seed = 2022
    np.random.seed(seed)
    random.seed(seed)

    reprod_log_ref = ReprodLogger()
    reprod_log_pad = ReprodLogger()

    torch_path = "./model_weights/model.pth"
    paddle_path = "./model_weights/model.pdparams"

    batch_size = 16
    base_lr = 0.01
    
    trainer_ref = build_ref_trainer(batch_size=batch_size, resume=True)
    train_loader = trainer_ref.data_loader
    model_ref = trainer_ref.model
    scheduler_main_ref = trainer_ref.scheduler_main
    scheduler_norm_ref = trainer_ref.scheduler_norm
    optimizer_main_ref = trainer_ref.optimizer_main
    optimizer_norm_ref = trainer_ref.optimizer_norm

    torch.save(model_ref.state_dict(), torch_path)
   # model_ref.eval()

    translate_weight(torch_path, paddle_path)

    trainer_pad = Trainer(train_batch_size=batch_size)
    model_pad = trainer_pad.model
    model_pad.set_state_dict(paddle.load(paddle_path))

    scheduler_main_pad = build_lr_scheduler(
            milestones=scheduler_main_ref.milestones,
            gamma=scheduler_main_ref.gamma,
            warmup_factor=scheduler_main_ref.warmup_factor,
            warmup_iters=scheduler_main_ref.warmup_iters,
            warmup_method=scheduler_main_ref.warmup_method,
            last_epoch=scheduler_main_ref.last_epoch-1,
            verbose=False
            )
    scheduler_norm_pad = build_lr_scheduler(
            milestones=scheduler_norm_ref.milestones,
            gamma=scheduler_norm_ref.gamma,
            warmup_factor=scheduler_norm_ref.warmup_factor,
            warmup_iters=scheduler_norm_ref.warmup_iters,
            warmup_method=scheduler_norm_ref.warmup_method,
            last_epoch=scheduler_norm_ref.last_epoch-1,
            verbose=False
            )

    optimizer_main_pad = build_optimizer(model_pad, base_lr=base_lr, lr_scheduler=scheduler_main_pad, momentum=0.9, flag='main')
    optimizer_norm_pad = build_optimizer(model_pad, base_lr=base_lr, lr_scheduler=scheduler_norm_pad, momentum=0, flag='norm')
    #optimizer_pad = Momentum(learning_rate=1e-5, momentum=0.9, parameters=model_pad.parameters())
    #optimizer_ref = SGD(params=model_ref.parameters(), lr=1e-5, momentum=0.9)
    opt_pad = trainer_pad.opt_setting('basic')
    opt_ref = trainer_ref.opt_setting('basic')

    for i in tqdm(range(5)):
        inputs_ref = next(train_loader.__iter__())
       # inputs_ref = translate_inputs_p2t(inputs_pad)
        outputs_ref = model_ref(inputs_ref, opt_ref)
        losses_ref = model_ref.losses(outputs_ref, opt_ref)
#        reprod_log_ref.add("output_%d"%(i), outputs_ref['outputs']['bn_features'].cpu().detach().numpy())
        reprod_log_ref.add("CEloss_%d"%(i), losses_ref['loss_cls'].cpu().detach().numpy())
      #  reprod_log_ref.add("Tripletloss_%d"%(i), losses_ref['loss_triplet'].cpu().detach().numpy())

        inputs_pad = translate_inputs_t2p(inputs_ref)

        outputs_pad = model_pad(inputs_pad, opt_pad)
        losses_pad = model_pad.losses(outputs_pad, opt_pad)
#        reprod_log_pad.add("output_%d"%(i), outputs_pad['outputs']['bn_features'].cpu().detach().numpy())
        reprod_log_pad.add("CEloss_%d"%(i), losses_pad['loss_cls'].cpu().detach().numpy())
     #   reprod_log_pad.add("Tripletloss_%d"%(i), losses_pad['loss_triplet'].cpu().detach().numpy())

        losses_ref['loss_cls'].backward()
        losses_pad['loss_cls'].backward()
        '''
        if i == 0:
            for key, value in model_ref.named_parameters():
                if 'fc' not in key and 'mean' not in key and 'var' not in key and value.grad is not None:
                    if 'heads.classifier_norm.weight' in key:
                        continue
                    reprod_log_ref.add(key+'_grad', value.grad.cpu().detach().numpy())
            for key, value in model_pad.named_parameters():
                if 'fc' not in key and 'mean' not in key and 'var' not in key and value.grad is not None:
                    if 'ins_n' in key:
                        key = key.replace('scale', 'weight')
                    reprod_log_pad.add(key+'_grad', value.grad.cpu().detach().numpy())
        '''
        #optimizer_pad.step()
        #optimizer_ref.step()

        optimizer_main_ref.step()
        scheduler_main_ref.step()

        optimizer_norm_ref.step()
        scheduler_norm_ref.step()

        optimizer_main_pad.step()
        scheduler_main_pad.step()

        optimizer_norm_pad.step()
        scheduler_norm_pad.step()

        '''
        if i == 0:
            for key, value in model_ref.named_parameters():
                if 'fc' not in key and 'mean' not in key and 'var' not in key:
                    reprod_log_ref.add(key+'_value', value.data.cpu().detach().numpy())
            for key, value in model_pad.named_parameters():
                if 'fc' not in key and 'mean' not in key and 'var' not in key:
                    if 'ins_n' in key:
                        key = key.replace('scale', 'weight')
                    reprod_log_pad.add(key+'_value', value.cpu().detach().numpy())
        '''
        optimizer_main_ref.zero_grad()
        optimizer_main_pad.clear_grad()
        optimizer_norm_ref.zero_grad()
        optimizer_norm_pad.clear_grad()
        #optimizer_pad.clear_grad()
        #optimizer_ref.zero_grad()

    reprod_log_ref.save('./result/backward_ref.npy')
    reprod_log_pad.save('./result/backward_paddle.npy')

    diff_helper = ReprodDiffHelper()

    info_ref = diff_helper.load_info("./result/backward_ref.npy")
    info_pad = diff_helper.load_info("./result/backward_paddle.npy")

    diff_helper.compare_info(info_ref, info_pad)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./result/log/backward_diff.log")
    

if __name__ == '__main__':
    backward_test()

