import sys
import paddle
import torch
import numpy as np
from reprod_log import ReprodLogger, ReprodDiffHelper
import random
sys.path.append('.')
from utils import translate_weight, build_ref_trainer, translate_inputs_t2p, translate_inputs_p2t
from modeling import Metalearning
from data import build_train_loader_for_m_resnet
from optim import build_lr_scheduler, build_optimizer
from tqdm import tqdm

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
   # model_ref.eval()

    translate_weight(torch_path, paddle_path)

    model_pad = Metalearning(751)
    model_pad.set_state_dict(paddle.load(paddle_path))
   # model_pad.eval()

    scheduler_pad = build_lr_scheduler(
        milestones=scheduler_ref.milestones,
        gamma=scheduler_ref.gamma,
        warmup_factor=scheduler_ref.warmup_factor,
        warmup_iters=scheduler_ref.warmup_iters,
        warmup_method=scheduler_ref.warmup_method,
        last_epoch=-1,
        verbose=False
        )
    
 #   model_pad.heads.classifier_norm.weight.set_value(model_pad.heads.classifier_norm.weight*1000)
    optimizer_pad = build_optimizer(model=model_pad, learning_rate=0.01, lr_scheduler=scheduler_pad, momentum=0.9, flag='main')        
  #  optimizer_pad = build_optimizer(model=model_pad, lr_scheduler=0.001, momentum=0.9, flag='main')
    #train_loader, mtrain_loader, mtest_loader, num_domains = build_train_loader_for_m_resnet(['Market1501'], batch_size=16, num_workers=0)
    for i in tqdm(range(5)):
        inputs_ref = next(train_loader.__iter__())
       # inputs_ref = translate_inputs_p2t(inputs_pad)
        outputs_ref = model_ref(inputs_ref, {'param_update': False, 'type_running_stats': 'general', 'each_domain': False})
        losses_ref = model_ref.losses(outputs_ref, opt={'loss':['CrossEntropyLoss', 'TripletLoss']})
#        reprod_log_ref.add("output_%d"%(i), outputs_ref['outputs']['bn_features'].cpu().detach().numpy())
        reprod_log_ref.add("CEloss_%d"%(i), losses_ref['loss_cls'].cpu().detach().numpy())
#        reprod_log_ref.add("Tripletloss_%d"%(i), losses_ref['loss_triplet'].cpu().detach().numpy())
        print('Paddle:','CELoss',str(losses_ref['loss_cls']))

        inputs_pad = translate_inputs_t2p(inputs_ref)

        outputs_pad = model_pad(inputs_pad, {'param_update': False,  'type_running_stats': 'general', 'each_domain': False})
        losses_pad = model_pad.losses(outputs_pad, opt={'loss':['CrossEntropyLoss', 'TripletLoss']})
#        reprod_log_pad.add("output_%d"%(i), outputs_pad['outputs']['bn_features'].cpu().detach().numpy())
        reprod_log_pad.add("CEloss_%d"%(i), losses_pad['loss_cls'].cpu().detach().numpy())
#        reprod_log_pad.add("Tripletloss_%d"%(i), losses_pad['loss_triplet'].cpu().detach().numpy())
        print('PyTorch:','CELoss',str(losses_ref['loss_cls']))

        losses_ref['loss_cls'].backward()
        losses_pad['loss_cls'].backward()
        '''
        if i == 1:
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
        
        optimizer_ref.step()
        scheduler_ref.step()

        optimizer_pad.step()
        scheduler_pad.step()
        '''
        if i == 1:
            for key, value in model_ref.named_parameters():
                if 'fc' not in key and 'mean' not in key and 'var' not in key:
                    reprod_log_ref.add(key+'_value', value.data.cpu().detach().numpy())
            for key, value in model_pad.named_parameters():
                if 'fc' not in key and 'mean' not in key and 'var' not in key:
                    if 'ins_n' in key:
                        key = key.replace('scale', 'weight')
                    reprod_log_pad.add(key+'_value', value.cpu().detach().numpy())
        '''
        optimizer_pad.clear_grad()
        optimizer_ref.zero_grad()


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

