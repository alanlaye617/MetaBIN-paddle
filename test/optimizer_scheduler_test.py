import sys
sys.path.append('.')
from optim import build_optimizer, build_lr_scheduler
from utils import build_ref_trainer, translate_params_name_t2p
from modeling import Metalearning
from tqdm import tqdm

base_lr = 0.01
gate_factor = 20

trainer_ref = build_ref_trainer(batch_size=16)
scheduler_main_ref = trainer_ref.scheduler_main
scheduler_norm_ref = trainer_ref.scheduler_norm
optimizer_main_ref = trainer_ref.optimizer_main
optimizer_norm_ref = trainer_ref.optimizer_norm

model_pad = Metalearning(num_classes=751)
scheduler_main_pad = build_lr_scheduler(
        milestones=scheduler_main_ref.milestones,
        gamma=scheduler_main_ref.gamma,
        warmup_factor=scheduler_main_ref.warmup_factor,
        warmup_iters=scheduler_main_ref.warmup_iters,
        warmup_method=scheduler_main_ref.warmup_method,
        last_epoch=-1,
        verbose=False
        )
scheduler_norm_pad = build_lr_scheduler(
        milestones=scheduler_norm_ref.milestones,
        gamma=scheduler_norm_ref.gamma,
        warmup_factor=scheduler_norm_ref.warmup_factor,
        warmup_iters=scheduler_norm_ref.warmup_iters,
        warmup_method=scheduler_norm_ref.warmup_method,
        last_epoch=-1,
        verbose=False
        )

optimizer_main_pad = build_optimizer(model_pad, learning_rate=base_lr, lr_scheduler=scheduler_main_pad, flag='main')
optimizer_norm_pad = build_optimizer(model_pad, learning_rate=base_lr, lr_scheduler=scheduler_norm_pad, flag='norm')


# optimitzer test    
dict_ref = {k['name']:{'lr': k['initial_lr'], 'weight_decay': k['weight_decay']}for k in optimizer_main_ref.param_groups}
dict_ref.update({k['name']:{'lr': k['initial_lr'], 'weight_decay': k['weight_decay']}for k in optimizer_norm_ref.param_groups})

dict_pad = {k['name']:{'lr': k['learning_rate'], 'weight_decay': k['regularization_coeff']}for k in optimizer_main_pad._param_groups}
dict_pad.update({k['name']:{'lr': k['learning_rate'], 'weight_decay': k['regularization_coeff']}for k in optimizer_norm_pad._param_groups})

for name_ref, v in tqdm(dict_ref.items()):
    name_pad = translate_params_name_t2p(name_ref)
    assert v['lr'] == dict_pad[name_pad]['lr'] and v['weight_decay'] == dict_pad[name_pad]['weight_decay'], AssertionError('{} in torch does not match {} in paddle'.format(name_ref, name_pad))

# lr scheduler test
for i in tqdm(range(20)):
    optimizer_main_ref.step()
    scheduler_main_ref.step()
    optimizer_main_pad.step()
    scheduler_main_pad.step()
    assert scheduler_main_pad.get_lr()*base_lr == scheduler_main_ref.get_lr()[0], AssertionError('main lr scheduler does not match.')
    optimizer_norm_ref.step()
    scheduler_norm_ref.step()
    optimizer_norm_pad.step()
    scheduler_norm_pad.step()
    assert scheduler_norm_pad.get_lr()*gate_factor*base_lr == scheduler_norm_ref.get_lr()[0], AssertionError('norm lr scheduler does not match.')

print('passed')