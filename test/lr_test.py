import sys
sys.path.append('.')
from train import Trainer
from utils.build_ref import build_ref_trainer
from utils import translate_params_name_t2p
from tqdm import tqdm

def get_curr_lr(param, lr_scheduler):
    assert len(param['params']) == 1
    local_lr = param['params'][0].optimize_attr.get('learning_rate', 1.0)
    global_lr = lr_scheduler.last_lr
    return local_lr * global_lr

def get_weight_decay(param):
    assert len(param['params']) == 1
    return param['params'][0].optimize_attr.get('weight_decay', param['regularization_coeff'])


trainer_ref = build_ref_trainer(batch_size=16, resume=False, eval_only=False)
trainer_pad = Trainer(train_batch_size=16)

optimizer_main_ref = trainer_ref.optimizer_main
optimizer_norm_ref = trainer_ref.optimizer_norm
optimizer_main_pad = trainer_pad.optimizer_main
optimizer_norm_pad = trainer_pad.optimizer_norm

scheduler_main_ref = trainer_ref.scheduler_main
scheduler_norm_ref = trainer_ref.scheduler_norm
scheduler_main_pad = trainer_pad.scheduler_main
scheduler_norm_pad = trainer_pad.scheduler_norm

assert optimizer_main_pad._default_dict['momentum'] == optimizer_main_ref.defaults['momentum'], 'The momentum of main optimizer is different.'
assert optimizer_norm_pad._default_dict['momentum'] == optimizer_norm_ref.defaults['momentum'], 'The momentum of norm optimizer is different.'

# stop gradient test
named_parameters_pad = {k: v for k, v in trainer_pad.model.named_parameters()}
for name_ref, param_ref in trainer_ref.model.named_parameters():
    name_pad = translate_params_name_t2p(name_ref)
    param_pad = named_parameters_pad[name_pad]
    assert param_pad.stop_gradient != param_ref.requires_grad, 'The gradient setting of {} is different.'.format(name_pad)

for i in tqdm(range(2000)):
    param_groups_ref = {**{k['name']: {'lr': k['lr'], 'weight_decay': k['weight_decay']} for k in optimizer_main_ref.param_groups},
                        **{k['name']: {'lr': k['lr'], 'weight_decay': k['weight_decay']} for k in optimizer_norm_ref.param_groups}}

    param_groups_pad = {**{k['name']: {'lr': get_curr_lr(k, scheduler_main_pad), 'weight_decay': get_weight_decay(k)} for k in optimizer_main_pad._param_groups},
                        **{k['name']: {'lr': get_curr_lr(k, scheduler_norm_pad), 'weight_decay': get_weight_decay(k)} for k in optimizer_norm_pad._param_groups}}

    assert len(param_groups_ref) == len(param_groups_pad), 'The length of param groups is different.'
    for name_ref, v in param_groups_ref.items():
        name_pad = translate_params_name_t2p(name_ref)

        # learning rate test
        lr_ref = param_groups_ref[name_ref]['lr']
        lr_pad = param_groups_pad[name_pad]['lr']
        assert lr_ref == lr_pad, 'The learning rate of {} is different.'.format(name_pad)

        # weight decay test
        weight_decay_ref = param_groups_ref[name_ref]['weight_decay']
        weight_decay_pad = param_groups_pad[name_pad]['weight_decay']
        assert weight_decay_ref == weight_decay_pad, 'The weight decay of {} is different.'.format(name_pad)
    
    # cyclic learning rate test
    cyclic_lr_pad = trainer_pad.cyclic_scheduler.get_lr()
    cyclic_lr_ref = trainer_ref.cyclic_optimizer.param_groups[0]['lr']
    assert cyclic_lr_pad == cyclic_lr_ref, 'The cyclic lr is different.'

    optimizer_main_pad.step()
    optimizer_norm_pad.step()
    optimizer_main_ref.step()
    optimizer_norm_ref.step()

    scheduler_main_pad.step()
    scheduler_norm_pad.step()
    scheduler_main_ref.step()
    scheduler_norm_ref.step()

    trainer_ref.cyclic_scheduler.step()
    trainer_pad.cyclic_scheduler.step()

print('PASS')