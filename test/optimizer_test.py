import sys
sys.path.append('.')
from optim import build_optimizer, build_lr_scheduler
from utils import build_ref_trainer, translate_params_name_t2p
from modeling import Metalearning

trainer_ref = build_ref_trainer(batch_size=16)
scheduler = build_lr_scheduler([1, 2])
optimizer_main_ref = trainer_ref.optimizer_main
optimizer_norm_ref = trainer_ref.optimizer_norm
model_pad = Metalearning(num_classes=751)
optimizer_main_pad = build_optimizer(model_pad, learning_rate=0.01, lr_scheduler=scheduler, flag='main')
optimizer_norm_pad = build_optimizer(model_pad, learning_rate=0.01, lr_scheduler=scheduler, flag='norm')
model_pad.named_parameters
dict_ref = {k['name']:{'lr': k['initial_lr'], 'weight_decay': k['weight_decay']}for k in optimizer_main_ref.param_groups}
dict_ref.update({k['name']:{'lr': k['initial_lr'], 'weight_decay': k['weight_decay']}for k in optimizer_norm_ref.param_groups})

dict_pad = {k['name']:{'lr': k['learning_rate'], 'weight_decay': k['regularization_coeff']}for k in optimizer_main_pad._param_groups}
dict_pad.update({k['name']:{'lr': k['learning_rate'], 'weight_decay': k['regularization_coeff']}for k in optimizer_norm_pad._param_groups})

for name_ref, v in dict_ref.items():
    name_pad = translate_params_name_t2p(name_ref)
    assert v['lr'] == dict_pad[name_pad]['lr'] and v['weight_decay'] == dict_pad[name_pad]['weight_decay'], ValueError('{} in torch does not match {} in paddle'.format(name_ref, name_pad))
print()
