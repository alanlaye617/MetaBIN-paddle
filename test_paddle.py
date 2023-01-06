#from utils.build_ref import build_ref_model
#import torch
#from utils import translate_weight
from train.trainer import Trainer
import paddle
trainer = Trainer(train_batch_size=32)
trainer.train()
#trainer.model.set_state_dict(paddle.load('model_weights/final.pdparams'))

#trainer.test(dataset_name='DukeMTMC', model=trainer.model, batch_size=128)