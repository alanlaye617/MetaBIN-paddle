import os 
import yaml
import copy
from collections import OrderedDict
import paddle
import paddle.optimizer
import paddle.amp
import numpy as np
import logging
import weakref
import time
import math
import re

from evaluation import ReidEvaluator, inference_on_dataset
from data import build_train_loader_for_m_resnet, build_reid_test_loader
from modeling import Metalearning
from optim import build_lr_scheduler, build_optimizer
from utils.events import EventStorage
from .hooks import *

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, train_batch_size=None, num_workers=2, mode='M-ResNet', output_dir='logs') -> None:
        self._hooks = []
        self.output_dir = output_dir
        assert mode in ['M-ResNet']
        self.cfg = self.get_cfg(mode)
        if train_batch_size != None: self.cfg['SOLVER']['IMS_PER_BATCH'] = train_batch_size
        
        self.data_loader, mtrain_loader, mtest_loader, num_domains = build_train_loader_for_m_resnet(
            dataset_list=self.cfg['DATASETS']['NAMES'],
            batch_size=self.cfg['SOLVER']['IMS_PER_BATCH'],
            num_instance=self.cfg['DATALOADER']['NUM_INSTANCE'],
            num_workers=num_workers,
            world_size=1)
        
        self.cfg['META']['DATA']['NUM_DOMAINS'] = num_domains
        self.auto_scale_hyperparams()

        num_classes = self.data_loader.dataset.num_classes
        self.model = self.build_model(num_classes, pretrain=True)

        self.scheduler_main = self.build_lr_scheduler(
            milestones=self.cfg['SOLVER']['STEPS'], 
            gamma=self.cfg['SOLVER']['GAMMA'],
            warmup_factor=self.cfg['SOLVER']['WARMUP_FACTOR'],
            warmup_iters=self.cfg['SOLVER']['WARMUP_ITERS'],
            warmup_method=self.cfg['SOLVER']['WARMUP_METHOD']
            )

        self.scheduler_norm = self.build_lr_scheduler(
            milestones=[100000000,1000000000],
            gamma=1.0,
            warmup_factor=self.cfg['SOLVER']['WARMUP_FACTOR'],
            warmup_iters=0,
            warmup_method=self.cfg['SOLVER']['WARMUP_METHOD']
            )

        self.optimizer_main = self.build_optimizer(
            model=self.model, 
            base_lr=self.cfg['SOLVER']['BASE_LR'],
            lr_scheduler=self.scheduler_main,
            momentum=self.cfg['SOLVER']['MOMENTUM'],
            flag='main')

        self.optimizer_norm = self.build_optimizer(
            model=self.model, 
            base_lr=self.cfg['SOLVER']['BASE_LR'],
            lr_scheduler=self.scheduler_norm,
            momentum=self.cfg['SOLVER']['MOMENTUM_NORM'],
            flag='norm')        
        
        meta_param = dict()
        meta_param['synth_data'] = self.cfg['META']['DATA']['SYNTH_FLAG']
        meta_param['synth_method'] = self.cfg['META']['DATA']['SYNTH_METHOD']
        meta_param['num_domain'] = self.cfg['META']['DATA']['NUM_DOMAINS']
        meta_param['whole'] = self.cfg['META']['DATA']['WHOLE']

        meta_param['meta_compute_layer'] = self.cfg['META']['MODEL']['META_COMPUTE_LAYER']
        meta_param['meta_update_layer'] = self.cfg['META']['MODEL']['META_UPDATE_LAYER']
        meta_param['meta_all_params'] = self.cfg['META']['MODEL']['ALL_PARAMS']

        meta_param['iter_init_inner'] = self.cfg['META']['SOLVER']['INIT']['INNER_LOOP']
        meta_param['iter_init_inner_first'] = self.cfg['META']['SOLVER']['INIT']['FIRST_INNER_LOOP']
        meta_param['iter_init_outer'] = self.cfg['META']['SOLVER']['INIT']['OUTER_LOOP']

        meta_param['update_ratio'] = self.cfg['META']['SOLVER']['LR_FACTOR']['META']
        # meta_param['update_ratio'] = cfg.META.SOLVER.LR_FACTOR.GATE_CYCLIC_RATIO
        # meta_param['update_ratio'] = cfg.META.SOLVER.LR_FACTOR.GATE_CYCLIC_PERIOD_PER_EPOCH
        meta_param['iters_per_epoch'] = self.cfg['SOLVER']['ITERS_PER_EPOCH']


        meta_param['iter_mtrain'] = self.cfg['META']['SOLVER']['MTRAIN']['INNER_LOOP']
        meta_param['shuffle_domain'] = self.cfg['META']['SOLVER']['MTRAIN']['SHUFFLE_DOMAIN']
        meta_param['use_second_order'] = self.cfg['META']['SOLVER']['MTRAIN']['SECOND_ORDER']
        meta_param['num_mtrain'] = self.cfg['META']['SOLVER']['MTRAIN']['NUM_DOMAIN']
        meta_param['freeze_gradient_meta'] = self.cfg['META']['SOLVER']['MTRAIN']['FREEZE_GRAD_META']
        meta_param['allow_unused'] = self.cfg['META']['SOLVER']['MTRAIN']['ALLOW_UNUSED']
        meta_param['zero_grad'] = self.cfg['META']['SOLVER']['MTRAIN']['BEFORE_ZERO_GRAD']
        meta_param['type_running_stats_init'] = self.cfg['META']['SOLVER']['INIT']['TYPE_RUNNING_STATS']
        meta_param['type_running_stats_mtrain'] = self.cfg['META']['SOLVER']['MTRAIN']['TYPE_RUNNING_STATS']
        meta_param['type_running_stats_mtest'] = self.cfg['META']['SOLVER']['MTEST']['TYPE_RUNNING_STATS']
        meta_param['auto_grad_outside'] = self.cfg['META']['SOLVER']['AUTO_GRAD_OUTSIDE']
        meta_param['inner_clamp'] = self.cfg['META']['SOLVER']['INNER_CLAMP']
        meta_param['synth_grad'] = self.cfg['META']['SOLVER']['SYNTH_GRAD']
        meta_param['constant_grad'] = self.cfg['META']['SOLVER']['CONSTANT_GRAD']
        meta_param['random_scale_grad'] = self.cfg['META']['SOLVER']['RANDOM_SCALE_GRAD']
        meta_param['print_grad'] = self.cfg['META']['SOLVER']['PRINT_GRAD']
        meta_param['one_loss_for_iter'] = self.cfg['META']['SOLVER']['ONE_LOSS_FOR_ITER']
        meta_param['one_loss_order'] = self.cfg['META']['SOLVER']['ONE_LOSS_ORDER']

        if self.cfg['META']['SOLVER']['MTEST']['ONLY_ONE_DOMAIN']:
            meta_param['num_mtest'] = 1
        else:
            meta_param['num_mtest'] = meta_param['num_domain'] - meta_param['num_mtrain']

        meta_param['sync'] = self.cfg['META']['SOLVER']['SYNC']
        meta_param['detail_mode'] = self.cfg['META']['SOLVER']['DETAIL_MODE']
        meta_param['stop_gradient'] = self.cfg['META']['SOLVER']['STOP_GRADIENT']
        meta_param['flag_manual_zero_grad'] = self.cfg['META']['SOLVER']['MANUAL_ZERO_GRAD']
        meta_param['flag_manual_memory_empty'] = self.cfg['META']['SOLVER']['MANUAL_MEMORY_EMPTY']



        meta_param['main_zero_grad'] = self.cfg['META']['NEW_SOLVER']['MAIN_ZERO_GRAD']
        meta_param['norm_zero_grad'] = self.cfg['META']['NEW_SOLVER']['NORM_ZERO_GRAD']
        meta_param['momentum_init_grad'] = self.cfg['META']['NEW_SOLVER']['MOMENTUM_INIT_GRAD']

        meta_param['write_period_param'] = self.cfg['SOLVER']['WRITE_PERIOD_PARAM']


        meta_param['loss_combined'] = self.cfg['META']['LOSS']['COMBINED']
        meta_param['loss_weight'] = self.cfg['META']['LOSS']['WEIGHT']
        meta_param['loss_name_mtrain'] = self.cfg['META']['LOSS']['MTRAIN_NAME']
        meta_param['loss_name_mtest'] = self.cfg['META']['LOSS']['MTEST_NAME']

        logger.info('-' * 30)
        logger.info('Meta-learning paramters')
        logger.info('-' * 30)
        for name, val in meta_param.items():
            logger.info('[M_param] {}: {}'.format(name, val))
        logger.info('-' * 30)

        meta_param['update_cyclic_ratio'] = self.cfg['META']['SOLVER']['LR_FACTOR']['META_CYCLIC_RATIO']
        meta_param['update_cyclic_period'] = self.cfg['META']['SOLVER']['LR_FACTOR']['META_CYCLIC_PERIOD_PER_EPOCH']
        meta_param['update_cyclic_new'] = self.cfg['META']['SOLVER']['LR_FACTOR']['META_CYCLIC_NEW']

        if meta_param['update_ratio'] == 0.0:
            if meta_param['update_cyclic_new']:
                meta_param['update_cyclic_up_ratio'] = self.cfg['META']['SOLVER']['LR_FACTOR']['META_CYCLIC_UP_RATIO']
                meta_param['update_cyclic_middle_lr'] = self.cfg['META']['SOLVER']['LR_FACTOR']['META_CYCLIC_MIDDLE_LR']

                one_period = int(meta_param['iters_per_epoch'] / meta_param['update_cyclic_period'])
                num_step_up = int(one_period * meta_param['update_cyclic_up_ratio'])
                num_step_down = one_period - num_step_up
                if num_step_up <= 0:
                    num_step_up = 1
                    num_step_down = one_period - 1
                if num_step_down <= 0:
                    num_step_up = one_period - 1
                    num_step_down = 1

                self.cyclic_scheduler = paddle.optimizer.lr.CyclicLR(
                    base_learning_rate= meta_param['update_cyclic_middle_lr'] / meta_param['update_cyclic_ratio'],
                    max_learning_rate= meta_param['update_cyclic_middle_lr'] * meta_param['update_cyclic_ratio'],
                    step_size_up = num_step_up,
                    step_size_down = num_step_down,
                )
        
        
        self._data_loader_iter = iter(self.data_loader)
        
        self.meta_param = meta_param
        if self.cfg['SOLVER']['AMP']:
            self.scaler = paddle.amp.GradScaler()
        else:
            self.scaler = None

        # balancing parameters
        self.bin_gates = [p for p in self.model.parameters() if getattr(p, 'bin_gate', False)]
        self.bin_names = [name for name, values in self.model.named_parameters() if getattr(values, 'bin_gate', False)]

        # Meta-leaning setting
        if len(self.meta_param) > 0:
            if mtrain_loader != None:
                self.data_loader_mtrain = mtrain_loader
                if isinstance(self.data_loader_mtrain, list):
                    self._data_loader_iter_mtrain = []
                    for x in self.data_loader_mtrain:
                        self._data_loader_iter_mtrain.append(iter(x))
                else:
                    self._data_loader_iter_mtrain = iter(self.data_loader_mtrain)
            else:
                self.data_loader_mtrain = None
                self._data_loader_iter_mtrain = self._data_loader_iter

            if mtest_loader != None:
                self.data_loader_mtest = mtest_loader
                if isinstance(self.data_loader_mtest, list):
                    self._data_loader_iter_mtest = []
                    for x in self.data_loader_mtest:
                        self._data_loader_iter_mtest.append(iter(x))
                else:
                    self._data_loader_iter_mtest = iter(self.data_loader_mtest)
            else:
                self.data_loader_mtest = None
                self._data_loader_iter_mtest = self._data_loader_iter_mtrain
            
            self.initial_requires_grad = self.grad_requires_init(model = self.model)
            find_group = ['layer1_conv_weight', 'layer1_conv_bias',
                          'layer1_bn_(weight|scale)', 'layer1_bn_bias',
                          'classifier_fc_weight', 'classifier_fc_bias',
                          'classifier_norm_(weight|scale)', 'classifier_norm_bias',]
            new_group = list(self.cat_tuples(self.meta_param['meta_compute_layer'], self.meta_param['meta_update_layer']))
            find_group.extend(new_group)
            find_group = list(set(find_group))
            idx_group, dict_group = self.find_selected_optimizer(find_group, self.optimizer_main)
            self.idx_group = idx_group
            self.idx_group_norm = [0, -1]
            self.dict_group = dict_group
            # self.inner_clamp = True
            self.print_flag = False
            # allocate whether each layer applies meta_learning (important!)
            self.all_layers = dict() # find all parameters
            for name, param in self.model.named_parameters():
                if '_mean' in name or '_variance' in name:
                    continue
                name = '.'.join(name.split('.')[:-1])
                raw_name = copy.copy(name)
                name = re.sub(r'\.(?P<n>\d+)\.', lambda x: '['+ x.group('n')+'].', name)  
                if name not in self.all_layers:
                    self.all_layers[name] = dict()
                    self.all_layers[name]['name'] = name
                    self.all_layers[name]['raw_name'] = raw_name       

            for name, val in self.all_layers.items(): # allocate ordered index corresponding to each parameter
                self.all_layers[name]['w_param_idx'] = None
                self.all_layers[name]['b_param_idx'] = None
                self.all_layers[name]['g_param_idx'] = None
                for i, g in enumerate(self.optimizer_main._param_groups):
                    if val['raw_name'] + '.weight' == g['name'] or val['raw_name'] + '.scale' == g['name']:
                        self.all_layers[name]['w_param_idx'] = i
                    elif val['raw_name'] + '.bias' == g['name']:
                        self.all_layers[name]['b_param_idx'] = i
                if self.optimizer_norm != None:
                    for i, g in enumerate(self.optimizer_norm._param_groups):
                        if val['raw_name'] + '.gate' == g['name']:
                            self.all_layers[name]['g_param_idx'] = i
            logger.info('[[Allocate compute_meta_params]]')
            new_object_name_params = 'compute_meta_params'
            new_object_name_gates = 'compute_meta_gates'
            if self.meta_param['meta_all_params']: # allocate all params (not used)
                raise AssertionError
                for name, val in self.all_layers.items():
                    if (val['w_param_idx'] != None) or (val['b_param_idx'] != None):
                        exec('self.model.{}.{} = {}'.format(name, new_object_name_params, True))
                    else:
                        exec('self.model.{}.{} = {}'.format(name, new_object_name_params, False))
                    if (val['g_param_idx'] != None):
                        exec('self.model.{}.{} = {}'.format(name, new_object_name_gates, True))
                    else:
                        exec('self.model.{}.{} = {}'.format(name, new_object_name_gates, False))

            else:
                for name, val in self.all_layers.items():
                    flag_meta_params = False
                    flag_meta_gates = False
                    for update_name in self.meta_param['meta_compute_layer']:
                        if 'gate' in update_name: # about gate parameters
                            split_update_name = update_name.split('_')
                            if len(split_update_name) == 1:  # gates of all bn layers
                                if 'bn' in name:
                                    flag_meta_gates = True # all bn layers
                            else:
                                flag_splits = np.zeros(len(split_update_name))
                                for i, splits in enumerate(split_update_name):
                                    if splits in name:
                                        flag_splits[i] = 1
                                if sum(flag_splits) >= len(split_update_name) - 1:
                                    flag_meta_gates = True
                            if flag_meta_gates:
                                break
                    for update_name in self.meta_param['meta_compute_layer']:
                        if 'gate' not in update_name: # about remaining parameters
                            split_update_name = update_name.split('_')
                            flag_splits = np.zeros(len(split_update_name), dtype=bool)
                            for i, splits in enumerate(split_update_name):
                                if splits in name:
                                    flag_splits[i] = True
                            flag_meta_params = all(flag_splits)
                            if flag_meta_params:
                                break
                    if flag_meta_params: # allocate flag_meta_params in each parameter
                        logger.info('{} is in the {}'.format(update_name, name))
                        exec('self.model.{}.{} = {}'.format(name, new_object_name_params, True))
                    else:
                        exec('self.model.{}.{} = {}'.format(name, new_object_name_params, False))

                    if flag_meta_gates: # allocate flag_meta_gates in each parameter
                        logger.info('{} is in the {}'.format(update_name, name))
                        exec('self.model.{}.{} = {}'.format(name, new_object_name_gates, True))
                    else:
                        exec('self.model.{}.{} = {}'.format(name, new_object_name_gates, False))

            logger.info('[[Exceptions 1]]') # exceptions for resnet50
            name = 'backbone.conv1'; update_name = 'layer0_conv'
            if update_name in self.meta_param['meta_compute_layer']:
                logger.info('{} is in the {}'.format(update_name, name))
                exec('self.model.{}.{} = {}'.format(name, new_object_name_params, True))
            name = 'backbone.conv1'; update_name = 'layer0'
            if update_name in self.meta_param['meta_compute_layer']:
                logger.info('{} is in the {}'.format(update_name, name))
                exec('self.model.{}.{} = {}'.format(name, new_object_name_params, True))
            name = 'backbone.bn1'; update_name = 'layer0_bn'
            if update_name in self.meta_param['meta_compute_layer']:
                logger.info('{} is in the {}'.format(update_name, name))
                exec('self.model.{}.{} = {}'.format(name, new_object_name_params, True))
            name = 'backbone.bn1'; update_name = 'layer0'
            if update_name in self.meta_param['meta_compute_layer']:
                logger.info('{} is in the {}'.format(update_name, name))
                exec('self.model.{}.{} = {}'.format(name, new_object_name_params, True))
                exec('self.model.{}.{} = {}'.format(name, new_object_name_gates, True))
            name = 'backbone.bn1'; update_name = 'layer0_bn_gate'
            if update_name in self.meta_param['meta_compute_layer']:
                logger.info('{} is in the {}'.format(update_name, name))
                exec('self.model.{}.{} = {}'.format(name, new_object_name_gates, True))

            for name, val in self.all_layers.items():
                exec("val['{}'] = self.model.{}.{}".format(new_object_name_params, name, new_object_name_params))
                exec("val['{}'] = self.model.{}.{}".format(new_object_name_gates, name, new_object_name_gates))

            logger.info('[[Summary]]')
            logger.info('Meta compute layer : {}'.format(self.meta_param['meta_compute_layer']))
            for name, val in self.all_layers.items():
                logger.info('Name: {}, meta_param: {}, meta_gate: {}'.format(name, val[new_object_name_params], val[new_object_name_gates]))
        else: # not used
            raise AssertionError
            find_group = ['layer1_conv_weight', 'layer1_conv_bias',
                          'layer1_bn_(weight|scale)', 'layer1_bn_bias',
                          'layer1_bn_mean_(weight|scale)', 'layer1_bn_var_weight',
                          'classifier_fc_weight', 'classifier_fc_bias',
                          'classifier_norm_(weight|scale)', 'classifier_norm_bias',]
            idx_group, dict_group = self.find_selected_optimizer(find_group, self.optimizer_main)
            self.idx_group = idx_group
            self.dict_group = dict_group
            # self.inner_clamp = True
            self.print_flag = False

        # TODO
        #self.checkpointer = Checkpointer(
        #        model,
        #        cfg.OUTPUT_DIR,
        #        save_to_disk=comm.is_main_process(),
        #        optimizer_main=optimizer_main,
        #        scheduler_main=self.scheduler_main,
        #        optimizer_norm=optimizer_norm,
        #        scheduler_norm=self.scheduler_norm,
        #    )
        self.start_iter = 0
        self.max_iter = self.cfg['SOLVER']['MAX_ITER']
        self.register_hooks(self.build_hooks())

    @staticmethod
    def get_cfg(mode, root='./configs'):
        path = os.path.join(root, mode+".yaml")
        with open(path, encoding="UTF-8") as cfg_file:
            cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
        return cfg

    @classmethod
    def build_lr_scheduler(cls, milestones, gamma=0.1,
                            warmup_factor=0.001, warmup_iters=1000,
                            warmup_method='linear', last_epoch=-1, verbose=False):
        return build_lr_scheduler(milestones, gamma, warmup_factor, warmup_iters,\
                                    warmup_method, last_epoch, verbose)

    @classmethod
    def build_optimizer(cls, model, base_lr, lr_scheduler, momentum, flag=None):
        return build_optimizer(model, base_lr, lr_scheduler, momentum, flag=flag)

    @classmethod
    def build_model(cls, num_classes, pretrain=True, pretrain_path='./model_weights/pretrained_resnet50.pdparams'):
        model = Metalearning(num_classes=num_classes)
        if pretrain:
            model.backbone.set_state_dict(paddle.load(pretrain_path))
        return model

    @classmethod
    def build_test_loader(cls, dataset_name, batch_size, num_workers=2, flag_test=True):
        return build_reid_test_loader(dataset_name, batch_size=batch_size, flag_test=flag_test, num_workers=num_workers)        

    @classmethod
    def build_evaluator(cls, num_query, output_dir=None):
        return ReidEvaluator(num_query=num_query)


    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.
        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h != None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def build_hooks(self):
        logger = logging.getLogger(__name__)
        ret = [
           # hooks.IterationTimer(),
            LRScheduler(self.optimizer_main,
                        self.scheduler_main,
                        self.optimizer_norm,
                        self.scheduler_norm),
            #PeriodicEval(period=400,
            #            dataset='Market1501',
            #            model=self.model,
            #            batch_size=128),
            #
            PeriodicEval(period=400,
                        dataset='DukeMTMC',
                        model=self.model,
                        batch_size=128)
        ]
        return ret

    def resume_or_load(self, resume=True):
        # TODO
        pass

    def grad_requires_init(self, model):
        out_requires_grad = dict()
        for name, param in model.named_parameters():
            if '_mean' in name or '_variance' in name:
                continue
            out_requires_grad[name] = not param.stop_gradient
        return out_requires_grad

    #####################################################################
    # about data processing
    #####################################################################
    def cat_tuples(self, tuple1, tuple2):
        list1 = list(tuple1)
        list2 = list(tuple2)
        list_all = list1.copy()
        list_all.extend(list2)
        list_all = list(set(list_all))
        if "" in list_all:
            list_all.remove("")
        list_all = tuple(list_all)
        return list_all

    def auto_scale_hyperparams(self):
        num_images = self.data_loader.batch_sampler.sampler.total_images
        num_classes = self.data_loader.dataset.num_classes
        if self.cfg['META']['DATA']['NAMES'] != "": # meta-learning
            if self.cfg['META']['SOLVER']['INIT']['INNER_LOOP'] == 0:
                iters_per_epoch = num_images // self.cfg['SOLVER']['IMS_PER_BATCH']
            else:
                iters_per_epoch = num_images // (self.cfg['SOLVER']['IMS_PER_BATCH'] * self.cfg['META']['SOLVER']['INIT']['INNER_LOOP'])
        else:
            iters_per_epoch = num_images // self.cfg['SOLVER']['IMS_PER_BATCH']
        self.cfg['SOLVER']['ITERS_PER_EPOCH'] = iters_per_epoch
        self.cfg['MODEL']['HEADS']['NUM_CLASSES'] = num_classes
        self.cfg['SOLVER']['MAX_ITER'] *= iters_per_epoch
        self.cfg['SOLVER']['WARMUP_ITERS'] *= iters_per_epoch
        self.cfg['SOLVER']['FREEZE_ITERS'] *= iters_per_epoch
        self.cfg['SOLVER']['DELAY_ITERS'] *= iters_per_epoch
        for i in range(len(self.cfg['SOLVER']['STEPS'])):
            self.cfg['SOLVER']['STEPS'][i] *= iters_per_epoch
        self.cfg['SOLVER']['SWA']['ITER'] *= iters_per_epoch
        self.cfg['SOLVER']['SWA']['PERIOD'] *= iters_per_epoch
        self.cfg['SOLVER']['CHECKPOINT_PERIOD'] *= iters_per_epoch
        num_mod = (self.cfg['SOLVER']['WRITE_PERIOD'] - self.cfg['TEST']['EVAL_PERIOD'] * iters_per_epoch) % self.cfg['SOLVER']['WRITE_PERIOD']
        self.cfg['TEST']['EVAL_PERIOD'] = self.cfg['TEST']['EVAL_PERIOD'] * iters_per_epoch + num_mod
        if self.cfg['SOLVER']['CHECKPOINT_SAME_AS_EVAL']:
            self.cfg['SOLVER']['CHECKPOINT_PERIOD'] = self.cfg['TEST']['EVAL_PERIOD']
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to num_classes={self.cfg['MODEL']['HEADS']['NUM_CLASSES']}, "
            f"max_Iter={self.cfg['SOLVER']['MAX_ITER']}, wamrup_Iter={self.cfg['SOLVER']['WARMUP_ITERS']}, "
            f"freeze_Iter={self.cfg['SOLVER']['FREEZE_ITERS']}, delay_Iter={self.cfg['SOLVER']['DELAY_ITERS']}, "
            f"step_Iter={self.cfg['SOLVER']['STEPS']}, ckpt_Iter={self.cfg['SOLVER']['CHECKPOINT_PERIOD']}, "
            f"eval_Iter={self.cfg['TEST']['EVAL_PERIOD']}."
        )

    
    def find_selected_optimizer(self, find_group, optimizer):
        # find parameter, lr, required_grad, shape
        logger.info('Storage parameter, lr, requires_grad, shape! in {}'.format(find_group))
        idx_group = []
        dict_group = dict()
        for j in range(len(find_group)):
            idx_local = []
            for i, x in enumerate(optimizer._param_groups):
                split_find_group = find_group[j].split('_')
                flag_splits = [len(re.compile(split).findall(x['name'])) > 0 for split in split_find_group]
                flag_target = all(flag_splits)
                if flag_target:
                    dict_group[x['name']] = i
                    idx_local.append(i)
            if len(idx_local) > 0:
                logger.info('Find {} in {}'.format(find_group[j], optimizer._param_groups[idx_local[0]]['name']))
                idx_group.append(idx_local[0])
            else:
                logger.info('error in find_group')
        idx_group = list(set(idx_group))
        return idx_group, dict_group



    def train(self):
        start_iter = self.start_iter
        max_iter = self.max_iter
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter
        self.global_meta_cnt = 0

        with EventStorage(start_iter) as self.storage:
            self.before_train() # check hooks.py, engine/defaults.py
            for self.iter in range(start_iter, max_iter):
                #print("\niter:", self.iter)
                self.before_step()
                if self.cfg['META']['DATA']['NAMES'] == '': # general learning (not meta-learning)
                    raise AssertionError
                    self.run_step() # not used
                else: # our model (MAML-based)
                    self.cnt = 0
                    self.data_time_all = 0.0
                    self.metrics_dict = dict()
                    if self.iter == 0:
                        max_init = self.meta_param['iter_init_inner_first']
                    else:
                        max_init = self.meta_param['iter_init_inner']
                    while (self.cnt < max_init):
                        self.run_step_meta_learning1() # update base model
                        self.cnt += 1

                    self.cnt = 0
                    while (self.cnt < self.meta_param['iter_init_outer']):
                        self.run_step_meta_learning2() # update balancing parameters (meta-learning)
                        self.cnt += 1
                        self.global_meta_cnt += 1
                    # print(self.iter)
                self.after_step()
            self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        # this guarantees, that in each hook's after_step, storage.iter == trainer.iter
        self.storage.step()

    #####################################################################
    # general learning (not meta-learning, not our model)
    #####################################################################
    def run_step(self):
        pass

    #####################################################################
    # forward
    #####################################################################
    def basic_forward(self, data, model, opt = None):
        #model = model.module if isinstance(model, DistributedDataParallel) else model
        if data != None:
            with paddle.amp.auto_cast(enable=self.scaler != None):
                outs = model(data, opt)
                loss_dict = model.losses(outs, opt)
                losses = sum(loss_dict.values())
                self.loss_file.write('\t'.join([str(self.iter)]+ [str(float(v.numpy())) for v in loss_dict.values()]) + '\n')
            self._detect_anomaly(losses, loss_dict)
        else:
            losses = None
            loss_dict = dict()
        print(self.iter, {k: float(v.numpy()) for k, v in loss_dict.items()})
        return losses, loss_dict

    #####################################################################
    # backward
    #####################################################################
    def basic_backward(self, losses, optimizer, retain_graph = False):
        if (losses is not None) and (optimizer is not None):
            optimizer.clear_grad()
            if self.scaler == None: # no AMP
                losses.backward(retain_graph = retain_graph)
                optimizer.step()
            else: # with AMP(automatic mixed precision)
                self.scaler.scale(losses).backward(retain_graph = retain_graph)
                # self.scaler.unscale_(optimizer)
                # paddle.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            for i in range(len(self.bin_gates)):
                self.bin_gates[i].set_value(self.bin_gates[i].clip(0, 1))

    #####################################################################
    # base model updates (not meta-learning)
    #####################################################################
    def run_step_meta_learning1(self): #TODO

        # initial setting
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        # 1) Meta-initialization
        name_loss = '1)'
        opt = self.opt_setting('basic')
        # self.grad_requires_check(self.model)

        # self.grad_setting('basic')  # Freeze "meta_update_layer"
        if self.cnt == 0:
            self.print_selected_optimizer('0) start', self.idx_group, self.optimizer_main, self.meta_param['detail_mode'])
            self.print_selected_optimizer('0) start', self.idx_group_norm, self.optimizer_norm, self.meta_param['detail_mode'])

        data, data_time = self.get_data(self._data_loader_iter, list_sample = None)
        self.data_time_all += data_time
        #!opt['domains'] = data['others']['domains']
        opt['domains'] = data['domains']
        with open(os.path.join(self.output_dir, 'train_loss.csv'), 'a+') as self.loss_file:
            losses, loss_dict = self.basic_forward(data, self.model, opt) # forward
        
        if self.meta_param['meta_all_params']:
            self.basic_backward(losses, self.optimizer_norm, retain_graph = True) #
        self.basic_backward(losses, self.optimizer_main) # backward

        if self.cnt == 0:
            for name, val in loss_dict.items():
                t = name_loss+name
                self.metrics_dict[t] = self.metrics_dict.get(t, 0) + val
            self.print_selected_optimizer('1) after meta-init', self.idx_group, self.optimizer_main, self.meta_param['detail_mode'])
            self.print_selected_optimizer('1) after meta-init', self.idx_group_norm, self.optimizer_norm, self.meta_param['detail_mode'])

        # if self.meta_param['flag_manual_zero_grad'] != 'hold':
        #     self.manual_zero_grad(self.model)
        #     self.optimizer_main.clear_grad()
        # if self.meta_param['flag_manual_memory_empty']:
        # paddle.cuda.empty_cache()
        if self.iter == 0:
            if self.optimizer_norm != None:
                self.optimizer_norm.clear_grad()
        #if self.meta_param['sync']: paddle.cuda.synchronize()

    #####################################################################
    # meta-learning (update balancing parameters)
    #####################################################################
    def run_step_meta_learning2(self):

        # start = time.perf_counter()
        # Meta-learning
        if self.meta_param['main_zero_grad']: self.optimizer_main.clear_grad()

        if self.cnt == 0:
            self.print_selected_optimizer('2) before meta-train', self.idx_group, self.optimizer_main, self.meta_param['detail_mode'])
            self.print_selected_optimizer('2) before meta-train', self.idx_group_norm, self.optimizer_norm, self.meta_param['detail_mode'])

        mtrain_losses = []
        mtest_losses = []
        cnt_local = 0
        while(cnt_local < self.meta_param['iter_mtrain']):
            if self.meta_param['shuffle_domain'] or \
                    (not self.meta_param['shuffle_domain'] and cnt_local == 0):
                list_all = np.random.permutation(self.meta_param['num_domain'])
                list_mtrain = list(list_all[0:self.meta_param['num_mtrain']])
                list_mtest = list(list_all[self.meta_param['num_mtrain']:
                                           self.meta_param['num_mtrain'] + self.meta_param['num_mtest']])

            # 2) Meta-train
            cnt_local += 1
            name_loss_mtrain = '2)'
            opt = self.opt_setting('mtrain')

            if self.meta_param['one_loss_for_iter']: # not used
                raise AssertionError
                num_losses = len(opt['loss'])
                num_rem = self.global_meta_cnt % num_losses
                if self.meta_param['one_loss_order'] == 'forward':
                    num_case = num_rem
                elif self.meta_param['one_loss_order'] == 'backward':
                    num_case = num_losses - num_rem - 1
                elif self.meta_param['one_loss_order'] == 'random':
                    num_case = np.random.permutation(num_losses)[0]
                opt['loss'] = tuple([opt['loss'][num_case]])

            # data loader
            if self.data_loader_mtest == None:
                if self.meta_param['whole']:
                    data, data_time = self.get_data(self._data_loader_iter_mtrain, list_sample = list_mtrain, opt = 'all')
                    self.data_time_all += data_time
                    data_mtrain = data[0]
                    data_mtest = data[1]
                else:   # not used
                    raise AssertionError
                    data_mtrain, data_time = self.get_data(self._data_loader_iter_mtrain, list_sample=list_mtrain)
                    self.data_time_all += data_time
                    data_mtest, data_time = self.get_data(self._data_loader_iter_mtrain, list_sample=list_mtest)
                    self.data_time_all += data_time
            else: # not used 
                raise AssertionError
                if self.meta_param['synth_data'] != 'none' and self.meta_param['synth_method'] != 'none':

                    if self.meta_param['synth_method'] == 'real': # mtrain (real) -> mtest (fake)
                        data_mtrain, data_time = self.get_data(self._data_loader_iter_mtrain, list_sample = list_mtrain)
                        self.data_time_all += data_time
                        data_mtest, data_time = self.get_data(self._data_loader_iter_mtest, list_sample = list_mtest)
                        self.data_time_all += data_time
                    elif self.meta_param['synth_method'] == 'real_all':  # mtrain (real) -> mtest (fake)
                        data_mtrain, data_time = self.get_data(self._data_loader_iter_mtrain)
                        self.data_time_all += data_time
                        data_mtest, data_time = self.get_data(self._data_loader_iter_mtest)
                        self.data_time_all += data_time
                    elif self.meta_param['synth_method'] == 'fake': # mtrain (fake) -> mtest (fake)
                        data_mtrain, data_time = self.get_data(self._data_loader_iter_mtest, list_sample = list_mtrain)
                        self.data_time_all += data_time
                        data_mtest, data_time = self.get_data(self._data_loader_iter_mtrain, list_sample = list_mtest)
                        self.data_time_all += data_time
                    elif self.meta_param['synth_method'] == 'fake_all':  # mtrain (real) -> mtest (fake)
                        data_mtrain, data_time = self.get_data(self._data_loader_iter_mtest)
                        self.data_time_all += data_time
                        data_mtest, data_time = self.get_data(self._data_loader_iter_mtrain)
                    elif self.meta_param['synth_method'] == 'alter':
                        if self.iter % 2 == 0:
                            data_mtrain, data_time = self.get_data(self._data_loader_iter_mtrain, list_sample=list_mtrain)
                            self.data_time_all += data_time
                            data_mtest, data_time = self.get_data(self._data_loader_iter_mtest, list_sample=list_mtest)
                            self.data_time_all += data_time
                        else:
                            data_mtrain, data_time = self.get_data(self._data_loader_iter_mtest, list_sample = list_mtrain)
                            self.data_time_all += data_time
                            data_mtest, data_time = self.get_data(self._data_loader_iter_mtrain, list_sample = list_mtest)
                            self.data_time_all += data_time
                    elif self.meta_param['synth_method'] == 'alter_all':
                        if self.iter % 2 == 0:
                            data_mtrain, data_time = self.get_data(self._data_loader_iter_mtrain)
                            self.data_time_all += data_time
                            data_mtest, data_time = self.get_data(self._data_loader_iter_mtest)
                            self.data_time_all += data_time
                        else:
                            data_mtrain, data_time = self.get_data(self._data_loader_iter_mtest)
                            self.data_time_all += data_time
                            data_mtest, data_time = self.get_data(self._data_loader_iter_mtrain)
                            self.data_time_all += data_time
                    elif self.meta_param['synth_method'] == 'both':
                        data_real, data_time = self.get_data(self._data_loader_iter_mtrain, list_sample = list_mtrain, opt = 'all')
                        self.data_time_all += data_time
                        data_fake, data_time = self.get_data(self._data_loader_iter_mtest, list_sample = list_mtrain, opt = 'all')
                        self.data_time_all += data_time
                        data_real_mtrain = data_real[0]
                        data_fake_mtrain = data_fake[0]
                        data_real_mtest = data_real[1]
                        data_fake_mtest = data_fake[1]
                        data_mtrain = self.cat_data(data_real_mtrain, data_fake_mtrain)
                        data_mtest = self.cat_data(data_real_mtest, data_fake_mtest)
                else:
                    data_mtrain, data_time = self.get_data(self._data_loader_iter_mtrain, list_sample = list_mtrain)
                    self.data_time_all += data_time
                    data_mtest, data_time = self.get_data(self._data_loader_iter_mtest, list_sample = list_mtest)
                    self.data_time_all += data_time

            # import matplotlib.pyplot as plt
            # plt.imshow(data_mtrain['images'][0].permute(1, 2, 0)/255)
            # plt.show()

            if (self.meta_param['synth_grad'] == 'none') or (self.meta_param['synth_grad'] == 'reverse'):
                
                '''?
                if self.meta_param['freeze_gradient_meta']:
                    self.grad_setting('mtrain_both')
                '''
                opt['domains'] = data_mtrain['domains']
                with open(os.path.join(self.output_dir, 'mtrain_loss.csv'), 'a+') as self.loss_file:
                    losses, loss_dict = self.basic_forward(data_mtrain, self.model, opt) # forward
                mtrain_losses.append(losses)

                if self.cnt == 0:
                    for name, val in loss_dict.items():
                        t = name_loss_mtrain + name
                        self.metrics_dict[t] = self.metrics_dict.get(t, 0) + val
                        #self.metrics_dict[t] = self.metrics_dict[t] + val if t in self.metrics_dict.keys() else val
            else:
                losses = []


            # 3) Meta-test
            name_loss_mtest = '3)'
            # self.grad_setting('mtrain_single') # melt only meta_compute parameters
            opt = self.opt_setting('mtest', losses) # auto_grad based on requires_grad of model

            print_grad_mean_list = list()
            print_grad_prob_list = list()
            if self.meta_param['print_grad'] and len(self.bin_names) > 0 \
                    and (self.iter + 1) % (self.cfg['SOLVER']['WRITE_PERIOD_BIN']) == 0:
                if self.cnt == 0:
                    with paddle.no_grad():
                        if len(opt['grad_params'])>0:
                            grad_cnt = 0
                            for grad_values in opt['grad_params']:
                                if 'gate' in opt['grad_name'][grad_cnt]:
                                    print_grad_mean_list.append(np.mean(grad_values.tolist()))
                                    print_grad_prob_list.append(np.mean([1.0 if k > 0 else 0.0 for k in grad_values.tolist()]))
                                grad_cnt += 1


            # self.grad_setting('mtrain_both') # melt both meta_compute and meta_update parameters
            opt['domains'] = data_mtest['domains']
            with open(os.path.join(self.output_dir, 'mtest_loss.csv'), 'a+') as self.loss_file:
                losses, loss_dict = self.basic_forward(data_mtest, self.model, opt) # forward

            mtest_losses.append(losses)
            if self.cnt == 0:
                for name, val in loss_dict.items():
                    t = name_loss_mtest + name
                    self.metrics_dict[t] = self.metrics_dict[t] + val if t in self.metrics_dict.keys() else val


        if self.meta_param['iter_init_outer'] == 1:
            if len(mtrain_losses) > 0:
                mtrain_losses = mtrain_losses[0]
            if len(mtest_losses) > 0:
                mtest_losses = mtest_losses[0]
        else:
            if len(mtrain_losses) > 0:
                mtrain_losses = paddle.sum(paddle.stack(mtrain_losses))
            if len(mtest_losses) > 0:
                mtest_losses = paddle.sum(paddle.stack(mtest_losses))

        if self.meta_param['loss_combined']:
            total_losses = self.meta_param['loss_weight'] * mtrain_losses + mtest_losses
        else:
            total_losses = mtest_losses
        total_losses /= float(self.meta_param['iter_mtrain'])

        if self.meta_param['meta_all_params']:
            self.basic_backward(total_losses, self.optimizer_main, retain_graph = True) #
        self.basic_backward(total_losses, self.optimizer_norm) # backward

        # if self.meta_param['flag_manual_zero_grad'] != 'hold':
        #     self.manual_zero_grad(self.model)
        #     self.optimizer_norm.clear_grad()
        # if self.meta_param['flag_manual_memory_empty']:
        #     paddle.cuda.empty_cache()

        #?if self.meta_param['sync']: paddle.cuda.synchronize()

        if self.cnt == 0:
            self.print_selected_optimizer('2) after meta-learning', self.idx_group, self.optimizer_main, self.meta_param['detail_mode'])
            self.print_selected_optimizer('2) after meta-learning', self.idx_group_norm, self.optimizer_norm, self.meta_param['detail_mode'])

        self.optimizer_main.clear_grad()
        if self.optimizer_norm != None:
            self.optimizer_norm.clear_grad()

        #if self.meta_param['freeze_gradient_meta']:
        #    self.grad_requires_recover(model=self.model, ori_grad=self.initial_requires_grad)

        if self.cnt == 0:
            self.metrics_dict["data_time"] = self.data_time_all
            self._write_metrics(self.metrics_dict)
        '''TODO
        with paddle.no_grad(): # for save balancing parameters
            if self.cnt == 0:
                if len(self.bin_names) > 0 and (self.iter + 1) % (self.cfg['SOLVER']['WRITE_PERIOD_BIN']) == 0:
                    start = time.perf_counter()
                    all_gate_dict = dict()
                    cnt_print = 0
                    for j in range(len(self.bin_names)):
                        name = '_'.join(self.bin_names[j].split('.')[1:]).\
                            replace('bn', 'b').replace('gate','g').replace('layer', 'l').replace('conv','c')
                        val_mean = paddle.mean(self.bin_gates[j].data).tolist()
                        val_std = paddle.std(self.bin_gates[j].data).tolist()
                        val_hist = paddle.histc(self.bin_gates[j].data, bins=20, min=0.0, max=1.0).int()
                        all_gate_dict[name + '_mean']= val_mean
                        all_gate_dict[name + '_std']= val_std
                        for x in paddle.nonzero(val_hist.data):
                            all_gate_dict[name + '_hist' + str(x[0].tolist())] = val_hist[x[0]].tolist()
                        # all_gate_dict['hist_' + name]= str(val_hist.tolist()).replace(' ','')
                        if self.meta_param['print_grad']:
                            if len(print_grad_mean_list) > 0:
                                all_gate_dict[name + '_grad_average'] = print_grad_mean_list[cnt_print]
                            if len(print_grad_prob_list) > 0:
                                all_gate_dict[name + '_grad_prob'] = print_grad_prob_list[cnt_print]
                        cnt_print += 1

                    self.storage.put_scalars(**all_gate_dict, smoothing_hint=False)
        '''

                # print(time.perf_counter() - start)


        # print("Processing time: {}".format(time.perf_counter() - start))

    #####################################################################
    # load data
    #####################################################################
    def get_data(self, data_loader_iter, list_sample = None, opt = None):
        start = time.perf_counter()
        if data_loader_iter != None:
            data = None
            while(data == None):
                if isinstance(data_loader_iter, list):
                    if list_sample == None:
                        data = self.data_aggregation(dataloader = data_loader_iter, list_num = [x for x in range(len(data_loader_iter))])
                    else:
                        data = self.data_aggregation(dataloader = data_loader_iter, list_num = [x for x in list_sample])

                else:
                    data = next(data_loader_iter)
                    if list_sample != None:
                        domain_idx = data['domains']
                        cnt = 0
                        for sample in list_sample:
                            if cnt == 0:
                                t_logical_domain = domain_idx == sample
                            else:
                                #t_logical_domain += domain_idx == sample
                                t_logical_domain = paddle.logical_or(t_logical_domain, domain_idx == sample)
                            cnt += 1

                        # data1
                        #if int(sum(t_logical_domain)) == 0:
                        if not any(t_logical_domain):
                            data = None
                            logger.info('No data including list_domain')
                        else:
                            data1 = dict()
                            for name, value in data.items():
                                if paddle.is_tensor(value):
                                    data1[name] = data[name][t_logical_domain]
                                elif isinstance(value, dict):
                                    data1[name] = dict()
                                    for name_local, value_local in value.items():
                                        if paddle.is_tensor(value_local):
                                            data1[name][name_local] = data[name][name_local][t_logical_domain]
                                elif isinstance(value, list):
                                    data1[name] = [x for i, x in enumerate(data[name]) if t_logical_domain[i]]

                        # data2 (if opt == 'all')
                        if opt == 'all':
                            t_logical_domain_reversed = t_logical_domain == False
                            if not any(t_logical_domain_reversed):
                            #if int(sum(t_logical_domain_reversed)) == 0:
                                data2 = None
                                logger.info('No data including list_domain')
                            else:
                                data2 = dict()
                                for name, value in data.items():
                                    if paddle.is_tensor(value):
                                        data2[name] = data[name][t_logical_domain_reversed]
                                    elif isinstance(value, dict):
                                        data2[name] = dict()
                                        for name_local, value_local in value.items():
                                            if paddle.is_tensor(value_local):
                                                data2[name][name_local] = data[name][name_local][t_logical_domain_reversed]
                                    elif isinstance(value, list):
                                        data2[name] = [x for i, x in enumerate(data[name]) if t_logical_domain_reversed[i]]
                            data = [data1, data2]
                        else:
                            data = data1
        else:
            data = None
            logger.info('No data including list_domain')
        data_time = time.perf_counter() - start
                # sample data
        return data, data_time

    #####################################################################
    # about data processing
    #####################################################################
    def data_aggregation(self, dataloader, list_num):
        data = None
        for cnt, list_idx in enumerate(list_num):
            if cnt == 0:
                data = next(dataloader[list_idx])
            else:
                for name, value in next(dataloader[list_idx]).items():
                    if paddle.is_tensor(value):
                        data[name] = paddle.concat((data[name], value), 0)
                    elif isinstance(value, dict):
                        for name_local, value_local in value.items():
                            if paddle.is_tensor(value_local):
                                data[name][name_local] = paddle.concat((data[name][name_local], value_local), 0)
                    elif isinstance(value, list):
                        data[name].extend(value)
        return data

    #####################################################################
    # about data processing
    #####################################################################
    def cat_data(self, data1, data2):
        for name, value in data2.items():
            if paddle.is_tensor(value):
                data1[name] = paddle.concat((data1[name], value), 0)
            elif isinstance(value, dict):
                for name_local, value_local in value.items():
                    if paddle.is_tensor(value_local):
                        data1[name][name_local] = paddle.concat(
                            (data1[name][name_local], value_local), 0)
            elif isinstance(value, list):
                data1[name].extend(value)
        return data1
    
    def _write_metrics(self, m):
        # TODO
        pass

    #####################################################################
    # detect anomaly
    #####################################################################
    def _detect_anomaly(self, losses, loss_dict):
        if not paddle.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    #####################################################################
    # delete gradient manually
    #####################################################################
    def manual_zero_grad(self, model):
        if self.meta_param['flag_manual_zero_grad'] == 'delete':
            for name, param in model.named_parameters():  # parameter grad_zero
                if param.grad != None:
                    param.grad = None
        elif self.meta_param['flag_manual_zero_grad'] == 'zero':
            for name, param in model.named_parameters():  # parameter grad_zero
                if param.grad != None:
                    if paddle.sum(param.grad) > 0:
                        param.grad.zero_()
        # return model
    
    @staticmethod
    def get_curr_lr(optimizer, param_idx):
        if isinstance(optimizer._learning_rate, paddle.optimizer.lr.LRScheduler):
            lr = optimizer._learning_rate.last_lr * optimizer._param_groups[param_idx]['learning_rate']
        else:
            lr = optimizer._learning_rate * optimizer._param_groups[param_idx]['learning_rate']
        return lr

    @staticmethod
    def get_curr_scale(scaler):
        if scaler._enable:
            return scaler._scale.item()
        else:
            return 1.0

    #####################################################################
    # set options (basic, mtrain, mtest) important!
    #####################################################################
    def opt_setting(self, flag, losses = None):
        if flag == 'basic':
            opt = {}
            opt['param_update'] = False
            opt['loss'] = self.cfg['MODEL']['LOSSES']['NAME']
            try:
                opt['type_running_stats'] = self.meta_param['type_running_stats_init']
            except:
                opt['type_running_stats'] = 'general'
            opt['each_domain'] = self.cfg['MODEL']['NORM']['EACH_DOMAIN_BASIC']
        elif flag == 'mtrain':
            opt = {}
            opt['param_update'] = False
            opt['loss'] = self.meta_param['loss_name_mtrain']
            opt['type_running_stats'] = self.meta_param['type_running_stats_mtrain']
            opt['each_domain'] = self.cfg['MODEL']['NORM']['EACH_DOMAIN_MTRAIN']
        elif flag == 'mtest':
            opt = {}
            opt['param_update'] = True
            opt['loss'] = self.meta_param['loss_name_mtest']
            opt['use_second_order'] = self.meta_param['use_second_order']
            opt['stop_gradient'] = self.meta_param['stop_gradient']
            opt['allow_unused'] = self.meta_param['allow_unused']
            # opt['zero_grad'] = self.meta_param['zero_grad']
            opt['type_running_stats'] = self.meta_param['type_running_stats_mtest']
            opt['inner_clamp'] = self.meta_param['inner_clamp']
            opt['each_domain'] = self.cfg['MODEL']['NORM']['EACH_DOMAIN_MTEST']

            # self.meta_param['update_cyclic_ratio']
            # self.meta_param['update_cyclic_period']
            if self.meta_param['update_ratio'] == 0.0:
                if self.meta_param['update_cyclic_new']: # cyclic update
                    self.cyclic_scheduler.step()
                    # ! meta_ratio = self.cyclic_optimizer.param_groups[0]['lr']
                    meta_ratio = self.cyclic_scheduler.get_lr()
                else: # not used (old version)
                    raise AssertionError
                    one_period = self.meta_param['iters_per_epoch'] / self.meta_param['update_cyclic_period']
                    b = math.log10(self.meta_param['update_cyclic_ratio'])
                    a = b / (one_period/4.0*1.0)
                    # for i in range(self.meta_param['iters_per_epoch']):
                    rem_val = self.iter % one_period
                    if  rem_val < (one_period/4.0*1.0): # 1st period
                        meta_ratio = a * rem_val # y = ax
                    elif  rem_val < (one_period/4.0*2.0): # 2nd period
                        rem_val -= one_period/4.0*1.0
                        meta_ratio = b - a * rem_val # y = b - ax
                    elif  rem_val < (one_period/4.0*3.0): # 3rd period
                        rem_val -= one_period/4.0*2.0
                        meta_ratio = - a * rem_val # y = - ax
                    else: # 4th period
                        rem_val -= one_period/4.0*3.0
                        meta_ratio = - b + a * rem_val # y = -b + ax
                    meta_ratio = pow(10, meta_ratio)
            else:
                raise AssertionError
                meta_ratio = self.meta_param['update_ratio']

            # allocate stepsize
            for name, val in self.all_layers.items(): # compute stepsize
                if self.all_layers[name]['w_param_idx'] != None:
                    self.all_layers[name]['w_step_size'] = self.get_curr_lr(self.optimizer_main, self.all_layers[name]['w_param_idx']) * meta_ratio
                else:
                    self.all_layers[name]['b_step_size'] = None

                if self.all_layers[name]['b_param_idx'] != None:
                    self.all_layers[name]['b_step_size'] = self.get_curr_lr(self.optimizer_main, self.all_layers[name]['b_param_idx']) * meta_ratio
                else:
                    self.all_layers[name]['b_step_size'] = None

                if self.all_layers[name]['g_param_idx'] != None:
                    self.all_layers[name]['g_step_size'] = self.get_curr_lr(self.optimizer_norm, self.all_layers[name]['g_param_idx']) * meta_ratio
                else:
                    self.all_layers[name]['g_step_size'] = None

            for name, val in self.all_layers.items(): # allocate stepsize
                if val['compute_meta_params']:
                    exec('self.model.{}.{} = {}'.format(name, 'w_step_size', val['w_step_size']))
                    exec('self.model.{}.{} = {}'.format(name, 'b_step_size', val['b_step_size']))
                if val['compute_meta_gates']:
                    exec('self.model.{}.{} = {}'.format(name, 'g_step_size', val['g_step_size']))

            opt['auto_grad_outside'] = self.meta_param['auto_grad_outside']

            # inner
            if opt['auto_grad_outside']: # compute gradient using meta-train loss
                # outer
                names_weights_copy = dict()
                if self.meta_param['momentum_init_grad'] > 0.0:
                    raise AssertionError
                    names_grads_copy = list()
                for name, param in self.model.named_parameters():
                    if '_mean' in name or '_variance' in name:
                        continue
                    if self.meta_param['meta_all_params']: # not used
                        raise AssertionError
                        if not param.stop_gradient:
                            names_weights_copy['self.model.' + name] = param
                            if self.meta_param['momentum_init_grad'] > 0.0:
                                names_grads_copy.append(copy.deepcopy(param.grad))
                        elif self.iter == 0:
                                logger.info("[{}] This parameter does have requires_grad".format(name))

                    else:
                        for compute_name in list(self.meta_param['meta_compute_layer']):
                            split_compute_name = compute_name.split('_')
                            if 'gate' in name:
                                if 'gate' not in split_compute_name:
                                    continue
                            else:  # 'weight' / 'bais
                                if 'gate' in split_compute_name:
                                    continue
                            flag_splits = [word in name for word in split_compute_name]
                            flag_target = all(flag_splits)
                            if flag_target:
                                if not param.stop_gradient:
                                    names_weights_copy['self.model.' + name] = param
                                    #if self.meta_param['momentum_init_grad'] > 0.0:
                                    #    names_grads_copy.append(copy.deepcopy(param.grad))
                                else:
                                    if self.iter == 0:
                                        logger.info("[{}] This parameter does have requires_grad".format(name))

                if self.meta_param['norm_zero_grad'] and self.optimizer_norm != None:
                    self.optimizer_norm.clear_grad()

                opt['grad_name'] = list(names_weights_copy.keys())
                if (self.meta_param['synth_grad'] == 'none') or (self.meta_param['synth_grad'] == 'reverse'):

                    if self.scaler != None:
                        if self.cfg['META']['SOLVER']['EARLY_SCALE']:
                            inv_scale = 1. / self.get_curr_scale(self.scaler)
                            losses *= inv_scale

                    if self.scaler != None:
                        grad_params = paddle.grad(
                            self.scaler.scale(losses), list(names_weights_copy.values()),
                            create_graph=opt['use_second_order'], allow_unused=opt['allow_unused'])
                    else: # not used
                        raise AssertionError
                        grad_params = paddle.grad(
                            losses, list(names_weights_copy.values()),
                            create_graph=opt['use_second_order'], allow_unused=opt['allow_unused'])
                    
                    if self.meta_param['synth_grad'] == 'reverse': # not used
                        raise AssertionError
                        for val in grad_params:
                            val *= -1.0
                else:  # not used
                    raise AssertionError
                    if self.meta_param['synth_grad'] == 'constant':
                        grad_params = list()
                        for val in names_weights_copy.values():
                            synth_grad = copy.deepcopy(val)
                            synth_grad[:] = self.meta_param['constant_grad']
                            grad_params.append(synth_grad)
                        grad_params = tuple(grad_params)
                    elif self.meta_param['synth_grad'] == 'random':
                        grad_params = list()
                        for val in names_weights_copy.values():
                            synth_grad = copy.deepcopy(val)
                            synth_grad[:] = paddle.randn(val.shape) * self.meta_param['random_scale_grad']
                            grad_params.append(synth_grad)
                        grad_params = tuple(grad_params)

                if opt['stop_gradient']:
                    grad_params = list(grad_params)
                    for i in range(len(grad_params)):
                        if grad_params[i] is not None:
                            grad_params[i] = paddle.to_tensor(grad_params[i], stop_gradient=True)
                        else:
                            if self.iter == 0:
                                logger.info("[{}th grad] This parameter does have gradient".format(i))
                    grad_params = tuple(grad_params)

                if self.meta_param['momentum_init_grad'] > 0.0:
                    raise AssertionError
                    grad_params = list(grad_params)
                    for i in range(len(grad_params)):
                        if grad_params[i] != None:
                            grad_params[i] = self.meta_param['momentum_init_grad'] * names_grads_copy[i].data + \
                                             (1 - self.meta_param['momentum_init_grad']) * grad_params[i].data
                        else:
                            if self.iter == 0:
                                logger.info("[{}th grad] This parameter does have gradient".format(i))
                    grad_params = tuple(grad_params)

                if self.scaler is not None:
                    if not self.cfg['META']['SOLVER']['EARLY_SCALE']: # not used
                        raise AssertionError
                        inv_scale = 1. / self.get_curr_scale(self.scaler)
                        opt['grad_params'] = [p * inv_scale if p is not None else None for p in grad_params ]
                    else:
                        opt['grad_params'] = [p if p is not None else None for p in grad_params ]
                else: # not used
                    raise AssertionError
                    opt['grad_params'] = [p if p is not None else None for p in grad_params ]
                opt['meta_loss'] = None
            else:
                opt['meta_loss'] = losses
        return opt

    #####################################################################
    # for logger
    #####################################################################
    def print_selected_optimizer(self, txt, idx_group, optimizer, detail_mode):
        return
        try:
            num_period = self.meta_param['write_period_param']
        except:
            num_period = 100
        if detail_mode and (self.iter <= 5 or self.iter % num_period == 0):
            if optimizer != None:
                num_float = 8
                for x in idx_group:
                    t_name = optimizer._param_groups[x]['name']
                    t_param = optimizer._param_groups[x]['params'][0].view(-1)[0]
                    t_lr = optimizer._param_groups[x]['lr']
                    t_grad = optimizer._param_groups[x]['params'][0].stop_gradient
                    t_grad_val = optimizer._param_groups[x]['params'][0].grad
                    if t_grad_val != None:
                        if paddle.sum(t_grad_val) == 0:
                            t_grad_val = 'Zero'
                        else:
                            t_grad_val = 'Exist'
                    # t_shape = optimizer.param_groups[x]['params'][0].shape
                    for name, param in self.model.named_parameters():
                        if name == t_name:
                            m_param = param.view(-1)[0]
                            m_grad = param.stop_gradient
                            m_grad_val = param.grad
                            if m_grad_val != None:
                                if paddle.sum(m_grad_val) == 0:
                                    m_grad_val = 'Zero'
                                else:
                                    m_grad_val = 'Exist'
                            val = paddle.sum(param - optimizer.param_groups[x]['params'][0])
                            break

    def test(self, dataset_name, model, batch_size, num_workers=2, evaluator=None):
        test_loader, num_query= build_reid_test_loader(dataset_name, batch_size, num_workers=num_workers, flag_test=True)
        if evaluator is None:
            evaluator = self.build_evaluator(num_query)
        metric = inference_on_dataset(model, test_loader, evaluator)
        with open(os.path.join(self.output_dir, 'metric'+ dataset_name +'.csv'), 'a+') as metric_file:
            metric_file.write('\t'.join([str(self.iter)]+ [str(v) for v in metric.values()]) + '\n')
        print('*'*50)
        print(dataset_name, self.iter)
        print(metric)
        print('*'*50)
        


