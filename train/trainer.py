from .base import Trainer
import numpy as np
import paddle
import os 


class MResNetTrainer(Trainer):
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
            opt = self.opt_setting('mtest') # auto_grad based on requires_grad of model
            if losses:
                losses.backward()
            '''
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

            '''
            # self.grad_setting('mtrain_both') # melt both meta_compute and meta_update parameters
            opt['domains'] = data_mtest['domains']
            with open(os.path.join(self.output_dir, 'mtest_loss.csv'), 'a+') as self.loss_file:
                losses, loss_dict = self.basic_forward(data_mtest, self.model, opt) # forward
            self.optimizer_norm.clear_grad()

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
            raise AssertionError
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

    def opt_setting(self, flag):
        if flag == 'basic':
            opt = {}
            opt['param_update'] = False
            opt['loss'] = self.cfg['MODEL']['LOSSES']['NAME']
            opt['type_running_stats'] = self.meta_param.get('type_running_stats_init', 'general') # 'general
            opt['each_domain'] = self.cfg['MODEL']['NORM']['EACH_DOMAIN_BASIC']
        elif flag == 'mtrain':
            opt = {}
            opt['param_update'] = False
            opt['loss'] = self.meta_param['loss_name_mtrain']
            opt['type_running_stats'] = self.meta_param['type_running_stats_mtrain']    # hold
            opt['each_domain'] = self.cfg['MODEL']['NORM']['EACH_DOMAIN_MTRAIN']
        elif flag == 'mtest':
            opt = {}
            opt['param_update'] = True
            opt['loss'] = self.meta_param['loss_name_mtest']
            opt['type_running_stats'] = self.meta_param['type_running_stats_mtest']     # hold
            opt['each_domain'] = self.cfg['MODEL']['NORM']['EACH_DOMAIN_MTEST']
            opt['inner_clamp'] = self.meta_param['inner_clamp']
            self.cyclic_scheduler.step()
            opt['meta_ratio'] = self.cyclic_scheduler.get_lr()
            opt['norm_lr'] = self.scheduler_norm.last_lr
            opt['main_lr'] = self.scheduler_norm.last_lr
        else:
            raise ValueError
        return opt
