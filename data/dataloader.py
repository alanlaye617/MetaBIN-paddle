import paddle
import paddle.io
from transforms import build_transforms
from .datasets import DukeMTMC, Market1501
from .datasets import CommDataset
import samplers



def create_dataset(dataset_name, root='datasets'):
    if dataset_name == 'Market1501':
        return Market1501(root=root)
    elif dataset_name == 'DukeMTMC':
        return DukeMTMC(root=root)
    else:
        raise KeyError('Unknown dataset:', dataset_name)


def make_sampler(train_set, batch_size, num_instance, num_workers,
                 mini_batch_size, drop_last=True, naive_way=True, delete_rem=True, seed=None, camera_to_domain=None):
    if naive_way:
        data_sampler = samplers.NaiveIdentitySampler(data_source=train_set.img_items,
                                                    batch_size=batch_size,
                                                    num_instances=num_instance, 
                                                    delete_rem=delete_rem, 
                                                    seed=seed)
    else:
        data_sampler = samplers.DomainSuffleSampler(data_source=train_set.img_items,
                                                    batch_size=batch_size,
                                                    num_instances=num_instance, 
                                                    delete_rem=delete_rem, 
                                                    seed=seed, 
                                                    camera_to_domain=camera_to_domain)
    # data_sampler = samplers.BalancedIdentitySampler(train_set.img_items,num_batch, num_instance) # other method
    # data_sampler = samplers.TrainingSampler(len(train_set)) # PK sampler
    batch_sampler = paddle.io.BatchSampler(sampler=data_sampler, batch_size=mini_batch_size, drop_last=drop_last)

    train_loader = paddle.io.DataLoader(
        dataset=train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
    )
    return train_loader


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, paddle.Tensor):
        out = paddle.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out
    else:
        raise ValueError
    """
    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return paddle.tensor(batched_inputs, dtype=paddle.float64)
    elif isinstance(elem, int_classes):
        return paddle.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs    
    """


train_loader_kwargs = {
    'drop_last': True,         # cfg.DATALOADER.DROP_LAST,
    'naive_way': True,         # cfg.DATALOADER.NAIVE_WAY,
    'delete_rem': False
}

meta_loader_kwargs = {
    'drop_last': True,
    'naive_way': False,
    'delete_rem': False
}


def build_reid_train_loader(dataset_names, num_workers,
                            train_batch_size, train_num_instance,
                            mtrain_batch_size, mtrain_num_instance,
                            mtest_batch_size, mtest_num_instance,
                            train_loader_kwargs, meta_loader_kwargs,
                            synth_flag=False, camera_to_domain=True, meta_loader_flag='diff', meta_data_names='DG', 
                            seed=None, world_size=1
                            ):
    """
    meta_data_names: META.DATA.NAMES
    """

    # transforms
    train_transforms = build_transforms(is_train=True, is_fake=False)
    if synth_flag:
        synth_flag = build_transforms(is_train=True, is_fake=True)
        meta_loader_flag = 'each'
    else:
        synth_transforms = None
    train_set_all = []
    train_items = list()
    domain_idx = 0
    camera_all = list()

    # load datasets
    for name in dataset_names:
        assert name in dataset, KeyError('Unknown dataset: ' + name)
        dataset = create_dataset(name)
        if len(dataset.train[0]) < 4:
            for i, x in enumerate(dataset.train):
                additional_info = {}
                if camera_to_domain:
                    additional_info['domains'] = dataset.train[i][2]
                    camera_all.append(dataset.train[i][2])
                else:
                    additional_info['domains'] = int(domain_idx)
                dataset.train[i] = list(dataset.train[i])
                dataset.train[i].append(additional_info)
                dataset.train[i] = tuple(dataset.train[i])
        domain_idx += 1
        train_items.extend(dataset.train)

    if camera_to_domain: # used for single-source DG
        num_domains = len(set(camera_all))
    else:
        num_domains = domain_idx
    
    # if not navie_way:
    #    logger.info('**[dataloader info: random domain shuffle]**')
    train_set = CommDataset(train_items, train_transforms, relabel=True)
    if (synth_transforms is not None) and (meta_data_names != ""): # used for synthetic (not used in MetaBIN)
        synth_set = CommDataset(train_items, synth_transforms, relabel=True)
    
    train_loader = make_sampler(
        train_set=train_set,
        batch_size=train_batch_size,                        # cfg.SOLVER.IMS_PER_BATCH,
        num_instance=train_num_instance,                    # cfg.DATALOADER.NUM_INSTANCE,
        num_workers=num_workers,
        mini_batch_size=train_batch_size//world_size,       # cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size(),
        drop_last=train_loader_kwargs['drop_last'],         # cfg.DATALOADER.DROP_LAST,
        naive_way=train_loader_kwargs['naive_way'],         # cfg.DATALOADER.NAIVE_WAY,
        delete_rem=train_loader_kwargs['delete_rem'],       # cfg.DATALOADER.DELETE_REM,
        )
    


    train_loader_add = {}
    train_loader_add['mtrain'] = None # mtrain dataset
    train_loader_add['mtest'] = None # mtest dataset

    if meta_loader_flag == 'each': # "each": meta-init / meta-train / meta-test
        make_mtrain = True
        make_mtest = True
    elif meta_loader_flag == 'diff': # "diff": meta-init / meta-final
        make_mtrain = True
        make_mtest = False
    elif meta_loader_flag == 'same': # "same": meta-init
        make_mtrain = False
        make_mtest = False
    else:
        print('error in meta_loader_flag', meta_loader_flag)

    train_loader_add['mtrain'] = [] if make_mtrain else None
    train_loader_add['mtest'] = [] if make_mtest else None

    if make_mtrain: # meta train dataset
        train_loader_add['mtrain'] = make_sampler(
            train_set=train_set,
            batch_size=mtrain_batch_size,                       #cfg.META.DATA.MTRAIN_MINI_BATCH,
            num_instance=mtrain_num_instance,                   #cfg.META.DATA.MTRAIN_NUM_INSTANCE,
            num_workers=num_workers,
            mini_batch_size=mtrain_batch_size//world_size,      # cfg.META.DATA.MTRAIN_MINI_BATCH // comm.get_world_size(),
            drop_last=meta_loader_kwargs['drop_last'],          # cfg.META.DATA.DROP_LAST,
            naive_way=meta_loader_kwargs['naive_way'],          # cfg.META.DATA.NAIVE_WAY,
            delete_rem=meta_loader_kwargs['delete_rem'],        # cfg.META.DATA.DELETE_REM,
            seed = seed
            )

    if make_mtest: # meta train dataset
        if synth_transforms is None:
            train_loader_add['mtest'] = make_sampler(
                train_set=train_set,
                batch_size=mtest_batch_size,                        # cfg.META.DATA.MTEST_MINI_BATCH,
                num_instance=mtest_num_instance,                    # cfg.META.DATA.MTEST_NUM_INSTANCE,
                num_workers=num_workers,
                mini_batch_size=mtest_batch_size//world_size,       # cfg.META.DATA.MTEST_MINI_BATCH // comm.get_world_size(),
                drop_last=meta_loader_kwargs['drop_last'],          # cfg.META.DATA.DROP_LAST,
                naive_way=meta_loader_kwargs['naive_way'],          # cfg.META.DATA.NAIVE_WAY,
                delete_rem=meta_loader_kwargs['delete_rem'],        # cfg.META.DATA.DELETE_REM,
                seed = seed)
        else:
            train_loader_add['mtest'] = make_sampler(
                train_set=synth_set,
                batch_size=mtest_batch_size,                        # cfg.META.DATA.MTEST_MINI_BATCH,
                num_instance=mtest_num_instance,                    # cfg.META.DATA.MTEST_NUM_INSTANCE,
                num_workers=num_workers,
                mini_batch_size=mtest_batch_size//world_size,       # cfg.META.DATA.MTEST_MINI_BATCH // comm.get_world_size(),
                drop_last=meta_loader_kwargs['drop_last'],          # cfg.META.DATA.DROP_LAST,
                naive_way=meta_loader_kwargs['naive_way'],          # cfg.META.DATA.NAIVE_WAY,
                delete_rem=meta_loader_kwargs['delete_rem'],        # cfg.META.DATA.DELETE_REM,
                seed = seed)
    return train_loader, train_loader_add, num_domains



def build_reid_test_loader(dataset_name, num_workers, batch_size, flag_test=True):
    test_transforms = build_transforms(is_train=False)
    dataset = create_dataset(dataset_name) 
    if flag_test:
        test_items = dataset.query + dataset.gallery
    else:
        test_items = dataset.train
    test_set = CommDataset(test_items, test_transforms, relabel=False)
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = paddle.io.BatchSampler(sampler=data_sampler, batch_size=batch_size, drop_last=False)
    test_loader = paddle.io.DataLoader(
        dataset=test_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=fast_batch_collator
    )
    return test_loader, len(dataset.query)


