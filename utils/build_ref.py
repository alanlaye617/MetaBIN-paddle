#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys
import re

sys.path.append('./refs')
from refs.fastreid.evaluation import ReidEvaluator
from refs.fastreid.config import get_cfg
from refs.fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from refs.fastreid.evaluation import ReidEvaluator
from refs.fastreid.solver import build_lr_scheduler, build_optimizer
from refs.fastreid.utils.checkpoint import Checkpointer
from refs.fastreid.utils.file_io import PathManager
from .path import PathManager
import numpy as np
from PIL import Image, ImageOps
import torch
import numpy as np
import random
import glob


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, num_query, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return ReidEvaluator(cfg, num_query)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # automatic OUTPUT dir
    cfg.merge_from_file(args.config_file)
    config_file_name = args.config_file.split('/')
    for i, x in enumerate(config_file_name):
        if x == 'configs':
            config_file_name[i] = 'logs'
        if '.yml' in x:
            config_file_name[i] = config_file_name[i][:-4]
    cfg.OUTPUT_DIR = '/'.join(config_file_name)

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.eval_only or args.dist_only or args.tsne_only or args.domain_only:
        if args.eval_only:
            tmp = 'eval'
        if args.dist_only:
            tmp = 'dist'
        if args.tsne_only:
            tmp = 'tsne'
        if args.domain_only:
            tmp = 'domain'
        default_setup(cfg, args, tmp=tmp)
    else:
        default_setup(cfg, args)
    return cfg

def create_cfg(config_file='./refs/configs/Sample/M-resnet.yml', eval_only=True, resume=True):
    args = default_argument_parser().parse_args()
    args.config_file = config_file
    args.eval_only = eval_only
    args.resume = resume
    cfg = setup(args)
    return cfg

def build_ref_trainer(batch_size, train_dataset=['Market1501'], test_dataset=['Market1501', 'DukeMTMC'], eval_only=True, resume=False, pretrain=True):
    cfg = create_cfg(eval_only=eval_only, resume=resume)
    cfg.defrost()
    cfg.DATASETS.NAMES = train_dataset
    cfg.DATASETS.TESTS = test_dataset
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.META.MTRAIN_MINI_BATCH = batch_size
    cfg.META.MTEST_MINI_BATCH = batch_size
    cfg.MODEL.BACKBONE.PRETRAIN = pretrain
    trainer = Trainer(cfg)
    if resume:
        trainer.resume_or_load(resume=resume)
    return trainer

def build_ref_model(num_classes, eval_only=True, resume=False, pretrain=True):
    cfg = create_cfg(eval_only=eval_only, resume=resume)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = pretrain
    cfg.MODEL.HEADS.NUM_CLASSES = num_classes
    model = Trainer.build_model(cfg)
    if resume:
        save_file = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
        with PathManager.open(save_file, "r") as f:
            last_saved = f.read().strip()
        path = os.path.join(cfg.OUTPUT_DIR, last_saved)
        Checkpointer(model).load(path)
    return model

def build_ref_evaluator(num_query):
    cfg = create_cfg()
    cfg.defrost()
    return ReidEvaluator(cfg, num_query)


def read_image_ref(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.
    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"
    Returns:
        image (np.ndarray): an HWC image
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        image = Image.fromarray(image)
        return image
