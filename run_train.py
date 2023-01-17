from engine import Trainer
import argparse
import paddle
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        "--config", 
        dest="cfg", 
        help="The config file.", 
        default=None, 
        type=str)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu.',
        type=int,
        default=96)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Number of workers for data loader.',
        type=int,
        default=0)
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Set the random seed during training.',
        default=None,
        type=int)
    return parser.parse_args()

def main(args):
    if args.seed is not None:
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')
    trainer = Trainer(
        cfg=args.cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    trainer.train()

if __name__ == '__main__':
    args = parse_args()
    main(args)