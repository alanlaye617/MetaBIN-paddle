from engine import Trainer
import argparse
import paddle
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluating')
    # params of eval
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
        default=128)
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='Dataset for evaluation.',
        default=None, 
        type=str)
    parser.add_argument(
        '--weight',
        dest='weight',
        help='path of the .pdparams file.',
        default=None, 
        type=str)
                
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Number of workers for data loader.',
        type=int,
        default=0)
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Set the random seed during evaluation.',
        default=None,
        type=int)
    return parser.parse_args()

def main(args):
    if args.seed is not None:
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    assert args.dataset in ["Market1501", "DukeMTMC"]
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')
    trainer = Trainer(
        cfg=args.cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    if args.weight:
        trainer.model.set_state_dict(paddle.load(args.weight))
    trainer.test(dataset_name=args.dataset, 
                 model=trainer.model,
                 num_workers=args.num_workers,
                 batch_size=args.batch_size)

if __name__ == '__main__':
    args = parse_args()
    main(args)