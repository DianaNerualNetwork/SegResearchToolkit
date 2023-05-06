import argparse
import os 
import random 

import cv2 
import numpy as np 
import torch 
from pytorch_lightning import seed_everything

from textimageseg.cvlibs import Config,SegBuilder
from textimageseg.utils import get_sys_env,logger,utils

def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--iters',
        dest='iters',
        help='iters in training.',
        type=int,
        default=None)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu.',
        type=int,
        default=None)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=None)
    parser.add_argument(
        '--resume_model',
        dest='resume_model',
        help='The path of the model to resume.',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot.',
        type=str,
        default='./output')
    parser.add_argument(
        '--do_eval',
        dest='do_eval',
        help='Whether to do evaluation while training.',
        action='store_true')
    parser.add_argument(
        '--use_vdl',
        dest='use_vdl',
        help='Whether to record the data to VisualDL during training.',
        action='store_true')
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Set the random seed during training.',
        default=None,
        type=int)
    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='Display logging information at every `log_iters`.',
        default=10,
        type=int)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Number of workers for data loader.',
        type=int,
        default=0)
    parser.add_argument(
        '--opts', help='Update the key-value pairs of all options.', nargs='+')
    parser.add_argument(
        '--keep_checkpoint_max',
        dest='keep_checkpoint_max',
        help='Maximum number of checkpoints to save.',
        type=int,
        default=5)
    parser.add_argument(
        '--save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=1000)
    return  parser.parse_args()

def main(args):
    if args.seed is not None:
        seed_everything(args.seed)
    assert args.cfg is not None, \
        'No configuration file specified, please set --config'
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')
    cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size,
        opts=args.opts)
    builder = SegBuilder(cfg)
    # utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_seed(args.seed)
    utils.set_cv2_num_threads(args.num_workers)

if __name__ == "__main__":
    args = parse_args()
    main(args)