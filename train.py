

import argparse
import os 
import random 

import cv2 
import numpy as np 
import torch 
from pytorch_lightning import seed_everything


from segall.cvlibs import Config,SegBuilder
from segall.utils import get_sys_env,logger,utils
from segall.core import train,train_3d

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
    print(cfg)
    assert cfg.mode in ["RGBSeg","Medical3DSeg"],"mode should in [RGBSeg,Medical3DSeg]"
    # utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_seed(args.seed)
    utils.set_cv2_num_threads(args.num_workers)

    # TODO multi_gpu train
    model=builder.model.to(device)
    train_dataset=builder.train_dataset
    val_dataset=builder.val_dataset if args.do_eval else None
    optim=builder.optimizer
    loss=builder.loss # loss is a dict  for example ： {'coef': [1], 'types': [CrossEntropyLoss()]} not a impl
    lr_she=builder.lr_scheduler
    if cfg.mode=="RGBSeg":
        train(
        model,
        train_dataset,
        val_dataset=val_dataset,
        optimizer=optim,
        lr_she=lr_she,
        save_dir=args.save_dir,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        resume_model=args.resume_model,
        save_interval=args.save_interval,
        log_iters=args.log_iters,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        #use_ema=args.use_ema,# TODO
        losses=loss,
        keep_checkpoint_max=args.keep_checkpoint_max, ## TODO
        #test_config=cfg.test_config,# TODO
        #precision=args.precision, # TODO
        #amp_level=args.amp_level, # TODO
        #profiler_options=args.profiler_options, # TODO
        #to_static_training=cfg.to_static_training # TODO
        device=device
        )
    elif cfg.mode=="Medical3DSeg":
        train_3d(
            model,
          train_dataset,
          val_dataset,
          optimizer=optim,
          lr_she=lr_she,
          save_dir=args.save_dir,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        resume_model=args.resume_model,
        save_interval=args.save_interval,
        log_iters=args.log_iters,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        #use_ema=args.use_ema,# TODO
        losses=loss,
        keep_checkpoint_max=args.keep_checkpoint_max, ## TODO
        device=device,
        #   profiler_options=None, # TODO
        #   to_static_training=False, # TODO
          sw_num=None,
          is_save_data=True,
          has_dataset_json=True

        )
    
    

if __name__ == '__main__':
    args = parse_args()
    main(args)