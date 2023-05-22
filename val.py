
import argparse
import os 

import torch

from segall.cvlibs import manager,Config,SegBuilder
from segall.core import evaluate
from segall.utils import get_sys_env,logger,utils


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # Common params
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help='The path of trained weights to be loaded for evaluation.',
        type=str)
    parser.add_argument(
        '--num_workers',
        help='Number of workers for data loader. Bigger num_workers can speed up data processing.',
        type=int,
        default=0)


    # Data augment params
    parser.add_argument(
        '--aug_eval',
        help='Whether to use mulit-scales and flip augment for evaluation.',
        action='store_true')
    parser.add_argument(
        '--scales',
        nargs='+',
        help='Scales for data augment.',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        help='Whether to use flip horizontally augment.',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        help='Whether to use flip vertically augment.',
        action='store_true')

    # Sliding window evaluation params
    parser.add_argument(
        '--is_slide',
        help='Whether to evaluate images in sliding window method.',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        nargs=2,
        help='The crop size of sliding window, the first is width and the second is height.'
        'For example, `--crop_size 512 512`',
        type=int)
    parser.add_argument(
        '--stride',
        nargs=2,
        help='The stride of sliding window, the first is width and the second is height.'
        'For example, `--stride 512 512`',
        type=int)

    # Other params
    parser.add_argument(
        '--data_format',
        help='Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')
    parser.add_argument(
        '--auc_roc',
        help='Whether to use auc_roc metric.',
        type=bool,
        default=False)
    parser.add_argument(
        '--opts',
        help='Update the key-value pairs of all options.',
        default=None,
        nargs='+')

    return parser.parse_args()


def merge_test_config(cfg, args):
    test_config = cfg.test_config
    if args.aug_eval:
        test_config['aug_eval'] = args.aug_eval
        test_config['scales'] = args.scales
        test_config['flip_horizontal'] = args.flip_horizontal
        test_config['flip_vertical'] = args.flip_vertical
    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride
    return test_config


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(args.config, opts=args.opts)
    builder = SegBuilder(cfg)
    test_config = merge_test_config(cfg, args)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    # utils.set_device(args.device)

    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = builder.model.to(device)
    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained weights successfully.')
    val_dataset = builder.val_dataset

    evaluate(model, val_dataset, num_workers=args.num_workers, device=device,**test_config)


if __name__ == '__main__':
    args = parse_args()
    main(args)
