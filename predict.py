
import argparse
import torch 

from segall.cvlibs import manager,Config,SegBuilder
from segall.utils import get_sys_env,get_image_list,utils,logger
from segall.core import predict
from segall.transforms import Compose


def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    # Common params
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help='The path of trained weights for prediction.',
        type=str)
    parser.add_argument(
        '--image_path',
        help='The image to predict, which can be a path of image, or a file list containing image paths, or a directory including images',
        type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the predicted results.',
        type=str,
        default='./output/result')

    # Data augment params
    parser.add_argument(
        '--aug_pred',
        help='Whether to use mulit-scales and flip augment for prediction',
        action='store_true')
    parser.add_argument(
        '--scales',
        nargs='+',
        help='Scales for augment, e.g., `--scales 0.75 1.0 1.25`.',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        help='Whether to use flip horizontally augment',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        help='Whether to use flip vertically augment',
        action='store_true')

    # Sliding window evaluation params
    parser.add_argument(
        '--is_slide',
        help='Whether to predict images in sliding window method',
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

    # Custom color map
    parser.add_argument(
        '--custom_color',
        nargs='+',
        help='Save images with a custom color map. Default: None, use paddleseg\'s default color map.',
        type=int)

    return parser.parse_args()


def merge_test_config(cfg, args):
    test_config = cfg.test_config
    if 'aug_eval' in test_config:
        test_config.pop('aug_eval')
    if args.aug_pred:
        test_config['aug_pred'] = args.aug_pred
        test_config['scales'] = args.scales
        test_config['flip_horizontal'] = args.flip_horizontal
        test_config['flip_vertical'] = args.flip_vertical
    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride
    if args.custom_color:
        test_config['custom_color'] = args.custom_color
    return test_config


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(args.config)
    builder = SegBuilder(cfg)
    test_config = merge_test_config(cfg, args)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    # utils.set_device(args.device)
    device=torch.device("cuda" if torch.cuda.is_available() else False)
    model = builder.model.to(device)
    transforms = Compose(builder.val_transforms)
    image_list, image_dir = get_image_list(args.image_path)
    logger.info('The number of images: {}'.format(len(image_list)))

    predict(
        model,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=args.save_dir,
       device=device)


if __name__ == '__main__':
    args = parse_args()
    main(args)
