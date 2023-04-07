
import os
import contextlib
import filelock
import tempfile
import random
from urllib.parse import urlparse, unquote
import sys

import yaml
import numpy as np
from pytorch_lightning import seed_everything

import torch
import cv2

from segall.utils import logger,seg_env,get_sys_env
from segall.utils.download import download_file_and_uncompress

def set_seed(seed=None):
    if seed is not None:
        seed_everything(seed=seed)


def show_cfg_info(config):
    msg = '\n---------------Config Information---------------\n'
    ordered_module = ('batch_size', 'iters', 'train_dataset', 'val_dataset',
                      'optimizer', 'lr_scheduler', 'loss', 'model')
    all_module = set(config.dic.keys())
    for module in ordered_module:
        if module in config.dic:
            module_dic = {module: config.dic[module]}
            msg += str(yaml.dump(module_dic, Dumper=NoAliasDumper))
            all_module.remove(module)
    for module in all_module:
        module_dic = {module: config.dic[module]}
        msg += str(yaml.dump(module_dic, Dumper=NoAliasDumper))
    msg += '------------------------------------------------\n'
    logger.info(msg)

def show_env_info():
    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

def set_cv2_num_threads(num_workers):
    # Limit cv2 threads if too many subprocesses are spawned.
    # This should reduce resource allocation and thus boost performance.
    nranks = torch.cuda.device_count() # ! get gpus num
    # paddle.distributed.ParallelEnv().nranks
    if nranks >= 8 and num_workers >= 8:
        logger.warning("The number of threads used by OpenCV is " \
            "set to 1 to improve performance.")
        cv2.setNumThreads(1)

@contextlib.contextmanager
def generate_tempdir(directory: str=None, **kwargs):
    '''Generate a temporary directory'''
    # ! 生成临时文件，为了储存从网站上下载
    directory = seg_env.TMP_HOME if not directory else directory
    with tempfile.TemporaryDirectory(dir=directory, **kwargs) as _dir:
        yield _dir

def download_pretrained_model(pretrained_model):
    """
    Download pretrained model from url.
    Args:
        pretrained_model (str): the url of pretrained weight
    Returns:
        str: the path of pretrained weight
    """
    assert urlparse(pretrained_model).netloc, "The url is not valid."
    
    pretrained_model = unquote(pretrained_model)  #
    savename = pretrained_model.split('/')[-1]
    if not savename.endswith(('tgz', 'tar.gz', 'tar', 'zip')):
        savename = pretrained_model.split('/')[-2]
        filename = pretrained_model.split('/')[-1]
    else:
        savename = savename.split('.')[0]
        filename = 'model.pth'
    
    with generate_tempdir() as _dir:
        with filelock.FileLock(os.path.join(seg_env.TMP_HOME, savename)):
            pretrained_model = download_file_and_uncompress(
                pretrained_model,
                savepath=_dir,
                cover=False,
                extrapath=seg_env.PRETRAINED_MODEL_HOME,
                extraname=savename,
                filename=filename)
            pretrained_model = os.path.join(pretrained_model, filename)
    return pretrained_model

def save_ckpt(ckpt_dir, model, optimizer,scheduler ,iters,cur_iou,best_iou):
    state = {
        'iters': iters,
        'cur_iou': cur_iou,
        'best_iou': best_iou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    ckpt_model_filename = "ckpt_iters_{}.pth".format(iters)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))

def resume(resume_model,model,optimizer,scheduler):
    if os.path.isfile(resume_model):
            logger.info("=> loading checkpoint '{}'".format(resume_model))
            checkpoint = torch.load(
                resume_model)
            iters = checkpoint['iters']
            best_IoU = checkpoint["best_iou"]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (iter_{})".format(
                resume_model, checkpoint['iters']))
    else:
        raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(resume_model))

def load_entire_model(model, pretrained):
    if pretrained is not None:
        load_pretrained_model(model, pretrained)
    else:
        logger.warning('Weights are not loaded for {} model since the '
                       'path of weights is None'.format(
                           model.__class__.__name__))
        
def load_pretrained_model(model, pretrained_model):
    
    if pretrained_model is not None:
        logger.info('Loading pretrained model from {}'.format(pretrained_model))

        if urlparse(pretrained_model).netloc:
            pretrained_model = download_pretrained_model(pretrained_model)
        
        if os.path.exists(pretrained_model):
            #
            para_state_dict = torch.load(pretrained_model)

            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict['state_dict']:
                    logger.warning("{} is not in pretrained model".format(k))
                elif list(para_state_dict['state_dict'][k].shape) != list(model_state_dict[k]
                                                            .shape):
                    logger.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict['state_dict'][k].shape, model_state_dict[k]
                                .shape))
                else:
                    model_state_dict[k] = para_state_dict['state_dict'][k]
                    num_params_loaded += 1
            model.load_state_dict(model_state_dict)
            logger.info("There are {}/{} variables loaded into {}.".format(
                num_params_loaded,
                len(model_state_dict), model.__class__.__name__))

        else:
            raise ValueError('The pretrained model directory is not Found: {}'.
                             format(pretrained_model))
    else:
        logger.info(
            'No pretrained model to load, {} will be trained from scratch.'.
            format(model.__class__.__name__))



def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        # TODO : Mulit-gpu load ckpt
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file,
                                    map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        epoch = checkpoint['epoch']
        if 'best_miou' in checkpoint:
            best_miou = checkpoint['best_miou']
            print('Best mIoU:', best_miou)
        else:
            best_miou = 0

        if 'best_miou_epoch' in checkpoint:
            best_miou_epoch = checkpoint['best_miou_epoch']
            print('Best mIoU epoch:', best_miou_epoch)
        else:
            best_miou_epoch = 0
        return epoch, best_miou, best_miou_epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        logger.info('No model needed to resume.')
        sys.exit(1)
    
        


def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 100000))

def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
        else:
            image_dir = os.path.dirname(image_path)
            with open(image_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line.split()) > 1:
                        line = line.split()[0]
                    image_list.append(os.path.join(image_dir, line))
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if f.startswith('.'):
                    continue
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be a path of image, or a file list containing image paths, or a directory including images.'
        )

    if len(image_list) == 0:
        raise RuntimeError(
            'There are not image file in `--image_path`={}'.format(image_path))

    return image_list, image_dir

def get_Medical_image_list(image_path, valid_suffix=None, filter_key=None):
    """Get image list from image name or image directory name with valid suffix.

    if needed, filter_key can be used to whether 'include' the key word.
    When filter_key is not None，it indicates whether filenames should include certain key.


    Args:
    image_path(str): the image or image folder where you want to get a image list from.
    valid_suffix(tuple): Contain only the suffix you want to include.
    filter_key(dict): the key(ignore case) and whether you want to include it. e.g.:{"segmentation": True} will futher filter the imagename with segmentation in it.

    """
    if valid_suffix is None:
        valid_suffix = [
            'nii.gz', 'nii', 'dcm', 'nrrd', 'mhd', 'raw', 'npy', 'mha'
        ]

    image_list = []
    if os.path.isfile(image_path):
        if image_path.split("/")[-1].split('.', maxsplit=1)[-1] in valid_suffix:
            if filter_key is not None:
                f_name = image_path.split("/")[
                    -1]  # TODO change to system invariant
                for key, val in filter_key.items():
                    if (key in f_name.lower()) is not val:
                        break
                else:
                    image_list.append(image_path)
            else:
                image_list.append(image_path)
        else:
            raise FileNotFoundError(
                '{} is not a file end with supported suffix, the support suffixes are {}.'
                .format(image_path, valid_suffix))

    # load image in a directory
    elif os.path.isdir(image_path):
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if f.split(".", maxsplit=1)[-1] in valid_suffix:
                    if filter_key is not None:
                        for key, val in filter_key.items():
                            if (key in f.lower()) is not val:
                                break
                        else:
                            image_list.append(os.path.join(root, f))
                    else:
                        image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '{} is not found. it should be a path of image, or a directory including images.'.
            format(image_path))

    if len(image_list) == 0:
        raise RuntimeError(
            'There are not image file in `--image_path`={}'.format(image_path))

    return image_list

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


class CachedProperty(object):
    """
    A property that is only computed once per instance and then replaces itself with an ordinary attribute.

    The implementation refers to https://github.com/pydanny/cached-property/blob/master/cached_property.py .
        Note that this implementation does NOT work in multi-thread or coroutine senarios.
    """

    def __init__(self, func):
        super().__init__()
        self.func = func
        self.__doc__ = getattr(func, '__doc__', '')

    def __get__(self, obj, cls):
        if obj is None:
            return self
        val = self.func(obj)
        # Hack __dict__ of obj to inject the value
        # Note that this is only executed once
        obj.__dict__[self.func.__name__] = val
        return val


def get_in_channels(model_cfg):
    if 'backbone' in model_cfg:
        return model_cfg['backbone'].get('in_channels', None)
    else:
        return model_cfg.get('in_channels', None)


def set_in_channels(model_cfg, in_channels):
    model_cfg = model_cfg.copy()
    if 'backbone' in model_cfg:
        model_cfg['backbone']['in_channels'] = in_channels
    else:
        model_cfg['in_channels'] = in_channels
    return model_cfg
