
import collections.abc
from itertools import combinations

import numpy as np 
import cv2
import torch
import torch.nn.functional as F 

def reverse_transform(pred, trans_info, mode='nearest',device=torch.device('cpu')):
    """recover pred to origin shape"""
    intTypeList = [torch.int8, torch.int16, torch.int32, torch.int64]
    dtype = pred.dtype
    
    for item in trans_info[::-1]:
        
        if isinstance(item[0], list):
            trans_mode = item[0][0]
        else:
            trans_mode = item[0]
        
        if  'resize' in trans_mode:
            h, w = item[1][0], item[1][1]
            if device == torch.device('cpu') and dtype in intTypeList:
                pred = torch.as_tensor(pred, dtype=torch.float32)
                pred = F.interpolate(pred, [h, w], mode=mode)
                pred = torch.as_tensor(pred, dtype)
            else:
                pred = F.interpolate(pred, [h, w], mode=mode)
        elif trans_mode == 'padding':
            h, w = item[1][0], item[1][1]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return pred

def flip_combination(flip_horizontal=False, flip_vertical=False):
    """
    Get flip combination.
    Args:
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.
    Returns:
        list: List of tuple. The first element of tuple is whether to flip horizontally,
            and the second is whether to flip vertically.
    """

    flip_comb = [(False, False)]
    if flip_horizontal:
        flip_comb.append((True, False))
    if flip_vertical:
        flip_comb.append((False, True))
        if flip_horizontal:
            flip_comb.append((True, True))
    return flip_comb


def tensor_flip(x, flip):
    """Flip tensor according directions"""
    if flip[0]:
        x = x[:, :, :, ::-1]
    if flip[1]:
        x = x[:, :, ::-1, :]
    return x


def slide_inference(model, im, crop_size, stride):
    """
    Infer by sliding window.
    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        crop_size (tuple|list). The size of sliding window, (w, h).
        stride (tuple|list). The size of stride, (w, h).
    Return:
        Tensor: The logit of input image.
    """
    h_im, w_im = im.shape[-2:]
    w_crop, h_crop = crop_size
    w_stride, h_stride = stride
    # calculate the crop nums
    rows = np.int(np.ceil(1.0 * (h_im - h_crop) / h_stride)) + 1
    cols = np.int(np.ceil(1.0 * (w_im - w_crop) / w_stride)) + 1
    # prevent negative sliding rounds when imgs after scaling << crop_size
    rows = 1 if h_im <= h_crop else rows
    cols = 1 if w_im <= w_crop else cols
    # TODO 'Tensor' object does not support item assignment. If support, use tensor to calculation.
    final_logit = None
    count = np.zeros([1, 1, h_im, w_im])
    for r in range(rows):
        for c in range(cols):
            h1 = r * h_stride
            w1 = c * w_stride
            h2 = min(h1 + h_crop, h_im)
            w2 = min(w1 + w_crop, w_im)
            h1 = max(h2 - h_crop, 0)
            w1 = max(w2 - w_crop, 0)
            im_crop = im[:, :, h1:h2, w1:w2]
            logits = model(im_crop)
            if not isinstance(logits, collections.abc.Sequence):
                raise TypeError(
                    "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                    .format(type(logits)))
            logit = logits[0].numpy()
            if final_logit is None:
                final_logit = np.zeros([1, logit.shape[1], h_im, w_im])
            final_logit[:, :, h1:h2, w1:w2] += logit[:, :, :h2 - h1, :w2 - w1]
            count[:, :, h1:h2, w1:w2] += 1
    if np.sum(count == 0) != 0:
        raise RuntimeError(
            'There are pixel not predicted. It is possible that stride is greater than crop_size'
        )
    final_logit = final_logit / count
    final_logit = torch.tensor(final_logit)
    return final_logit


def inference(model,
              im,
              trans_info=None,
              is_slide=False,
              stride=None,
              crop_size=None,
              class_label=None):
    """
    Inference for image.
    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        trans_info (list): Image shape informating changed process. Default: None.
        is_slide (bool): Whether to infer by sliding window. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.
    Returns:
        Tensor: If ori_shape is not None, a prediction with shape (1, 1, h, w) is returned.
            If ori_shape is None, a logit with shape (1, num_classes, h, w) is returned.
    """
    
    if not is_slide:
        logits = model(im)
        if not isinstance(logits, collections.abc.Sequence):
            raise TypeError(
                "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                .format(type(logits)))
        
        logit=logits[0]
    else:
        logit = slide_inference(model, im, crop_size=crop_size, stride=stride)
    
    if trans_info is not None:
        
        logit = reverse_transform(logit, trans_info, mode='bilinear')
        pred = torch.argmax(logit, dim=1, keepdim=True)
        return pred, logit
    else:
        return logit


def aug_inference(model,
                  im,
                  trans_info,
                  scales=1.0,
                  flip_horizontal=False,
                  flip_vertical=False,
                  is_slide=False,
                  stride=None,
                  crop_size=None):
    """
    Infer with augmentation.
    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        trans_info (list): Transforms for image.
        scales (float|tuple|list):  Scales for resize. Default: 1.
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.
        is_slide (bool): Whether to infer by sliding wimdow. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.
    Returns:
        Tensor: Prediction of image with shape (1, 1, h, w) is returned.
    """
    if isinstance(scales, float):
        scales = [scales]
    elif not isinstance(scales, (tuple, list)):
        raise TypeError(
            '`scales` expects float/tuple/list type, but received {}'.format(
                type(scales)))
    final_logit = 0
    h_input, w_input = im.shape[-2], im.shape[-1]
    flip_comb = flip_combination(flip_horizontal, flip_vertical)
    for scale in scales:
        h = int(h_input * scale + 0.5)
        w = int(w_input * scale + 0.5)
        im = F.interpolate(im, [h, w], mode='bilinear')
        for flip in flip_comb:
            im_flip = tensor_flip(im, flip)
            logit = inference(
                model,
                im_flip,
                is_slide=is_slide,
                crop_size=crop_size,
                stride=stride)
            logit = tensor_flip(logit, flip)
            logit = F.interpolate(logit, [h_input, w_input], mode='bilinear')

            logit = F.softmax(logit, axis=1)
            final_logit = final_logit + logit

    final_logit = reverse_transform(final_logit, trans_info, mode='bilinear')
    pred = torch.argmax(final_logit, axis=1, keepdim=True)

    return pred, final_logit