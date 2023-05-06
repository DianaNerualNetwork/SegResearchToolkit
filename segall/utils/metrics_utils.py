# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as skmetrics

__all__=['calculate_area','auc_roc','mean_iou','dice','class_measurement',
         'accuracy','kappa']

def calculate_area(pred, label, num_classes, ignore_index=255):
    """
    Calculate intersect, prediction and label area
    Args:
        pred (Tensor): The prediction by model.
        label (Tensor): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.
    Returns:
        Tensor: The intersection area of prediction and the ground on all class.
        Tensor: The prediction area on all class.
        Tensor: The ground truth area on all class
    """
    if len(pred.shape) == 4:
        pred = torch.squeeze(pred, axis=1)
    if len(label.shape) == 4:
        label = torch.squeeze(label, axis=1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(pred.shape,
                                                           label.shape))
    pred_area = []
    label_area = []
    intersect_area = []
    mask = label != ignore_index

    for i in range(num_classes):
        pred_i = torch.logical_and(pred == i, mask)
        label_i = label == i
        intersect_i = torch.logical_and(pred_i, label_i)
        pred_area.append(torch.unsqueeze(torch.sum(torch.as_tensor(pred_i, dtype=torch.int32)),dim=0))
        label_area.append(torch.unsqueeze(torch.sum(torch.as_tensor(label_i, dtype=torch.int32)),dim=0))
        intersect_area.append(torch.unsqueeze(torch.sum(torch.as_tensor(intersect_i, dtype=torch.int32)),dim=0))
    
    pred_area = torch.cat(pred_area)
    label_area = torch.cat(label_area)
    intersect_area = torch.cat(intersect_area)

    return intersect_area, pred_area, label_area


def auc_roc(logits, label, num_classes, ignore_index=None):
    """
    Calculate area under the roc curve
    Args:
        logits (Tensor): The prediction by model on testset, of shape (N,C,H,W) .
        label (Tensor): The ground truth of image.   (N,1,H,W)
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.
    Returns:
        auc_roc(float): The area under roc curve
    """
    if ignore_index or len(np.unique(label)) > num_classes:
        raise RuntimeError('labels with ignore_index is not supported yet.')

    if len(label.shape) != 4:
        raise ValueError(
            'The shape of label is not 4 dimension as (N, C, H, W), it is {}'.
            format(label.shape))

    if len(logits.shape) != 4:
        raise ValueError(
            'The shape of logits is not 4 dimension as (N, C, H, W), it is {}'.
            format(logits.shape))

    N, C, H, W = logits.shape
    logits = np.transpose(logits, (1, 0, 2, 3))
    logits = logits.reshape([C, N * H * W]).transpose([1, 0])

    label = np.transpose(label, (1, 0, 2, 3))
    label = label.reshape([1, N * H * W]).squeeze()

    if not logits.shape[0] == label.shape[0]:
        raise ValueError('length of `logit` and `label` should be equal, '
                         'but they are {} and {}.'.format(logits.shape[0],
                                                          label.shape[0]))

    if num_classes == 2:
        auc = skmetrics.roc_auc_score(label, logits[:, 1])
    else:
        auc = skmetrics.roc_auc_score(label, logits, multi_class='ovr')

    return auc


def mean_iou(intersect_area, pred_area, label_area):
    """
    Calculate iou.
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.
    Returns:
        np.ndarray: iou on all classes.
        float: mean iou of all classes.
    """
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    label_area = label_area.cpu().numpy()
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), miou


def dice(intersect_area, pred_area, label_area):
    """
    Calculate DICE.
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.
    Returns:
        np.ndarray: DICE on all classes.
        float: mean DICE of all classes.
    """
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    label_area = label_area.cpu().numpy()
    union = pred_area + label_area
    class_dice = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            dice = 0
        else:
            dice = (2 * intersect_area[i]) / union[i]
        class_dice.append(dice)
    mdice = np.mean(class_dice)
    return np.array(class_dice), mdice


# This is a deprecated function, please use class_measurement function.
def accuracy(intersect_area, pred_area):
    """
    Calculate accuracy
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes..
        pred_area (Tensor): The prediction area on all classes.
    Returns:
        np.ndarray: accuracy on all classes.
        float: mean accuracy.
    """
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    class_acc = []
    for i in range(len(intersect_area)):
        if pred_area[i] == 0:
            acc = 0
        else:
            acc = intersect_area[i] / pred_area[i]
        class_acc.append(acc)
    macc = np.sum(intersect_area) / np.sum(pred_area)
    return np.array(class_acc), macc


def class_measurement(intersect_area, pred_area, label_area):
    """
    Calculate accuracy, calss precision and class recall.
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.
    Returns:
        float: The mean accuracy.
        np.ndarray: The precision of all classes.
        np.ndarray: The recall of all classes.
    """
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    label_area = label_area.cpu().numpy()

    mean_acc = np.sum(intersect_area) / np.sum(pred_area)
    class_precision = []
    class_recall = []
    for i in range(len(intersect_area)):
        precision = 0 if pred_area[i] == 0 \
            else intersect_area[i] / pred_area[i]
        recall = 0 if label_area[i] == 0 \
            else intersect_area[i] / label_area[i]
        class_precision.append(precision)
        class_recall.append(recall)

    return mean_acc, np.array(class_precision), np.array(class_recall)


def kappa(intersect_area, pred_area, label_area):
    """
    Calculate kappa coefficient
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes..
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.
    Returns:
        float: kappa coefficient.
    """
    intersect_area = intersect_area.cpu().numpy().astype(np.float64)
    pred_area = pred_area.cpu().numpy().astype(np.float64)
    label_area = label_area.cpu().numpy().astype(np.float64)
    total_area = np.sum(label_area)
    po = np.sum(intersect_area) / total_area
    pe = np.sum(pred_area * label_area) / (total_area * total_area)
    kappa = (po - pe) / (1 - pe)
    return kappa


def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    For every value v in `labels`, the value in the output will be either 1 or 0. Each vector along the `dim`-th
    dimension has the "one-hot" format, i.e., it has a total length of `num_classes`,
    with a one and `num_class-1` zeros.
    Note that this will include the background label, thus a binary mask should be treated as having two classes.
    Args:
        labels: input tensor of integers to be converted into the 'one-hot' format. Internally `labels` will be
            converted into integers `labels.long()`.
        num_classes: number of output channels, the corresponding length of `labels[dim]` will be converted to
            `num_classes` from `1`.
        dtype: the data type of the output one_hot label.
        dim: the dimension to be converted to `num_classes` channels from `1` channel, should be non-negative number.
    Example:
    For a tensor `labels` of dimensions [B]1[spatial_dims], return a tensor of dimensions `[B]N[spatial_dims]`
    when `num_classes=N` number of classes and `dim=1`.
    .. code-block:: python
        from monai.networks.utils import one_hot
        import torch
        a = torch.randint(0, 2, size=(1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=0)
        print(out.shape)  # torch.Size([2, 2, 2, 2])
        a = torch.randint(0, 2, size=(2, 1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=1)
        print(out.shape)  # torch.Size([2, 2, 2, 2, 2])
    """

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)
    if len(labels.shape) == 4:
        labels = labels.unsqueeze(dim=1)
    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels