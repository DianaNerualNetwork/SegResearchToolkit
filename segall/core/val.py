
import os 
import numpy as np 
import time

import torch
import torch.nn as nn 
import torch.nn.functional as F 

import os
import json


from segall.core import infer
from segall.utils import TimeAverager, calculate_eta, logger, metrics_utils, progbar
from segall.models.losses import loss_computation,loss_3d_computation

np.set_printoptions(suppress=True)



def evaluate(model,
             eval_dataset, 
             aug_eval=False,
             scales=1.0,
             flip_horizontal=False,
             flip_vertical=False,
             is_slide=False,
             stride=None,
             crop_size=None,
             precision='fp32', # TODO
             amp_level='O1', # TODO
             num_workers=0,
             print_detail=True,
             auc_roc=False,
             device=torch.device('cpu')):
    """
    Launch evalution.
    Args:
        model（nn.Module): A semantic segmentation model.
        eval_dataset (torch.utils.data.Dataset): Used to read and process validation datasets.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to evaluate by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        precision (str, optional): Use AMP if precision='fp16'. If precision='fp32', the evaluation is normal.
        amp_level (str, optional): Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, the input data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel and batchnorm. Default is O1(amp)
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.
        auc_roc(bool, optional): whether add auc_roc metric
    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()
    
    
    sampler = torch.utils.data.SequentialSampler(eval_dataset)
    batch_sampler=torch.utils.data.sampler.BatchSampler(sampler,batch_size=1,drop_last=True)
        
    loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        shuffle=False,
        num_workers=num_workers,
        )

    total_iters = len(loader)
    intersect_area_all = torch.zeros([1], dtype=torch.int64)
    pred_area_all = torch.zeros([1], dtype=torch.int64)
    label_area_all = torch.zeros([1], dtype=torch.int64)
    logits_all = None
    label_all = None

    if print_detail:
        logger.info("Start evaluating (total_samples: {}, total_iters: {})...".
                    format(len(eval_dataset), total_iters))
    #TODO(chenguowei): fix log print error with multi-gpus
    progbar_val = progbar.Progbar(
        target=total_iters, verbose=1 )
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    
    with torch.no_grad():
        for iter, data in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)
            label = torch.as_tensor(data['label'],dtype=torch.int64).to(device)

            if aug_eval:
                pred, logits = infer.aug_inference(
                        model,
                        data['img'].to(device),
                        trans_info=data['trans_info'],
                        scales=scales,
                        flip_horizontal=flip_horizontal,
                        flip_vertical=flip_vertical,
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
            else:
               
                pred, logits = infer.inference(
                        model,
                        data['img'].to(device),
                        trans_info=data['trans_info'],
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)

            intersect_area, pred_area, label_area = metrics_utils.calculate_area(
                pred,
                label,
                eval_dataset.num_classes,
                ignore_index=eval_dataset.ignore_index)

            
            intersect_area_all = intersect_area_all.to(device) + intersect_area.to(device)
            pred_area_all = pred_area_all.to(device) + pred_area.to(device)
            label_area_all = label_area_all.to(device) + label_area.to(device)

            if auc_roc:
                logits = F.softmax(logits, axis=1)
                if logits_all is None:
                    logits_all = logits.numpy()
                    label_all = label.numpy()
                else:
                    logits_all = np.concatenate(
                            [logits_all, logits.numpy()])  # (KN, C, H, W)
                    label_all = np.concatenate([label_all, label.numpy()])

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if  print_detail:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                              ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    metrics_input = (intersect_area_all, pred_area_all, label_area_all)
    class_iou, miou = metrics_utils.mean_iou(*metrics_input)
    acc, class_precision, class_recall = metrics_utils.class_measurement(
        *metrics_input)
    kappa = metrics_utils.kappa(*metrics_input)
    class_dice, mdice = metrics_utils.dice(*metrics_input)

    if auc_roc:
        auc_roc = metrics_utils.auc_roc(
            logits_all, label_all, num_classes=eval_dataset.num_classes)
        auc_infor = ' Auc_roc: {:.4f}'.format(auc_roc)

    if print_detail:
        # ! 打印logger
        infor = "[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} Dice: {:.4f}".format(
            len(eval_dataset), miou, acc, kappa, mdice)
        infor = infor + auc_infor if auc_roc else infor
        logger.info(infor)
        logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
        logger.info("[EVAL] Class Precision: \n" + str(
            np.round(class_precision, 4)))
        logger.info("[EVAL] Class Recall: \n" + str(np.round(class_recall, 4)))
    return miou, acc, class_iou, class_precision, kappa

