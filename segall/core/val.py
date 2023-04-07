
import os 
import numpy as np 
import time

import torch
import torch.nn as nn 
import torch.nn.functional as F 

import os
import json


from segall.core import infer
from segall.utils import metrics, TimeAverager, calculate_eta, logger, progbar, save_array
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
                # if precision == 'fp16':
                #     with paddle.amp.auto_cast(
                #             level=amp_level,
                #             enable=True,
                #             custom_white_list={
                #                 "elementwise_add", "batch_norm",
                #                 "sync_batch_norm"
                #             },
                #             custom_black_list={'bilinear_interp_v2'}):
                #         pred, logits = infer.aug_inference(
                #             model,
                #             data['img'],
                #             trans_info=data['trans_info'],
                #             scales=scales,
                #             flip_horizontal=flip_horizontal,
                #             flip_vertical=flip_vertical,
                #             is_slide=is_slide,
                #             stride=stride,
                #             crop_size=crop_size)
                # else:
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
                # if precision == 'fp16':
                #     with paddle.amp.auto_cast(
                #             level=amp_level,
                #             enable=True,
                #             custom_white_list={
                #                 "elementwise_add", "batch_norm",
                #                 "sync_batch_norm"
                #             },
                #             custom_black_list={'bilinear_interp_v2'}):
                #         pred, logits = infer.inference(
                #             model,
                #             data['img'],
                #             trans_info=data['trans_info'],
                #             is_slide=is_slide,
                #             stride=stride,
                #             crop_size=crop_size)
                # else:
                pred, logits = infer.inference(
                        model,
                        data['img'].to(device),
                        trans_info=data['trans_info'],
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)

            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred,
                label,
                eval_dataset.num_classes,
                ignore_index=eval_dataset.ignore_index)

            # # Gather from all ranks
            # if nranks > 1:
            #     intersect_area_list = []
            #     pred_area_list = []
            #     label_area_list = []
            #     paddle.distributed.all_gather(intersect_area_list,
            #                                   intersect_area)
            #     paddle.distributed.all_gather(pred_area_list, pred_area)
            #     paddle.distributed.all_gather(label_area_list, label_area)

            #     # Some image has been evaluated and should be eliminated in last iter
            #     if (iter + 1) * nranks > len(eval_dataset):
            #         valid = len(eval_dataset) - iter * nranks
            #         intersect_area_list = intersect_area_list[:valid]
            #         pred_area_list = pred_area_list[:valid]
            #         label_area_list = label_area_list[:valid]

            #     for i in range(len(intersect_area_list)):
            #         intersect_area_all = intersect_area_all + intersect_area_list[
            #             i]
            #         pred_area_all = pred_area_all + pred_area_list[i]
            #         label_area_all = label_area_all + label_area_list[i]
            # else:
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
    class_iou, miou = metrics.mean_iou(*metrics_input)
    acc, class_precision, class_recall = metrics.class_measurement(
        *metrics_input)
    kappa = metrics.kappa(*metrics_input)
    class_dice, mdice = metrics.dice(*metrics_input)

    if auc_roc:
        auc_roc = metrics.auc_roc(
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


def evaluate_3d(
        model,
        eval_dataset,
        losses,
        num_workers=0,
        print_detail=True,
        auc_roc=False,
        writer=None,
        save_dir=None,
        sw_num=None,
        is_save_data=True,
        has_dataset_json=True,device=torch.device('cpu') ):
    """
    Launch evalution.
    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        losses(dict): Used to calculate the loss. e.g: {"types":[loss_1...], "coef": [0.5,...]}
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.
        auc_roc(bool, optional): whether add auc_roc metric.
        writer: visualdl log writer.
        save_dir(str, optional): the path to save predicted result.
        sw_num:sw batch size.
        is_save_data:use savedata function
        has_dataset_json:has dataset_json
    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    new_loss = dict()
    new_loss['types'] = [losses['types'][0]]
    new_loss['coef'] = [losses['coef'][0]]
    model.eval()
    model=model.to(device)
    sampler = torch.utils.data.SequentialSampler(eval_dataset)
    batch_sampler=torch.utils.data.sampler.BatchSampler(sampler,batch_size=1,drop_last=True)
        
    loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        shuffle=False,
        num_workers=num_workers,
        )

    if has_dataset_json:

        with open(eval_dataset.dataset_json_path, 'r', encoding='utf-8') as f:
            dataset_json_dict = json.load(f)

    total_iters = len(loader)
    logits_all = None
    label_all = None

    if print_detail:
        logger.info("Start evaluating (total_samples: {}, total_iters: {})...".
                    format(len(eval_dataset), total_iters))
    progbar_val = progbar.Progbar(
        target=total_iters, verbose=1)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()

    mdice = 0.0
    channel_dice_array = np.array([])
    loss_all = 0.0

    with torch.no_grad():
        for iter, (im, label, idx) in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)

            if has_dataset_json:
                image_json = dataset_json_dict["training"][idx[0].split("/")[-1]
                                                           .split(".")[0]]
            else:
                image_json = None
            im=im.to(device)
            label = torch.as_tensor(label,dtype=torch.int32).to(device)

            if sw_num:
                pred, logits = infer.inference(  # reverse transform here
                    model,
                    im,
                    ori_shape=label.shape[-3:],
                    transforms=eval_dataset.transforms.transforms,
                    sw_num=sw_num)

            else:
                pred, logits = infer.inference(  # reverse transform here
                    model,
                    im,
                    ori_shape=label.shape[-3:],
                    transforms=eval_dataset.transforms.transforms)

            if writer is not None:  # TODO visualdl single channel pseudo label map transfer to
                pass

            if hasattr(model, "postprocess"):
                logits, label = model.postprocess(logits, label)
                # Update pred from postprocessed logits
                pred = torch.argmax(
                    logits[0], axis=1, keepdim=True, dtype='int32')

            # logits [N, num_classes, D, H, W] Compute loss to get dice
            loss, per_channel_dice = loss_3d_computation(logits, label, new_loss)
            loss = sum(loss)

            if auc_roc:
                logits = F.softmax(logits, axis=1)
                if logits_all is None:
                    logits_all = logits.numpy()
                    label_all = label.numpy()
                else:
                    logits_all = np.concatenate(
                        [logits_all, logits.numpy()])  # (KN, C, H, W)
                    label_all = np.concatenate([label_all, label.numpy()])

            loss_all += loss.numpy()
            mdice += np.mean(per_channel_dice)
            if channel_dice_array.size == 0:
                channel_dice_array = per_channel_dice
            else:
                channel_dice_array += per_channel_dice
            if is_save_data:
                if iter < 5:
                    if image_json is None:
                        raise ValueError(
                            "No json file is loaded. Please check if the dataset is preprocessed and `has_dataset_json` is True."
                        )
                    save_array(
                        save_path=os.path.join(save_dir, str(iter)),
                        save_content={
                            'pred': pred.numpy(),
                            'label': label.numpy(),
                            'img': im.numpy()
                        },
                        form=('npy', 'nii.gz'),
                        image_infor={
                            "spacing": image_json.get("spacing_resample",
                                                      image_json["spacing"]),
                            'direction': image_json["direction"],
                            "origin": image_json["origin"],
                            'format': "xyz"
                        })

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

    mdice /= total_iters
    channel_dice_array /= total_iters
    loss_all /= total_iters

    result_dict = {"mdice": mdice}
    if auc_roc:
        auc_roc = metrics.auc_roc(
            logits_all, label_all, num_classes=eval_dataset.num_classes)
        auc_infor = 'Auc_roc: {:.4f}'.format(auc_roc)
        result_dict['auc_roc'] = auc_roc

    if print_detail:
        infor = "[EVAL] #Images: {}, Dice: {:.4f}, Loss: {:6f}".format(
            len(eval_dataset), mdice, loss_all[0])
        infor = infor + auc_infor if auc_roc else infor
        logger.info(infor)
        logger.info("[EVAL] Class dice: \n" + str(
            np.round(channel_dice_array, 4)))

    return result_dict
