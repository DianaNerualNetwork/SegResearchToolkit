
import os 
import time 
from collections import deque
import shutil
from copy import deepcopy

import torch 
import torch.nn.functional as F 

from segall.utils import TimeAverager,calculate_eta,load_ckpt,logger,worker_init_fn,save_ckpt
from segall.core.val import evaluate

def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))


def loss_computation(logits_list, labels,  losses):
    # ! 实例化loss，并核对是否配置错误
    check_logits_losses(logits_list, losses)
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        coef_i = losses['coef'][i]
        # if loss_i.__class__.__name__ in ('BCELoss', ) and loss_i.edge_label:
        #     # Use edges as labels According to loss type.
        #     loss_list.append(coef_i * loss_i(logits, edges))
        if loss_i.__class__.__name__ == 'MixedLoss':
            mixed_loss_list = loss_i(logits, labels)
            for mixed_loss in mixed_loss_list:
                loss_list.append(coef_i * mixed_loss)
        elif loss_i.__class__.__name__ in ("KLLoss", ):
            loss_list.append(coef_i *
                             loss_i(logits_list[0], logits_list[1].detach()))
        else:
            loss_list.append(coef_i * loss_i(logits, labels))
    return loss_list

def train(model,
          train_dataset,
          val_dataset=None,
          optimizer=None,
          lr_she=None,
          save_dir='output',
          iters=10000,
          batch_size=2,
          resume_model=None,
          save_interval=1000,
          log_iters=10,
          num_workers=0,
          use_vdl=False,
          use_ema=False,
          losses=None,
          keep_checkpoint_max=5, # 保存几个iters的结果
          test_config=None,
          precision='fp32', # TODO
          amp_level='O1',  #  TODO 
          profiler_options=None,
          to_static_training=False,
          device=torch.device("cpu")):
    
    if use_ema:
        # ! use ema ?
        ema_model = deepcopy(model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.required_grad = False
    
    model.train()
    
    start_iter=0

    # model test
    

    if resume_model is not None:
        pass # TODO: Rusemue model training

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir, exist_ok=True)
    sampler = torch.utils.data.SequentialSampler(train_dataset)
    batch_sampler=torch.utils.data.sampler.BatchSampler(sampler,batch_size=batch_size,drop_last=True)
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
         )


    if use_vdl:
        from torch.utils.tensorboard import SummaryWriter
        log_writer = SummaryWriter(save_dir)
    
    
    avg_loss = 0.0
    avg_loss_list = []
    iters_per_epoch = len(batch_sampler)
    # ! print 
    print("Preparing training dataset......")
    print("training dataset total samples: ",iters_per_epoch)
    print("Prepare training dataset done, ready to training.....")
    best_mean_iou = -1.0
    best_ema_mean_iou = -1.0
    best_model_iter = -1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()
    iter_ = start_iter
    while iter_ < iters:
        # TODO ema
        # if iter == start_iter and use_ema:
        #     init_ema_params(ema_model, model)
        for data in loader:
            # ! each iters get once batch
            iter_ += 1
            if iter_ > iters:
                break
            reader_cost_averager.record(time.time() - batch_start)
            images = data['img'].to(device)
            labels = torch.as_tensor(data['label'],dtype=torch.int64).to(device)
            
            logits_list = model(images)
            
            # computation for loss
            loss_list=loss_computation(
                logits_list=logits_list,
                labels=labels,
                losses=losses
            )
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            
            lr=lr_she.get_last_lr()[0]
        
            if isinstance(lr_she, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_she.step(loss)
            else:
                lr_she.step()
        
            optimizer.zero_grad()

            avg_loss += float(loss.detach())
            if not avg_loss_list:
                # 如果avg_loss为空
                avg_loss_list = [l.detach().numpy() for l in loss_list]
            else:
                for i in range(len(loss_list)):
                    avg_loss_list[i] += loss_list[i].detach().numpy()
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)
            # print(avg_loss_list)
            if (iter_) % log_iters == 0: # and local_rank == 0:
                avg_loss /= log_iters
                #avg_loss_list = [l[0] / log_iters for l in avg_loss_list]
                avg_loss_list = [l / log_iters for l in avg_loss_list] ##### ?????? multi loss is right?
                remain_iters = iters - iter_
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                    .format((iter_ - 1) // iters_per_epoch + 1,
                            iter_,iters,avg_loss,lr,avg_train_batch_cost, avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(), eta
                            ))
                
                if use_vdl:
                    log_writer.add_scalar('Train/loss', avg_loss, iter_)
                    # Record all losses if there are more than 2 losses.
                    if len(avg_loss_list) > 1:
                        avg_loss_dict = {}
                        for i, value in enumerate(avg_loss_list):
                            avg_loss_dict['loss_' + str(i)] = value
                        for key, value in avg_loss_dict.items():
                            log_tag = 'Train/' + key
                            log_writer.add_scalar(log_tag, value, iter_)

                    log_writer.add_scalar('Train/lr', lr, iter_)
                    log_writer.add_scalar('Train/batch_cost',
                                          avg_train_batch_cost, iter_)
                    log_writer.add_scalar('Train/reader_cost',
                                          avg_train_reader_cost, iter_)
                avg_loss = 0.0
                avg_loss_list = []
                reader_cost_averager.reset()
                batch_cost_averager.reset()

            # TODO
            if use_ema:
                pass

            if (iter_ % save_interval == 0 or iter_ == iters) and (val_dataset is not None):
                # !! Question !!
                num_workers=1 if num_workers>0 else 0 # ? why do this ?

                
                mean_iou, acc, _, _, _ = evaluate(
                    model,
                    val_dataset,
                    num_workers=num_workers,
                    precision=precision,
                    amp_level=amp_level,
                    device=device
                    )
                
                model.train()
        
            if (iter_ % save_interval == 0 or iter_ == iters):
                current_save_dir = os.path.join(save_dir,
                                                    "iter_{}".format(iter_))
                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                save_ckpt(current_save_dir,model,optimizer,iter_) # save checkpoint for resume 
                save_models.append(current_save_dir)
                if len(save_models)>keep_checkpoint_max>0:
                    # 
                    model_to_remove=save_models.popleft()
                    shutil.rmtree(model_to_remove)
                if val_dataset is not None:
                    if mean_iou>best_mean_iou:
                        best_mean_iou = mean_iou
                        best_model_iter = iter_
                        best_model_dir = os.path.join(save_dir, "best_model")
                        if not(( os.path.isdir(best_model_dir)) and (os.path.exists(best_model_dir))):
                            os.mkdir(best_model_dir) # if path isnt a dir or exist 
                        save_ckpt(best_model_dir,model,optimizer,"best_model")
                        
                    logger.info(
                        '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'
                        .format(best_mean_iou, best_model_iter))

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/mIoU', mean_iou, iter_)
                        log_writer.add_scalar('Evaluate/Acc', acc, iter_)
                
            batch_start = time.time()
    # Sleep for a second to let dataloader release resources.
    time.sleep(1)
    if use_vdl:
        log_writer.close()
  