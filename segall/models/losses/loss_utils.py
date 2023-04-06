
import torch

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


def check_logits_3d_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))


def loss_3d_computation(logits_list, labels, losses, edges=None):
    check_logits_3d_losses(logits_list, losses)
    loss_list = []
    per_channel_dice = None

    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        coef_i = losses['coef'][i]

        if loss_i.__class__.__name__ in ('BCELoss3D', 'FocalLoss3D'
                                         ) and loss_i.edge_label:
            # If use edges as labels According to loss type.
            loss_list.append(coef_i * loss_i(logits, edges))
        elif loss_i.__class__.__name__ == 'MixedLoss3D':
            mixed_loss_list, per_channel_dice = loss_i(logits, labels)
            for mixed_loss in mixed_loss_list:
                loss_list.append(coef_i * mixed_loss)
        elif loss_i.__class__.__name__ in ("KLLoss3D", ):
            loss_list.append(coef_i *
                             loss_i(logits_list[0], logits_list[1].detach()))
        elif loss_i.__class__.__name__ in ["DiceLoss3D", "MultipleLoss3D"]:
            loss, per_channel_dice = loss_i(logits, labels)
            loss_list.append(coef_i * loss)
        else:
            loss_list.append(coef_i * loss_i(logits, labels))

    return loss_list, per_channel_dice


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
    (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # new axis order
    axis_order = (1, 0) + tuple(range(2, len(tensor.shape)))
    # print(axis_order)
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = torch.transpose(tensor, 0,1)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return torch.flatten(transposed, start_dim=1, end_dim=-1)

def class_weights(tensor):
    # normalize the input first
    tensor = torch.nn.functional.softmax(tensor, axis=1)
    flattened = flatten(tensor)
    nominator = (1. - flattened).sum(-1)
    denominator = flattened.sum(-1)
    class_weights = nominator / denominator
    class_weights.stop_gradient = True

    return class_weights