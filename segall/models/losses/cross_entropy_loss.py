
import torch
import torch.nn as nn
import torch.nn.functional as F 

from segall.cvlibs import manager

@manager.LOSSES.add_component
class CrossEntropyLoss(nn.Module):
    """
    Implements the cross entropy loss function.

    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0].
            When its value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining. Default ``1.0``.
        avg_non_ignore (bool, optional): Whether the loss is only averaged over non-ignored value of pixels. Default: True.
        
    """
    def __init__(self, weight=None,
                 ignore_index=255,
                 
                 avg_non_ignore=True,
                 device=torch.device('cpu'),
                  data_format='NCHW') -> None:
        super(CrossEntropyLoss,self).__init__()
        self.EPS=1.e-8
        self.ignore_index=ignore_index
        
        self.avg_non_ignore = avg_non_ignore
        self.device=device
        self.data_format=data_format
        if weight is not None:
            self.weight=torch.tensor(weight,dtype=torch.float32).to(device)
        else:
            self.weight=None

    def forward(self,logit,label,semantic_weights=None):
        # channel_axis = 1 if self.data_format == 'NCHW' else -1
        # if self.weight is not None and logit.shape[channel_axis] != len(
        #         self.weight):
        #     raise ValueError(
        #         'The number of weights = {} must be the same as the number of classes = {}.'
        #         .format(len(self.weight), logit.shape[channel_axis]))

        # if channel_axis == 1:
        #     logit = logit.transpose(1,2).transpose(2,3).to(self.device) # NHWC
        # else:
        logit=logit.to(self.device)
        label = torch.as_tensor(label,dtype=torch.int64).to(self.device)
        
        loss = F.cross_entropy(
            logit,
            label,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.weight)

        return self._post_process_loss(logit, label, semantic_weights, loss)

    def _post_process_loss(self, logit, label, semantic_weights, loss):
        """
        Consider mask and top_k to calculate the final loss.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            semantic_weights (Tensor, optional): Weights about loss for each pixels,
                shape is the same as label.
            loss (Tensor): Loss tensor which is the output of cross_entropy. If soft_label
                is False in cross_entropy, the shape of loss should be the same as the label.
                If soft_label is True in cross_entropy, the shape of loss should be
                (N, D1, D2,..., Dk, 1).
        Returns:
            (Tensor): The average loss.
        """
        if self.avg_non_ignore:
            mask = torch.as_tensor(label != self.ignore_index, dtype=torch.float32)
        else:
            mask = torch.ones(label.shape, dtype='float32')
        mask.required_grad = False
        label.required_grad = True

        if loss.ndim > mask.ndim:
            loss = torch.squeeze(loss, dim=-1)
        loss = loss * mask
        if semantic_weights is not None:
            loss = loss * semantic_weights

        if self.weight is not None:
            _one_hot = F.one_hot(label * mask, logit.shape[-1])
            coef = torch.sum(_one_hot * self.weight, dim=-1)
        else:
            coef = torch.ones_like(label)

        
        avg_loss = torch.mean(loss) / (torch.mean(mask * coef) + self.EPS)
        

        return avg_loss
