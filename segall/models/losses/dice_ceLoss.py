import torch

from segall.models.losses import DiceLoss3D
from segall.models.losses import CrossEntropyLoss
from segall.cvlibs import manager 

@manager.LOSSES.add_component
class DiceCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice=DiceLoss3D(sigmoid_norm=False)
        self.celoss=CrossEntropyLoss()
    def __call__(self,pred,label):
        return self.dice(pred,label)+self.celoss(pred,label)

