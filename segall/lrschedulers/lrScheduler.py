

import torch
from segall.cvlibs import manager
from segall.utils import logger

import math 
from torch.optim.lr_scheduler import _LRScheduler
    
@manager.LRSCHEDULER.add_component
class OneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    def __init__(self,*args, **kwds) -> None:
        super().__init__(*args, **kwds)
    
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)


