

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



@manager.LRSCHEDULER.add_component
class CosineAnnealingWithWarmUp(_LRScheduler):
    """
            optimizer (Optimizer): Wrapped optimizer.
            first_cycle_steps (int): First cycle step size.
            cycle_mult(float): Cycle steps magnification. Default: -1.
            max_lr(float): First cycle's max learning rate. Default: 0.1.
            min_lr(float): Min learning rate. Default: 0.001.
            warmup_steps(int): Linear warmup step size. Default: 0.
            gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
            last_epoch (int): The index of last epoch. Default: -1.
        """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 cycle_steps: int,
                 max_lr_steps: int,
                 max_lr: float = 1e-2,
                 min_lr: float = 1e-8,
                 warmup_steps: int = 0,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < cycle_steps
        assert warmup_steps + max_lr_steps < cycle_steps

        self.cycle_steps = cycle_steps  # first cycle step size
        self.max_lr_steps = max_lr_steps
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size

        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWithWarmUp, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr[0] - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        elif self.step_in_cycle >= self.warmup_steps and self.step_in_cycle < self.warmup_steps + self.max_lr_steps:
            return [self.max_lr[0] for _ in self.base_lrs]
        else:
            return [base_lr + (self.max_lr[0] - base_lr) * (1 + math.cos(math.pi * (
                    self.step_in_cycle - self.warmup_steps - self.max_lr_steps) / (
                    self.cycle_steps - self.warmup_steps - self.max_lr_steps))) / 2 for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
        else:
            self.step_in_cycle = epoch

        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
