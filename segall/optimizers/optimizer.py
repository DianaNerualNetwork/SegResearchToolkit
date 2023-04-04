# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


import torch
from segall.cvlibs import manager
from segall.utils import logger
import segall.optimizers.custom_optimizers  as custom_opt
class BaseOptimizer(object):
    def __init__(self, weight_decay=None,custom_cfg=None):
        if weight_decay is not None:
            assert isinstance(weight_decay, float), \
                "`weight_decay` must be a float."
        if weight_decay is not None:
            assert isinstance(weight_decay, float), \
                "`weight_decay` must be a float."
        if custom_cfg is not None:
            assert isinstance(custom_cfg, list), "`custom_cfg` must be a list."
            for item in custom_cfg:
                assert isinstance(
                    item, dict), "The item of `custom_cfg` must be a dict"
        self.weight_decay = weight_decay
        self.custom_cfg=custom_cfg
        self.args = {'weight_decay': weight_decay}


    def __call__(self, model, lr):
        # Create optimizer
        pass

    def _collect_params(self, model):
        # Collect different parameter groups
        if self.custom_cfg is None or len(self.custom_cfg) == 0:
            return model.parameters()

        groups_num = len(self.custom_cfg) + 1
        params_list = [[] for _ in range(groups_num)]
        for name, param in model.named_parameters():
            if param.stop_gradient:
                continue
            for idx, item in enumerate(self.custom_cfg):
                if item['name'] in name:
                    params_list[idx].append(param)
                    break
            else:
                params_list[-1].append(param)

        res = []
        for idx, item in enumerate(self.custom_cfg):
            lr_mult = item.get("lr_mult", 1.0)
            weight_decay_mult = item.get("weight_decay_mult", None)
            param_dict = {'params': params_list[idx], 'learning_rate': lr_mult}
            if self.weight_decay is not None and weight_decay_mult is not None:
                param_dict[
                    'weight_decay'] = self.weight_decay * weight_decay_mult
            res.append(param_dict)
        res.append({'params': params_list[-1]})

        msg = 'Parameter groups for optimizer: \n'
        for idx, item in enumerate(self.custom_cfg):
            params_name = [p.name for p in params_list[idx]]
            item = item.copy()
            item['params_name'] = params_name
            msg += 'Group {}: \n{} \n'.format(idx, item)
        msg += 'Last group:\n params_name: {}'.format(
            [p.name for p in params_list[-1]])
        logger.info(msg)

        return res


@manager.OPTIMIZERS.add_component
class SGD(BaseOptimizer):
    def __init__(self,lr, momentum=0,weight_decay=None, custom_cfg=None):
        super().__init__(weight_decay, custom_cfg=None)
        self.momentum=momentum
        self.lr=lr
    def __call__(self,model):
        param=self._collect_params(model)
        return torch.optim.SGD(params=param,lr=self.lr,momentum=self.momentum,**self.args)


