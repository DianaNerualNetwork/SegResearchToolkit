import six
import codecs
import os
from ast import literal_eval
from typing import Any, Dict, Optional

import yaml
import torch 


from textimageseg.cvlibs import config_checker as checker
from textimageseg.cvlibs import manager
from textimageseg.utils import logger,utils

_INHERIT_KEY = '_inherited_'
_BASE_KEY = '_base_'

class Config(object):
    def __init__(self,
                 path: str,# ! config文件地址
                 learning_rate: Optional[float]=None, # !Optional类型定义,方便了模块typing描述支持空值的函数类型和支持空值的结果类型
                 batch_size: Optional[int]=None,
                iters: Optional[int]=None,
                opts: Optional[list]=None,
                checker: Optional[checker.ConfigChecker]=None,
                 ) -> None:
        # ! 断言：path一定是存在的文件夹，并且是以yml，yaml为后缀的文件
        assert os.path.exists(path), \
            'Config path ({}) does not exist'.format(path)
        assert path.endswith('yml') or path.endswith('yaml'), \
            'Config file ({}) should be yaml format'.format(path)
        
        self.dic = self._parse_from_yaml(path) # ! 读取yaml/yml文件内容函数
        # !更新yaml文件中的内容
        self.dic = self.update_config_dict(
            self.dic,
            learning_rate=learning_rate,
            batch_size=batch_size,
            iters=iters,
            opts=opts)

        if checker is None:
            checker = self._build_default_checker()
        checker.apply_all_rules(self)
    
    @property
    def batch_size(self) -> int:
        return self.dic.get('batch_size')

    @property
    def iters(self) -> int:
        return self.dic.get('iters')

    @property
    def to_static_training(self) -> bool:
        return self.dic.get('to_static_training', False)

    @property
    def mode(self):
        return self.dic.get('mode')

    @property
    def model_cfg(self) -> Dict:
        return self.dic.get('model', {}).copy()

    @property
    def loss_cfg(self) -> Dict:
        return self.dic.get('loss', {}).copy()

    @property
    def distill_loss_cfg(self) -> Dict:
        return self.dic.get('distill_loss', {}).copy()

    @property
    def lr_scheduler_cfg(self) -> Dict:
        return self.dic.get('lr_scheduler', {}).copy()

    @property
    def optimizer_cfg(self) -> Dict:
        return self.dic.get('optimizer', {}).copy()

    @property
    def train_dataset_cfg(self) -> Dict:
        return self.dic.get('train_dataset', {}).copy()

    @property
    def val_dataset_cfg(self) -> Dict:
        return self.dic.get('val_dataset', {}).copy()

    # TODO merge test_config into val_dataset
    @property
    def test_config(self) -> Dict:
        return self.dic.get('test_config', {}).copy()

    @classmethod
    def update_config_dict(cls, dic: dict, *args, **kwargs) -> dict:
        return update_config_dict(dic, *args, **kwargs)

    @classmethod
    def _parse_from_yaml(cls, path: str, *args, **kwargs) -> dict:
        return parse_from_yaml(path, *args, **kwargs)
    
    @classmethod
    def _build_default_checker(cls):
        rules = []
        rules.append(checker.DefaultPrimaryRule())
        rules.append(checker.DefaultSyncNumClassesRule())
        rules.append(checker.DefaultSyncImgChannelsRule())
        # Losses
        rules.append(checker.DefaultLossRule('loss'))
        rules.append(checker.DefaultSyncIgnoreIndexRule('loss'))
        # Distillation losses
        rules.append(checker.DefaultLossRule('distill_loss'))
        rules.append(checker.DefaultSyncIgnoreIndexRule('distill_loss'))

        return checker.ConfigChecker(rules, allow_update=True)

    def __str__(self) -> str:
        # Use NoAliasDumper to avoid yml anchor 
        return yaml.dump(self.dic, Dumper=utils.NoAliasDumper)
    

def parse_from_yaml(path: str):
    """Parse a yaml file and build config"""
    with codecs.open(path, 'r', 'utf-8') as file:
        dic = yaml.load(file, Loader=yaml.FullLoader)

    if _BASE_KEY in dic:
        base_files = dic.pop(_BASE_KEY)
        if isinstance(base_files, str):
            base_files = [base_files]
        for bf in base_files:
            base_path = os.path.join(os.path.dirname(path), bf)
            base_dic = parse_from_yaml(base_path)
            dic = merge_config_dicts(dic, base_dic)

    return dic

def merge_config_dicts(dic, base_dic):
    """Merge dic to base_dic and return base_dic."""
    base_dic = base_dic.copy()
    dic = dic.copy()

    if not dic.get(_INHERIT_KEY, True):
        # 如果没有_INHERIT_KEY
        dic.pop(_INHERIT_KEY)
        return dic

    for key, val in dic.items():
        if isinstance(val, dict) and key in base_dic:
            base_dic[key] = merge_config_dicts(val, base_dic[key])
        else:
            base_dic[key] = val

    return base_dic

def update_config_dict(dic: dict,
                       learning_rate: Optional[float]=None,
                       batch_size: Optional[int]=None,
                       iters: Optional[int]=None,
                       opts: Optional[list]=None):
    """Update config"""
    # TODO: If the items to update are marked as anchors in the yaml file,
    # we should synchronize the references.
    dic = dic.copy()

    if learning_rate:
        dic['optimizer']['lr'] = learning_rate
    if batch_size:
        dic['batch_size'] = batch_size
    if iters:
        dic['iters'] = iters

    if opts is not None:
        for item in opts:
            assert ('=' in item) and (len(item.split('=')) == 2), "--opts params should be key=value," \
                " such as `--opts batch_size=1 test_config.scales=0.75,1.0,1.25`, " \
                "but got ({})".format(opts)

            key, value = item.split('=')
            if isinstance(value, six.string_types):
                try:
                    value = literal_eval(value)
                except ValueError:
                    pass
                except SyntaxError:
                    pass
            key_list = key.split('.')

            tmp_dic = dic
            for subkey in key_list[:-1]:
                assert subkey in tmp_dic, "Can not update {}, because it is not in config.".format(
                    key)
                tmp_dic = tmp_dic[subkey]
            tmp_dic[key_list[-1]] = value

    return dic