# copyright (c) 2023 Baird Authors. All Rights Reserve.
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
# Thanks to

import sys
import time

import torch

levels = {0: 'ERROR', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}
log_level = 2


def log(level=2, message=""):
    
    current_time = time.time()
    time_array = time.localtime(current_time)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    if log_level >= level:
        print("{} [{}]\t{}".format(current_time, levels[level], message)
                  .encode("utf-8").decode("latin1"))
        sys.stdout.flush()


def debug(message=""):
    log(level=3, message=message)


def info(message=""):
    log(level=2, message=message)


def warning(message=""):
    log(level=1, message=message)


def error(message=""):
    log(level=0, message=message)

# import datetime
# import logging
# import os
# import sys,traceback
# import torch.distributed as dist
# import os 
# import csv
# from typing import List,Union

# _logger = None

# """
# # ! create_logger  创建日志器
# # ! get_created_logger_names 获取日志器名称
# # ! _read_file_line  获取文件行内容
# # ! error_traceback 日志器输出异常回溯

# """
# __all__=['create_logger',"get_created_logger_names","_read_file_line","error_traceback"]

# created_logger_names=[]

# def get_created_logger_names() ->List[str]:
#     """
#     获取已有日志器的名称
#     """
#     return created_logger_names


# def create_logger(logger_name: str,
#                   save_path :str =None ):
    

#     # ! crate logger
#     logger=logging.getLogger(logger_name)
#     # 判断是否已经存在logger
#     if logger_name in created_logger_names:
#         return logger 
    
#     # ?关闭关联日志器信息传播
#     logger.propagate=False

#     # 配置日志器记录级别
#     logger.setLevel(logging.INFO) #
#     # 配置格式化器
#     log_formarter=logging.Formatter(
#         '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
#         datefmt="%Y/%m/%d %H:%M:%S"
#     )
#     # 配置标准输出的处理器
#     stream_handler = logging.StreamHandler(stream=sys.stdout) # 得到标准输出
#     stream_handler.setLevel(logging.DEBUG)
#     stream_handler.setFormatter(log_formarter)
#     logger.addHandler(stream_handler)
    
#     # 配置文件输出的日志处理器
#     if save_path is not None:
#         # ! 判断文件
#         if save_path.endswith(".log") or \
#             save_path.endswith(".txt"):
#             # 如果是存在哪个文件里
#             if os.path.dirname(save_path): # ! 当前目录中是否存在目录
#                 if not os.path.isdir(os.path.dirname(save_path)): # ! 目录是否存在
#                      # 允许递归的实现目录创建
#                     os.makedirs(os.path.dirname(save_path))
#             log_filename=save_path
#         elif os.path.isdir(save_path):
#             # logging_starttime = datetime.datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
#             os.mkdir(save_path)# os.path.join(save_path,logging_starttime)
#             log_filename=os.path.join(save_path,"log.txt")# 如果传入的是目录，那就在这个目录文件下创建log文件
#             if not os.path.exists(log_filename):
#                 # 如果不存在
#                 open(log_filename,"w") # 创建空txt
            
#         else:
#             raise ValueError("save_path is not a file or a dir.(path at : {0})".format(save_path))
#         file_handler=logging.FileHandler(filename=log_filename,mode="w")
#         file_handler.setLevel(logging.DEBUG)
#         file_handler.setFormatter(log_formarter)
#         logger.addHandler(file_handler)

#     created_logger_names.append(logger_name)
#     return logger


# def _read_file_line(logger:logging.Logger,
#                     file_path:str,
#                     line: Union[int,List[int]]=0):
#     """读取文件指定行内容文本信息
#     """
#     if not os.path.isfile(file_path):
#         # 如果是个文件且不存在
#         raise ValueError("file path is not exist.")

#     # 判断输入到line是int 还是 List[int]
#     if isinstance(line,int):
#         line=[line,line+1]
#     else:
#         if not (len(line) ==2 and line[0]<line[1]):
#             raise ValueError("line should get 2 element and endline > startline")
    
#     line_strs=[]
#     with open(file_path,'r') as f:
#         line_strs=f.readlines()[line[0]:line[1]]
#     return ''.join(line_strs).strip() # 自动去掉首尾的空白符：空格，换行符


# def error_traceback(logger: logging.Logger,
#                     lasterrorline_offset: int=0,
#                     num_lines: int=1) -> None:
#     """利用日志器输出异常的回溯信息
#         desc:
#             Parameters:
#                 logger: 回溯文件使用的日志器(loggin.Logger)
#                 lasterrorline_offset: 最后一级有效回溯帧中实际错误信息所在行
#                                       与帧中记录行的行偏移值(int)
#                                       eg:
#                                           error_traceback(): ')'所在行往前算
#                 num_lines: 展示最后一级有效回溯帧所在实际行+其后的共n行的信息
#             Returns:
#                 None
#     """
#     # 1.获取最近抛出的异常形成的回溯数据
#     _e_type, _, _tb = sys.exc_info()
#     if _e_type == None: # None, 表示非有效异常或没有异常
#         return
#     # 2.由于回溯生成在该函数内，
#     #   而实际异常发生在该函数外，
#     #   因此，需要取最后一级回溯之前的堆栈信息
#     _summarys = traceback.extract_stack()[:-1]

#     # 3.取出最后一帧回溯
#     last_summary = _summarys[-1]
#     # 4.更新剩余回溯
#     if len(_summarys) == 1: # 仅仅包含一帧回溯
#         _summarys = []
#     else: # 否则获取最后帧外的回溯
#         _summarys = _summarys[:-1]

#     # 5.遍历异常发出后栈中最后帧除外的回溯信息
#     for _summary in _summarys:
#         # 帧所在文件
#         e_file = _summary.filename
#         # 帧所在函数
#         e_fcuntion_name = _summary.name
#         # 帧目标行(回溯行)
#         e_line = _summary.lineno
#         # 利用日志器输出回溯信息
#         logger.error("File \"{0}\", line {1}, in {2}\n\n\t{3}".format(
#             e_file, e_line, e_fcuntion_name,
#             _read_file_line(
#                 logger=logger,
#                 file_path=e_file,
#                 # 回溯帧目标行-1:
#                 # 由于帧行记录从1-n，读取文件行缓存的list序号从0-(n-1)
#                 # 因此帧行-1才能得到准确的行
#                 line=e_line - 1)))
#     # 6.输出最后的回溯帧，需要修正偏移位置
#     logger.error("File \"{0}\", line {1}, in {2}\n\n\t{3}".format(
#             last_summary.filename, last_summary.lineno - lasterrorline_offset,
#             last_summary.name,
#             _read_file_line(
#                 logger=logger,
#                 file_path=last_summary.filename,
#                 # 6.1之所以-lasterrorline_offset:
#                 #   由于实际发生/抛出错误/异常行的
#                 #   实际行号在error_traceback()的')'之前
#                 # 6.2输入一个list[start_line, end_line]，获取连续行的信息
#                 line=[last_summary.lineno - 1 - lasterrorline_offset,
#                       last_summary.lineno - 1 - lasterrorline_offset + num_lines])))


# class CSVLogger(object):
#     def __init__(self, keys, path, append=False):
#         super(CSVLogger, self).__init__()
#         self._keys = keys
#         self._path = path
#         if append is False or not os.path.exists(self._path):
#             with open(self._path, 'w') as f:
#                 w = csv.DictWriter(f, self._keys)
#                 w.writeheader()

#     def write_logs(self, logs):
#         with open(self._path, 'a') as f:
#             w = csv.DictWriter(f, self._keys)
#             w.writerow(logs)
