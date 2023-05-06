
import os 
import sys 
import zipfile 
import functools
import numpy as np
from prepare import Prep
from values import *
from segall.utils import wrapped_partial,resample
from uncompress import join_paths
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

class Prep_BraTs2020(Prep):
    def __init__(self, ) -> None:
        super().__init__()