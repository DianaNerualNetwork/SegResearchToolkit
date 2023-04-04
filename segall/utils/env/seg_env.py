
import os 
from segall.utils import logger

def _get_user_home():
    return os.path.expanduser('~')


def _get_seg_home():  
    if 'SEG_HOME' in os.environ:
        home_path = os.environ['SEG_HOME']
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                logger.warning('SEG_HOME {} is a file!'.format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), '.segall')


def _get_sub_home(directory):
    home = os.path.join(_get_seg_home(), directory)
    if not os.path.exists(home):
        os.makedirs(home, exist_ok=True)
    return home


USER_HOME = _get_user_home()
SEG_HOME = _get_seg_home()
DATA_HOME = _get_sub_home('dataset')
TMP_HOME = _get_sub_home('tmp')
PRETRAINED_MODEL_HOME = _get_sub_home('pretrained_model')
