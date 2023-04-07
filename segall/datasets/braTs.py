import os
import sys
import numpy as np

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))

from segall.cvlibs import manager
from segall.transforms import MedicalCompose
from segall.datasets import MedicalDataset

URL = ' '  # todo: add coronavirus url after preprocess


@manager.DATASETS.add_component
class BraTsDataset(MedicalDataset):
    """
    The BraTs dataset is ...(todo: add link and description)
    Args:
        dataset_root (str): The dataset directory. Default: None
        result_root(str): The directory to save the result file. Default: None
        transforms (list): Transforms for image.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val'). Default: 'train'.
        Examples:
            transforms=[]
            dataset_root = "data/lung_coronavirus/lung_coronavirus_phase0/"
            dataset = LungCoronavirus(dataset_root=dataset_root, transforms=[], num_classes=3, mode="train")
            for data in dataset:
                img, label = data
                print(img.shape, label.shape) # (1, 128, 128, 128) (128, 128, 128)
                print(np.unique(label))
    """

    def __init__(self,
                 dataset_root=None,
                 result_dir=None,
                 transforms=None,
                 num_classes=None,
                 mode='train',
                 ignore_index=255,
                 dataset_json_path=""):
        super(BraTsDataset, self).__init__(
            dataset_root,
            result_dir,
            transforms,
            num_classes,
            mode,
            ignore_index,
            data_URL=URL,
            dataset_json_path=dataset_json_path)


# if __name__ == "__main__":
#     dataset = LungCoronavirus(
#         dataset_root="data/lung_coronavirus/lung_coronavirus_phase0",
#         result_dir="data/lung_coronavirus/lung_coronavirus_phase1",
#         transforms=[],
#         mode="train",
#         num_classes=23)
#     for item in dataset:
#         img, label = item
#         print(img.dtype, label.dtype)