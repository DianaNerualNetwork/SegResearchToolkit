import os
import sys
import numpy as np

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
import torch
from segall.cvlibs import manager
from segall.transforms import BraTsCompose
from segall.datasets import MedicalDataset
import segall.transforms.functional_3d as F
import SimpleITK as sitk
import glob

URL = ' '  # todo: add coronavirus url after preprocess


@manager.DATASETS.add_component
class BraTsDataset(torch.utils.data.Dataset):
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
        |-data
        |---BraTS2020_Training
        |
        |---BraTS2020_Validation
    """


    def __init__(self,
                 dataset_root=None,
                 result_dir=None,
                 transforms=None,
                 num_classes=None,
                 mode='train',
                 ignore_index=255,
                 dataset_json_path="",
                 RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD=3):
        super(BraTsDataset, self).__init__(
            )

        self.dataset_root = dataset_root
        self.result_dir = result_dir
        self.transforms = BraTsCompose(transforms)
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.ignore_index = ignore_index  # todo: if labels only have 1/0/2, ignore_index is not necessary
        self.dataset_json_path = dataset_json_path
        self.RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD
        
        mode_util=""
        if self.mode=="train":
            file_name_list=os.listdir(os.path.join(self.dataset_root,"BraTS2020_Training"))
            mode_util="BraTS2020_Training"
        elif self.mode=="val":
            file_name_list=os.listdir(os.path.join(self.dataset_root,"BraTS2020_Validation"))
            mode_util="BraTS2020_Validation"
        else:
            raise "mode shoud be train or val"
        # print(glob.glob(os.path.join(os.path.join(self.dataset_root,"BraTS2020_Training"),file_name_list[0])+"/*.nii.gz"))

        for name in file_name_list:
            paths=sorted(glob.glob(os.path.join(os.path.join(self.dataset_root,mode_util),name)+"/*.nii.gz"))
            self.file_list.append(paths)
            
    def __getitem__(self, idx):
        files=self.file_list[idx]
        
        images,labels=self.transforms(files)
        print("transforms",images.shape)
        return images,labels
    
    def __len__(self):
        return len(self.file_list)

    
    # def generate_txt(self):



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