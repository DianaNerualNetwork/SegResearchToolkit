## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

<div align="center">

<p align="center">
  <img src="./doc/images/MixLab.jpg" align="middle" width = "500" />
</p>



**A High-Efficient Research Development Toolkit for Image Segmentation Based on Pytorch.**


⚡[**HowTo**](#-HOWTOs) **|** 🔧[**Installation**](doc/INSTALL.md) **|** 💻[**Training Commands**](doc/trainingcommand.md) **|** 🐢[**DatasetPrepare**](docs/DatasetPreparation.md) **|** 🏰[**Model Zoo**](doc/modelZoo/ModelZoo.md)


[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
</div>




## 📢 New Features/Updates


  - ✅[2023-03-29] : fire: SegAll v0.1 is released! 
  - ✅[2023-04-03] : fixed some transforms bug,and add U2Net, data prepare for optic_disc(MICCAI2021)。
 - ✅[2023-04-05] : Open New branch:Medical3D Segmentation,and get scripts to prepare lung_coronavirus dataset to download from the Internet.


## ✨  Introduction

SegAll is an end-to-end high-efficent research development toolkit for image segmentation based on Pytorch, which  helps both developers and researchers in the whole process of designing segmentation models, training models, optimizing performance and inference speed, and deploying models. A lot of well-trained models and various real-world applications in both industry and academia help users conveniently build hands-on experiences in image segmentation.

**Note: This Toolkit many codes based on or inspired by [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) Toolkit.Thanks to their wonderful and elegant work!**

📕 The supported domains are the following:

- ✅RGB Segmentation
- ✅RGBD Segmentation
- ✅Medical3D Segmentation





## ✨  Overview to SegAll

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Models</b>
      </td>
      <td colspan="2">
        <b>Components</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <details><summary><b>RGB Segmentation</b></summary>
          <ul>
          <li>
              <a  href="./doc/models/DeepLabV3p/README.md">DeepLabV3</a>
            </li>
            <li>
              <a  href="./doc/models/U2Net/README.md">U2Net</a>
            </li>
          </ul>
        </details>
        <details><summary><b>RGBD Segmentation</b></summary>
          <ul>
            <li>ESANet</li>
          </ul>
        </details>
        <details><summary><b>Medical3D Segmentation</b></summary>
          <ul>
            <li>
            <a  href="./doc/models/U2Net/README.md">VNet</a>
            </li>
            <li>
            <a  href="./doc/models/PANet_PriorAttention/README.md">PANet</a>
            </li>
          </ul>
      </td>
      <td>
        <details><summary><b>Backbones</b></summary>
          <ul>
            <li><a href="./doc/models/backbone/ResNet/README.md">ResNet</a></li>
          </ul>
        </details>
        <details><summary><b>Losses</b></summary>
          <ul>
            <li><a href="./segall/models/losses/cross_entropy_loss.py">Cross Entropy Loss</a></li>
          </ul>
        </details>
        <details><summary><b>Metrics</b></summary>
          <ul>
            <li>mIoU</li>
            <li>Accuracy</li>
            <li>Kappa</li>
            <li>Dice</li>
            <li>AUC_ROC</li>
          </ul>  
        </details>
      </td>
      <td>
        <details><summary><b>Datasets</b></summary>
          <details><summary><b>RGB Datasets</b></summary>
          <ul>
            <li><a href="./paddleseg/datasets/optic_disc_seg.py">OPTIC DISC SEG</a></li>
          </ul>
          </details>
          <details><summary><b>RGBD Datasets</b></summary>
          <ul>
            <li><a href="./paddleseg/datasets/ade.py">SUNRGBD</a></li>
            <li><a href="./paddleseg/datasets/ade.py">NYUV2</a></li>  
          </ul>
          </details>
        </details>
        <details><summary><b>Data Augmentation</b></summary>
          <details><summary><b>RGB Data Augmentation</b></summary>
            <ul>
            <li>Flipping</li>  
            <li>Resize</li>  
            <li>ResizeByLong</li>
            <li>ResizeByShort</li>
            <li>LimitLong</li>  
            <li>ResizeRangeScaling</li>  
            <li>ResizeStepScaling</li>
            <li>Normalize</li>
            <li>Padding</li>
            <li>PaddingByAspectRatio</li>
            <li>RandomPaddingCrop</li>  
            <li>RandomCenterCrop</li>
            <li>ScalePadding</li>
            <li>RandomNoise</li>  
            <li>RandomBlur</li>  
            <li>RandomRotation</li>  
            <li>RandomScaleAspect</li>  
            <li>RandomDistort</li>  
            <li>RandomAffine</li>  
          </ul>
        </details>
        <details><summary><b>RGB Data Augmentation</b></summary>
        </details>
      </td>
      <td>
</td>
    </tr>
  </tbody>
</table>

<details open>
<summary>Supported backbones and swallow dates whole </summary>

- ✅ [ResNet](./doc/models/backbone/ResNet/README.md)
</details>

<details open>
<summary>Supported architecture and swallow dates whole </summary>
<details> <summary>RGB Segmentation </summary>

- ✅ [DeepLabv3p](./doc/models/DeepLabV3p/README.md)
- ✅ [U2Net](./doc/models/U2Net/README.md)
</details>
<details> <summary>Meidical3D Segmentation </summary>

- ✅ [VNet](./doc/models/DeepLabV3p/README.md)
- [x] [PANet](./doc/models/PANet_PriorAttention/README.md)
</details>
</details>



## <img src="https://user-images.githubusercontent.com/34859558/190046287-31b0467c-1a7e-4bf2-9e5e-40ff3eed94ee.png" width="25"/> Structure
This part shows you the whole picture of our repository, which is easy to expand with different model and datasets. Our file tree is as follows:

```bash
├── Benckmark_data_prepare  # All benckmark data prepare(SUNRGB-D,CitySpace,BrasTs..) stays here.
├── configs         # All configuration stays here.
├── data            # Data stays here.
├── segall  
│   ├── core        # the core training, val and test file.
│   ├── datasets  
│   ├── models  
│   ├── transforms  # the online data transforms
│   └── utils       # all kinds of utility files
├── export.py
├── tools           # Data preprocess including fetch data, process it and split into training and validation set
├── train.py
└── val.py
```


## ⚡  Data PrePare
You Can Choose the following benckmark:(Please Click them to read how to prepare it)
<details open>
<summary>Supported download benckmark from website </summary>

- ✅ [MICCAI2021 DISC SEG](Benckmark_data_prepare/RGB/MICCAI2021/README.md)
- ✅ [Medical3D- Lung-S](Benckmark_data_prepare/Meidical3D/lung.md)
- ⏳ RGBD- SUNRGB-D
- ⏳ RGBD- NYUv2
- ⏳ CitySpace


</details>





## 📜 License and Acknowledgement

This project is released under the [Apache 2.0 license](LICENSE.txt).<br>
More details about **license** and **acknowledgement** are in [LICENSE](LICENSE/README.md).

## 🌏 Citations

If SegAll helps your research or work, please cite SegAll.<br>


author: BairdXiong,Graduated from [WuhanUniversity of Techonolgy](http://www.whut.edu.cn/) 2023 . Studying for a master's degree in electronic information at [SIAT](https://www.siat.ac.cn/)(Shenzhen Institute of Advanced Technology，Chinese Academy of Sciences) Now.


The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.
```bash
@misc{Segall,
  author =       {BingXiong},
  title =        {{SegAll}: A High-Efficient Research Development Toolkit for Image Segmentation Based on Pytorch.},
  howpublished = {\url{https://github.com/DianaNerualNetwork/SegAll}},
  year =         {2023}
}

@misc{liu2021paddleseg,
      title={PaddleSeg: A High-Efficient Development Toolkit for Image Segmentation},
      author={Yi Liu and Lutao Chu and Guowei Chen and Zewu Wu and Zeyu Chen and Baohua Lai and Yuying Hao},
      year={2021},
      eprint={2101.06175},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{paddleseg2019,
    title={PaddleSeg, End-to-end image segmentation kit based on PaddlePaddle},
    author={PaddlePaddle Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleSeg}},
    year={2019}
}
```

# 🚀 Introduction to  MixLab
Our research group focus on intelligent computational methods and clinical application research on multi-modal tumor imaging such as computed tomography(CT), magnetic resonance imaging(MRI), Ultrasound(US), Endoscopy and Pathology. Since malignant tumor is a complex and heterogeneous disease, as well as the physical, chemical, and biological characteristics of different imaging technique, a single imaging method has its limitations in resolution, sensitivity, and contrast for tumor imaging. To leverage the information of cross-modality imaging, we develop learning-based computational methods to capture micro-level tumor characteristics by computing imaging and automatically segment tumor/OARs and extract information from radiology images and digital pathology for diagnosis, treatment planning, intraoperative navigation, and prognosis prediction.

The Qin Group focuses on cutting‐edge research in the interdisciplinary of computer science, medical physics, and biomedical science, developing and applying computational methods to biomedical optical imaging and multi-modality of medical data analysis for precision tumor diagnosis and therapy. The lab develops novel methods in computing optical imaging, image computing and analysis, computing visualization, and AI, including computing reconstruction, unsupervised generative network, multi-modal fusion, and multi-task learning, weak supervision, mixed reality, and decision theory to tackle the challenges of leveraging multi-modal medical data. Our exciting work is bridging the field of biomedical engineering domains with multidisciplinary collaborations with top scientists at the Chinese Academy of Sciences as well as with other institutions internationally.

We collaborate with a variety of investigators at the Chinese Academy of Sciences as well as with clinical investigators outside SIAT. Our ultimate goal is to bridge the gap between clinical knowledge and engineering--for all oncology knowledge and research data to be captured, structured, accessed, and processed by computers so that we can create and provide novel tools in image workstations for improving clinical effectiveness and efficiency.

The miX-Lab is in the Research Center for Medical Robotics and Minimally Invasive Surgical Devices of the Institute of Biomedical and Health Engineering at Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences.

## Welcome like-minded students to join our laboratory !!! 
### Our Lab [Homepage](http://www.qin-mixlab.cn/)
