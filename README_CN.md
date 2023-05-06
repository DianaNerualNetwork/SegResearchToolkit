
## <div align="center">[English](README.md) | 简体中文</b></div>

<div align="center">

<p align="center">
  <img src="./doc/images/MixLab.jpg" align="middle" width = "500" />
</p>



**一个高效的Pytorch端到端语义分割科研工具箱.**


⚡[**HowTo**](#-HOWTOs) **|** 🔧[**安装环境**](doc/INSTALL.md) **|** 💻[**训练命令**](doc/trainingcommand.md) **|** 🐢[**数据准备**](docs/DatasetPreparation.md) **|** 🏰[**模型库**](doc/modelZoo/ModelZoo.md)


[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
</div>




## 📢 新增特性/更新资讯


  - ✅[2023-03-29] : fire: SegResearchToolkit v0.1 is released! 
  - ✅[2023-04-03] : fixed some transforms bug,and add U2Net, data prepare for optic_disc(MICCAI2021)。
 - ✅[2023-04-05] : Open New branch:Medical3D Segmentation,and get scripts to prepare lung_coronavirus dataset to download from the Internet.


## ✨ 简介


SegResearchToolkit是一个基于Pytorch的端到端高效图像分割研究开发工具包，它帮助开发人员和研究人员设计分割模型、训练模型、优化性能和推理速度以及部署模型的整个过程。工业界和学术界大量训练有素的模型和各种真实世界的应用程序帮助用户方便地建立图像分割的动手体验。 

**Note: 此工具箱很多代码思路和原型大多来自 [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) .感谢百度飞桨PaddleSeg的工程师们的智慧结晶!**

📕 现阶段支持的分割领域如下:

- ✅RGB Segmentation
- ✅RGBD Segmentation
- ✅Medical3D Segmentation





## ✨  Overview to SegResearchToolkit

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
<summary>支持的骨干网络并速读其论文请点击： </summary>

- ✅ [ResNet](./doc/models/backbone/ResNet/README.md)
</details>

<details open>
<summary>支持的网络结构并速读其论文请点击： </summary>
<details> <summary>RGB Segmentation </summary>

- ✅ [DeepLabv3p](./doc/models/DeepLabV3p/README.md)
- ✅ [U2Net](./doc/models/U2Net/README.md)
</details>
<details> <summary>Meidical3D Segmentation </summary>

- ✅ [VNet](./doc/models/DeepLabV3p/README.md)
- [x] [PANet](./doc/models/PANet_PriorAttention/README.md)
</details>
</details>



## <img src="https://user-images.githubusercontent.com/34859558/190046287-31b0467c-1a7e-4bf2-9e5e-40ff3eed94ee.png" width="25"/> 套件架构
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


## ⚡  数据准备
你可以选择下面展示的benckmark公开数据集基线：（请点击他们观看如何做数据准备）
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

- [segmentation-anything](https://github.com/facebookresearch/segment-anything)
- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)

## 🌏 Citations

如果SegResearchToolkit对你的工作或科研有帮助，请引用SegResearchToolkit<br>

作者: BairdXiong,本科毕业于 [武汉理工大学](http://www.whut.edu.cn/) 2023 . 现于中科院深圳先进技术研究院攻读电子信息专硕 [SIAT](https://www.siat.ac.cn/)(Shenzhen Institute of Advanced Technology，Chinese Academy of Sciences) .


The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.
```bash
@misc{Segall,
  author =       {BingXiong},
  title =        {{SegResearchToolkit}: A High-Efficient Research Development Toolkit for Image Segmentation Based on Pytorch.},
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

# 🚀 MixLab 介绍
我们的研究小组专注于计算机断层扫描（CT）、磁共振成像（MRI）、超声（US）、内窥镜和病理学等多模态肿瘤成像的智能计算方法和临床应用研究。由于恶性肿瘤是一种复杂的异质性疾病，以及不同成像技术的物理、化学和生物学特性，单一的成像方法在肿瘤成像的分辨率、灵敏度和对比度方面都有局限性。为了利用跨模态成像的信息，我们开发了基于学习的计算方法，通过计算成像来捕捉微观层面的肿瘤特征，并自动分割肿瘤/OAR，并从放射学图像和数字病理中提取信息，用于诊断、治疗计划、术中导航和预后预测。


秦博专注于计算机科学、医学物理学和生物医学跨学科的前沿研究，开发并应用计算方法进行生物医学光学成像和多模态医学数据分析，以实现精确的肿瘤诊断和治疗。该实验室在计算光学成像、图像计算和分析、计算可视化和人工智能方面开发了新方法，包括计算重建、无监督生成网络、多模态融合和多任务学习、弱监督、混合现实和决策理论，以应对利用多模态医疗数据的挑战。我们令人兴奋的工作是通过与中国科学院顶尖科学家以及国际其他机构的多学科合作，在生物医学工程领域架起桥梁。


我们与中国科学院的各种研究人员以及SIAT以外的临床研究人员合作。我们的最终目标是融合临床知识和工程之间的差距——通过计算机捕获、结构化、访问和处理所有肿瘤学知识和研究数据，以便我们能够在图像工作站中创建和提供新的工具，以提高临床有效性和效率。


miXlab位于中国科学院深圳高级技术学院生物医学与健康工程研究所医疗机器人与微创外科器械研究中心。

## 欢迎更多志同道合的同学加入我们实验室 !!! 
### 我们 [实验室主页](http://www.qin-mixlab.cn/) 欢迎了解！
