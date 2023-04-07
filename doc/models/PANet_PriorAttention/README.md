# Prior Attention Network for Multi-Lesion Segmentation in Medical Images

**Authors:**  
> Xiangyu Zhao[1][2][3], Peng Zhang[1][2], Fan Song[1][2], Chenbin Ma[1][2], Guangda Fan[1][2], Yangyang Sun[1][2], Youdan Feng[1][2], Guanglei Zhang[1][2]*  

**Institution:**
> [1] School of Biological Science and Medical Engineering, Beihang University, Beijing, China  
> [2] Beijing Advanced Innovation Center for Biomedical Engineering, Beihang University, Beijing, China  
> [3] School of Biomedical Engineering, Shanghai Jiao Tong University, Shanghai, China  
> *Corresponding Author: Guanglei Zhang

manuscript link:  
- https://arxiv.org/abs/2110.04735 (preprint on arXiv, early version)  
- https://ieeexplore.ieee.org/document/9852260 (IEEE TMI, journal version)  

This repo contains the implementation of 3D segmentation of BraTS 2020 with the proposed *Prior Attention Network*.  
**If you use our code, please cite the paper:**  
> @ARTICLE{9852260,  
  author={Zhao, Xiangyu and Zhang, Peng and Song, Fan and Ma, Chenbin and Fan, Guangda and Sun, Yangyang and Feng, Youdan and Zhang, Guanglei},  
  journal={IEEE Transactions on Medical Imaging},   
  title={Prior Attention Network for Multi-Lesion Segmentation in Medical Images},   
  year={2022},  
  volume={},  
  number={},  
  pages={1-1},  
  doi={10.1109/TMI.2022.3197180}}  

## Methods
In this paper we propose a novel *Prior Attention Network* with intermediate supervision, parameterized skip connections and deep supervision strategy to address multi-lesion segmentation problems in medical images.  
### Network Topology
![network](https://user-images.githubusercontent.com/53631393/136913718-e94f7ba1-8444-4445-8682-692ff6a99a62.png)
### Attention Guiding Decoder
![AGD](https://user-images.githubusercontent.com/53631393/136913725-04e109d3-8081-49ca-948c-54e866692200.png)

## Results
### Quantitative Results
![Snipaste_2021-10-12_15-47-15](https://user-images.githubusercontent.com/53631393/136914282-3dd5a697-711b-4653-adb8-a6d2c98705f5.png)
### Qualitative Results
![vis3d](https://user-images.githubusercontent.com/53631393/136914543-023500b6-9a57-4f21-9f94-77961c7e9917.png)
### Ablation Analysis
![Snipaste_2021-10-12_15-47-32](https://user-images.githubusercontent.com/53631393/136914298-b76690c2-987d-4a3b-98da-9ab42f44ed10.png)

## Usage
### Data Preparation
Please download BraTS 2020 data according to `https://www.med.upenn.edu/cbica/brats2020/data.html`.  
Unzip downloaded data at `./data` folder (please create one) and remove all the csv files in the folder, or it will cause errors.
