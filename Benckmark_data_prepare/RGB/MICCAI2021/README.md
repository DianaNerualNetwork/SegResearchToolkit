


# 1	数据来源

本项目数据集来源于中国广州中山大学中山眼科中心，数据集中包含 200 个 2D 眼底彩照样本，分别为：训练集 100 对，测试集 100 对（用于平台提交，无标注）。

**【金标准制作的方法】**

各样本的初始视杯视盘分割区域标注由中国中山大学中山眼科中心4名临床眼科医生手动完成。这4名医生在没有获得任何患者信息或数据中疾病流行情况的情况下，独立对图像中视盘和视杯区域进行勾勒。随后，4个初始标注结果汇总给任务二中的更高级医生进行融合。视杯视盘分割结果的融合采用多数投票的方式，融合医生检查初始的分割标注，并选择取哪几位医生标注结果的交集作为最终视杯视盘分割金标准。

![](https://ai.bdstatic.com/file/A91B4E0AFF1342B0A7B9EFFAE194EBA8)

**【数据格式说明】**

- 训练集和测试集的图像：100 个样本编号 0001-0100/0101-0200，JPG格式。
- 训练集的标注：100 个样本编号 0001-0100，PNG格式；其中的像素值对应关系为：0 - 视杯、128 - 视盘中非视杯区域、255 - 其他区域。

# 准备方式：

运行以下命令：
```bash
bash Benckmark_data_prepare/RGB/MICCAI2021/disc_cpu.sh 

```