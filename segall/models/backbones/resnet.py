import torch
import torch.nn as nn 
import torch.nn.functional as F
 
from segall.cvlibs import manager
from segall.models import layers 
from segall.utils import utils

from collections import OrderedDict

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 is_vd_mode=False,
                 act=None,
                  **kwargs) -> None:
        super(ConvBN,self).__init__()
        if dilation != 1 and kernel_size != 3:
            raise RuntimeError("When the dilation isn't 1," \
                "the kernel_size should be 3.")
        self.is_vd_mode=is_vd_mode
        self._pool2d_avg = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
            padding=0,
            ceil_mode=True,
            )
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 \
                if dilation == 1 else dilation,
            dilation=dilation,
            groups=groups,
            bias=False,
            )

        self._batch_norm = nn.BatchNorm2d(
            out_channels)
        self._act_op = layers.Activation(act=act)

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        y = self._act_op(y)

        return y
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels,out_channels,stride,shortcut=True,dilation=1,if_first=False, **kwargs) -> None:
        super(BottleneckBlock,self).__init__()
        """
        Args: 
            in_channels: the number of  input feature map channels
            out_channels: the number of  output feature map channels
            stride:  kernel move stride every step
            shortcut: is shortcut or not    
            dilation: dilation rate
            is_frist: is first block or not ,if it is: the frist block use pooling layer instead
        """
        self.conv0 = ConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            )

        self.dilation = dilation

        self.conv1 = ConvBN(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            dilation=dilation,
            )
        self.conv2 = ConvBN(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            )

        if not shortcut:
            # ! 如果不使用shortcut
            self.short = ConvBN(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first or stride == 1 else True,
                )
        self.shortcut=shortcut
        # NOTE: Use the wrap layer for quantization training (PaddleSeg)
        self.add = layers.Add()
        self.relu = layers.Activation(act="relu")
    def forward(self,inputs):
        
        y=self.conv0(inputs)
        conv1=self.conv1(y)
        conv2=self.conv2(conv1)

        if self.shortcut:
            shortcut=inputs
        else:
            shortcut=self.short(inputs)

        y = self.add(shortcut, conv2)
        y = self.relu(y)
        return y

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilation=1,
                 shortcut=True,
                 if_first=False,
                 ):
        super(BasicBlock, self).__init__()
        self.conv0 = ConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            act='relu',
            )
        self.conv1 = ConvBN(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=dilation,
            act=None)

        if not shortcut:
            self.short = ConvBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first or stride == 1 else True)

        self.shortcut = shortcut
        self.dilation = dilation
        
        self.add = layers.Add()
        self.relu = layers.Activation(act="relu")

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = self.add(short, conv1)
        y = self.relu(y)

        return y

class ResNet_vd(nn.Module):
    def __init__(self, layers=50, output_stride=8, multi_grid=(1, 1, 1), in_channels=3, pretrained=None):
        super(ResNet_vd, self).__init__()

        self.conv1_logit = None
        self.layers = layers
        supported_layers = [18, 34, 50, 101]
        assert layers in supported_layers, "Supported layers are {} but input layer is {}".format(
            supported_layers, layers)

        self.multi_grid = multi_grid

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        self.num_channels = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
        self.num_filters = [64, 128, 256, 512]

        self.feat_channels = [c * 4 for c in self.num_filters] if layers >= 50 else self.num_filters

        self.dilation_dict = None
        if output_stride == 8:
            self.dilation_dict = {2: 2, 3: 4}
        elif output_stride == 16:
            self.dilation_dict = {3: 2}

        self.conv1_1 = ConvBN(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            act='relu',
        )
        self.conv1_2 = ConvBN(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu',
        )
        self.conv1_3 = ConvBN(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu',
        )
        self.pool2d_max = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1)

        if self.layers >= 50:
            self.layer1 = self._make_layer(
                block=BottleneckBlock, stage=1, in_channels=self.num_channels[0],
                out_channels=self.num_filters[0], block_num=depth[0],mode='bottle'
            )
            self.layer2 = self._make_layer(
                block=BottleneckBlock, stage=1, in_channels=self.num_channels[1],
                out_channels=self.num_filters[1], block_num=depth[1],mode='bottle'
            )
            self.layer3 = self._make_layer(
                block=BottleneckBlock, stage=2, in_channels=self.num_channels[2],
                out_channels=self.num_filters[2], block_num=depth[2],mode='bottle'
            )
            self.layer4 = self._make_layer(
                block=BottleneckBlock, stage=4, in_channels=self.num_channels[3],
                out_channels=self.num_filters[3], block_num=depth[3],mode='bottle'
            )
        else:
            self.layer1 = self._make_layer(
                block=BasicBlock, stage=1, in_channels=self.num_channels[0],
                out_channels=self.num_filters[0], block_num=depth[0],
            )
            self.layer2 = self._make_layer(
                block=BasicBlock, stage=1, in_channels=self.num_channels[1],
                out_channels=self.num_filters[1], block_num=depth[1],
            )
            self.layer3 = self._make_layer(
                block=BasicBlock, stage=2, in_channels=self.num_channels[2],
                out_channels=self.num_filters[2], block_num=depth[2],
            )
            self.layer4 = self._make_layer(
                block=BasicBlock, stage=4, in_channels=self.num_channels[3],
                out_channels=self.num_filters[3], block_num=depth[3],
            )

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        self.conv1_logit = y.clone()
        y = self.pool2d_max(y)

        feat_list = []
        feat1 = self.layer1(y)
        feat_list.append(feat1)
        feat2 = self.layer2(feat1)
        feat_list.append(feat2)
        feat3 = self.layer3(feat2)
        feat_list.append(feat3)
        feat4 = self.layer4(feat3)
        feat_list.append(feat4)
        return feat_list

    def _make_layer(self, block, stage, in_channels, out_channels, block_num,mode='basic'):
        layers = []
        shortcut = False
        for i in range(block_num):
            dilation_rate = self.dilation_dict[i] if (self.dilation_dict) and (i in self.dilation_dict) else 1
            if stage == 3:
                dilation_rate = dilation_rate * self.multi_grid[i]
            if mode=="basic":
                layers.append(
                block(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    stride=2 if i == 0 and stage != 0 and dilation_rate == 1 else 1,
                    shortcut=shortcut,
                    if_first=(block_num == i == 0),
                    dilation=dilation_rate,)
                )
            elif mode=="bottle":
                layers.append(
                block(
                    in_channels=in_channels if i == 0 else out_channels*4,
                    out_channels=out_channels,
                    stride=2 if i == 0 and stage != 0 and dilation_rate == 1 else 1,
                    shortcut=shortcut,
                    if_first=(block_num == i == 0),
                    dilation=dilation_rate,)
                )
            shortcut = True
        return nn.Sequential(*layers)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.pretrained:
            if self.layers == 18:
                state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet18'])
            elif self.layers == 34:
                state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet34'])
            elif self.layers == 50:
                state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet50'])
            elif self.layers == 101:
                state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet101'])
            elif self.layers == 152:
                state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet152'])
            self.load_state_dict(state_dict, strict=False)






@manager.BACKBONES.add_component
def ResNet18_vd(**args):
    model = ResNet_vd(layers=18, **args)
    return model

@manager.BACKBONES.add_component
def ResNet34_vd(**args):
    model = ResNet_vd(layers=34, **args)
    return model


@manager.BACKBONES.add_component
def ResNet50_vd(**args):
    model = ResNet_vd(layers=50, **args)
    return model


@manager.BACKBONES.add_component
def ResNet101_vd(**args):
    model = ResNet_vd(layers=101, **args)
    return model

if __name__ == "__main__":
    model = ResNet101_vd()
    data = torch.randn([1, 3, 256, 256])
    print(model(data)[0].shape)
