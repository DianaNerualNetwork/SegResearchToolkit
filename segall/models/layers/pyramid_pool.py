
import torch
import torch.nn.functional as F
from torch import nn

from segall.models import layers


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    Args:
        aspp_ratios (tuple): The dilation rate using in ASSP module.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        use_sep_conv (bool, optional): If using separable conv in ASPP module. Default: False.
        image_pooling (bool, optional): If augmented with image-level features. Default: False
    """

    def __init__(self,
                 aspp_ratios,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_sep_conv=False,
                 image_pooling=False,
                 ):
        super().__init__()

        self.align_corners = align_corners
        
        self.aspp_blocks = nn.ModuleList()

        for ratio in aspp_ratios:
            if use_sep_conv and ratio > 1:
                conv_func = layers.SeparableConvBNReLU
            else:
                conv_func = layers.ConvBNReLU

            block = conv_func(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1 if ratio == 1 else 3,
                dilation=ratio,
                padding=0 if ratio == 1 else ratio,
                )
            self.aspp_blocks.append(block)

        out_size = len(self.aspp_blocks)

        if image_pooling:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(
                    output_size=(1, 1)),
                layers.ConvBNReLU(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    ))
            out_size += 1
        self.image_pooling = image_pooling

        self.conv_bn_relu = layers.ConvBNReLU(
            in_channels=out_channels * out_size,
            out_channels=out_channels,
            kernel_size=1,
            )

        self.dropout = nn.Dropout(p=0.1)  # drop rate

    def forward(self, x):
        outputs = []
        
        interpolate_shape = x.shape[2:]
        axis = 1
        
        for block in self.aspp_blocks:
            y = block(x)
            outputs.append(y)

        if self.image_pooling:
            img_avg = self.global_avg_pool(x)
            img_avg = F.interpolate(
                img_avg,
                interpolate_shape,
                mode='bilinear',
                align_corners=self.align_corners,
               )
            outputs.append(img_avg)

        x = torch.cat(outputs, axis=axis)
        x = self.conv_bn_relu(x)
        x = self.dropout(x)

        return x
