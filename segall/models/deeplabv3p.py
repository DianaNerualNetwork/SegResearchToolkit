import torch
import torch.nn as nn
import torch.nn.functional as F

from segall.cvlibs import manager
from segall.models import layers
from segall.utils import utils

__all__=["Deeplabv3p"]

@manager.MODELS.add_component
class Deeplabv3p(nn.Module):
    def __init__(self, num_classes, 
                 backbone,
                 backbone_indices=(0, 3),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256,
                 align_corners=False,
                 pretrained=None,
                 **kwargs) -> None:
        super(Deeplabv3p,self).__init__()
        # TODO : multi-gpu training
        device="cuda"   if torch.cuda.is_available() else "cpu"
        self.backbone=backbone.to(device)
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head= DeepLabv3pHead(
            num_classes,
            backbone_indices,
            backbone_channels,
            aspp_ratios,
            aspp_out_channels,
            align_corners,
            )
        self.align_corners = align_corners
        self.pretrained=pretrained
        

    def forward(self,x):
        feat_list=self.backbone(x)
        logit_list = self.head(feat_list)
        
        ori_shape = x.shape[2:]
        
        return [
            F.interpolate(
                logit,
                ori_shape,
                mode='bilinear',
                align_corners=self.align_corners,
               ) for logit in logit_list
        ]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class DeepLabv3pHead(nn.Module):
    def __init__(self,num_classes,
                 backbone_indices,
                 backbone_channels,
                 aspp_ratios,
                 aspp_out_channels,
                 align_corners, **kwargs):
        super().__init__()
        self.aspp = layers.ASPPModule(
            aspp_ratios,
            backbone_channels[1],
            aspp_out_channels,
            align_corners,
            use_sep_conv=True,
            image_pooling=True,
           )
        self.decoder = Decoder(
            num_classes,
            backbone_channels[0],
            align_corners,
            )
        self.backbone_indices = backbone_indices

    def forward(self, feat_list):
        logit_list = []
        low_level_feat = feat_list[self.backbone_indices[0]]
        x = feat_list[self.backbone_indices[1]]
        x = self.aspp(x)
        logit = self.decoder(x, low_level_feat)
        logit_list.append(logit)

        return logit_list

class Decoder(nn.Module):
    """
    Decoder module of DeepLabV3P model

    Args:
        num_classes (int): The number of classes.
        in_channels (int): The number of input channels in decoder module.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 align_corners,
                 ):
        super(Decoder, self).__init__()

        
        self.conv_bn_relu1 = layers.ConvBNReLU(
            in_channels=in_channels,
            out_channels=48,
            kernel_size=1,
            )

        self.conv_bn_relu2 = layers.SeparableConvBNReLU(
            in_channels=304,
            out_channels=256,
            kernel_size=3,
            padding=1,
            )
        self.conv_bn_relu3 = layers.SeparableConvBNReLU(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1,
            )
        self.conv = nn.Conv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
            )

        self.align_corners = align_corners

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv_bn_relu1(low_level_feat)
     
        low_level_shape = low_level_feat.shape[-2:]
        axis = 1
        
        x = F.interpolate(
            x,
            low_level_shape,
            mode='bilinear',
            align_corners=self.align_corners,
            )
        x = torch.cat([x, low_level_feat], axis=axis)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.conv(x)
        return x
