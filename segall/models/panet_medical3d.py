
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from segall.cvlibs import manager
from segall.models import ConvNorm,ResBlock,ResBottleneck,ScaleUpsample,AttentionConnection

class Backbone(nn.Module):
    """
    Model backbone to extract features
    """
    def __init__(self, input_channels=3, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), **kwargs):
        super().__init__()
        self.nb_filter = channels
        strides=tuple(strides)
        self.strides = strides + (5 - len(strides)) * (1,)
        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck

        if kwargs['norm'] == 'GROUP':
            self.conv0_0 = nn.Sequential(
                nn.Conv3d(input_channels, self.nb_filter[0], kernel_size=3, stride=self.strides[0], padding=1),
                nn.ReLU()
            )
        else:
            self.conv0_0 = res_unit(input_channels, self.nb_filter[0], self.strides[0], **kwargs)
        self.conv1_0 = res_unit(self.nb_filter[0], self.nb_filter[1], self.strides[1], **kwargs)
        self.conv2_0 = res_unit(self.nb_filter[1], self.nb_filter[2], self.strides[2], **kwargs)
        self.conv3_0 = res_unit(self.nb_filter[2], self.nb_filter[3], self.strides[3], **kwargs)
        self.conv4_0 = res_unit(self.nb_filter[3], self.nb_filter[4], self.strides[4], **kwargs)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        return x0_0, x1_0, x2_0, x3_0, x4_0


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, **kwargs):
        super().__init__()
        self.W_g = ConvNorm(F_g, F_int, kernel_size=1, stride=1, activation=False, **kwargs)

        self.W_x = ConvNorm(F_l, F_int, kernel_size=1, stride=2, activation=False, **kwargs)

        self.psi = nn.Sequential(
            ConvNorm(F_int, 1, kernel_size=1, stride=1, activation=False, **kwargs),
            nn.Sigmoid()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * self.upsample(psi)
    
class ParallelDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        assert isinstance(in_channels, (tuple, list)) and len(in_channels) == 3
        self.midchannels = in_channels[0] // 2
        self.conv3_0 = ConvNorm(in_channels[0], self.midchannels, 1, 1, **kwargs)
        self.conv4_0 = ConvNorm(in_channels[1], self.midchannels, 1, 1, **kwargs)
        self.conv5_0 = ConvNorm(in_channels[2], self.midchannels, 1, 1, **kwargs)

        self.conv4_5 = ConvNorm(2 * self.midchannels, self.midchannels, 3, **kwargs)
        self.conv3_4 = ConvNorm(2 * self.midchannels, self.midchannels, 3, **kwargs)

        self.conv_out = nn.Conv3d(3 * self.midchannels, out_channels, kernel_size=1)

    def forward(self, x3, x4, x5):
        # x1 has the fewest channels and largest resolution
        # x3 has the most channels and the smallest resolution
        size = x3.shape[2:]

        # first interpolate three feature maps to the same resolution
        f3 = self.conv3_0(x3)  # (None, midchannels, h3, w3)
        f4 = self.conv4_0(F.interpolate(x4, size, mode='trilinear', align_corners=False))  # (None, midchannels, h3, w3)
        level5 = self.conv5_0(F.interpolate(x5, size, mode='trilinear', align_corners=False))  # (None, midchannels, h3, w3)

        # fuse feature maps
        level4 = self.conv4_5(torch.cat([f4, level5], dim=1))  # (None, midchannels, h3, w3)
        level3 = self.conv3_4(torch.cat([f3, level4], dim=1))  # (None, midchannels, h3, w3)

        fused_out_reduced = torch.cat([level3, level4, level5], dim=1)

        out = self.conv_out(fused_out_reduced)

        return out


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        assert isinstance(in_channels, (tuple, list)) and len(in_channels) == 3
        self.midchannels = in_channels[0] // 2

        self.conv5_4 = ConvNorm(in_channels[2], in_channels[1], 1, 1, **kwargs)
        self.conv4_0 = ConvNorm(in_channels[1], in_channels[1], 3, 1, **kwargs)
        self.conv4_3 = ConvNorm(in_channels[1], in_channels[0], 1, 1, **kwargs)
        self.conv3_0 = ConvNorm(in_channels[0], in_channels[0], 3, 1, **kwargs)

        self.conv_out = nn.Conv3d(in_channels[0], out_channels, kernel_size=1)

    def forward(self, x3, x4, x5):
        # x1 has the fewest channels and largest resolution
        # x3 has the most channels and the smallest resolution
        x5_up = self.conv5_4(F.interpolate(x5, size=x4.shape[2:], mode='trilinear', align_corners=False))
        x4_refine = self.conv4_0(x5_up + x4)
        x4_up = self.conv4_3(F.interpolate(x4_refine, size=x3.shape[2:], mode='trilinear', align_corners=False))
        x3_refine = self.conv3_0(x4_up + x3)

        out = self.conv_out(x3_refine)

        return out
    
class SegmentationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        raise NotImplementedError('Forward method must be implemented before calling it!')


class UNet(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(32, 64, 128, 256, 512),
                 use_deconv=False, strides=(2, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        # upsample for the decoder
        self.up4_3 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-4], scale_factor=strides[-4])

        # deep supervision
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_1 = self.conv3_1(torch.cat([x3, self.up4_3(x4)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, self.up3_2(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, self.up2_1(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, self.up1_0(x1_3)], dim=1))

        
        out = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return out




class AttentionUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, channels=(32, 64, 128, 256, 512),
                 use_deconv=False, strides=(2, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        # upsample for the decoder
        self.up4_3 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-4], scale_factor=strides[-4])

        # deep supervision
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

        self.gate4_3 = AttentionGate(nb_filter[4], nb_filter[3], nb_filter[3])
        self.gate3_2 = AttentionGate(nb_filter[3], nb_filter[2], nb_filter[2])
        self.gate2_1 = AttentionGate(nb_filter[2], nb_filter[1], nb_filter[1])
        self.gate1_0 = AttentionGate(nb_filter[1], nb_filter[0], nb_filter[0])

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_1 = self.conv3_1(torch.cat([self.gate4_3(x4, x3), self.up4_3(x4)], 1))
        x2_2 = self.conv2_2(torch.cat([self.gate3_2(x3_1, x2), self.up3_2(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([self.gate2_1(x2_2, x1), self.up2_1(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([self.gate1_0(x1_3, x0), self.up1_0(x1_3)], 1))

        out = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return [out]


class CascadedUNet(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(32, 64, 128, 256, 512),
                 use_deconv=False, strides=(2, 2, 2, 2), **kwargs):
        super().__init__()
        self.first_stage = UNet(1, input_channels, channels, use_deconv, strides, **kwargs)
        self.second_stage = UNet(num_classes, input_channels, channels, use_deconv, strides, **kwargs)

    def forward(self, x):
        roi = self.first_stage(x)
        roi_ = (torch.sigmoid(roi) > 0.5).float()
        roi_input = x * (1 + roi_)
        fine_seg = self.second_stage(roi_input)

        return [roi,fine_seg]


class EnhancedUNet(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(32, 64, 128, 256, 512),
                 use_deconv=False, strides=(2, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResBlock if nb_filter[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        # upsample for the decoder
        self.up4_3 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-4], scale_factor=strides[-4])

        # deep supervision
        self.convds3 = nn.Conv3d(nb_filter[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(nb_filter[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(nb_filter[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_1 = self.conv3_1(torch.cat([x3, self.up4_3(x4)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, self.up3_2(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, self.up2_1(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, self.up1_0(x1_3)], dim=1))

        
        x3_1 = F.interpolate(self.convds3(x3_1), size=size, mode='trilinear', align_corners=False)
        x2_2 = F.interpolate(self.convds2(x2_2), size=size, mode='trilinear', align_corners=False)
        x1_3 = F.interpolate(self.convds1(x1_3), size=size, mode='trilinear', align_corners=False)
        x0_4 = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return [x3_1,x2_2,x1_3,x0_4]

head_list = ['fcn', 'parallel']
head_map = {'fcn': FCNHead,
            'parallel': ParallelDecoder}

@manager.MODELS.add_component
class PriorAttentionNet(SegmentationNetwork):
    """
    The proposed Prior Attention Network for 3D BraTS segmentation.
    """
    def __init__(self, num_classes, head='fcn', input_channels=3, channels=(32, 64, 128, 256, 512),
                 use_deconv=False, strides=(2, 2, 2, 2), **kwargs):
        super().__init__()
        assert head in head_list
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter

        self.head = head
        self.one_stage = head_map[head](in_channels=nb_filter[2:], out_channels=1)

        res_unit = ResBlock if nb_filter[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        # downsample attention
        self.conv_down = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=2)

        # upsample for the decoder
        self.up4_3 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-4], scale_factor=strides[-4])

        # parameterized skip connection
        self.skip_3 = AttentionConnection()
        self.skip_2 = AttentionConnection()
        self.skip_1 = AttentionConnection()
        self.skip_0 = AttentionConnection()

        # deep supervision
        self.convds3 = nn.Conv3d(nb_filter[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(nb_filter[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(nb_filter[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        attention = self.one_stage(x2, x3, x4)

        act_attention = torch.sigmoid(attention)  # attention shape is the same as x2

        attention_x3 = self.conv_down(act_attention)
        attention3_1 = F.interpolate(attention_x3, size=x3.shape[2:], mode='trilinear', align_corners=False)
        attention2_2 = F.interpolate(act_attention, size=x2.shape[2:], mode='trilinear', align_corners=False)
        attention1_3 = F.interpolate(act_attention, size=x1.shape[2:], mode='trilinear', align_corners=False)
        attention0_4 = F.interpolate(act_attention, size=x0.shape[2:], mode='trilinear', align_corners=False)

        x3_1 = self.conv3_1(torch.cat([self.skip_3(x3, attention3_1), self.up4_3(x4)], dim=1))  # (nb_filter[3], H3, W3, D3)
        x2_2 = self.conv2_2(torch.cat([self.skip_2(x2, attention2_2), self.up3_2(x3_1)], dim=1))  # (nb_filter[2], H2, W2, D2)
        x1_3 = self.conv1_3(torch.cat([self.skip_1(x1, attention1_3), self.up2_1(x2_2)], dim=1))  # (nb_filter[1], H1, W1, D3)
        x0_4 = self.conv0_4(torch.cat([self.skip_0(x0, attention0_4), self.up1_0(x1_3)], dim=1))  # (nb_filter[0], H0, W0, D0)

        
        attention = F.interpolate(attention, size=size, mode='trilinear', align_corners=False)  # intermediate
        x3_1 = F.interpolate(self.convds3(x3_1), size=size, mode='trilinear', align_corners=False)
        x2_2 = F.interpolate(self.convds2(x2_2), size=size, mode='trilinear', align_corners=False)
        x1_3 = F.interpolate(self.convds1(x1_3), size=size, mode='trilinear', align_corners=False)
        x0_4 = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        if self.training:
            
            return [attention,x3_1,x2_2,x1_3,x0_4]
        
        return [x0_4]

