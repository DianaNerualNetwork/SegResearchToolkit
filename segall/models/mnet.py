
# import torch
# from torch.nn import Module, AvgPool2d, MaxPool2d, \
#     Conv2dTranspose, Upsample, Conv2d, ModuleList
# from segall.models.layers import ConvBNReLU
# from segall.cvlibs import manager

# @manager.MODELS.add_component
# class MNet(Module):
#     def __init__(self, num_classes=2):
#         super(MNet, self).__init__()

#         self.encoder_head_list = ModuleList([
#             ConvBNReLU(3, channel, 3)
#             for channel in [64, 128, 256]
#         ])

#         encoder_channel_list = [[[3, 32], [32, 32]],
#                                 [[96, 64], [64, 64]],
#                                 [[192, 128], [128, 128]],
#                                 [[384, 256], [256, 256]],
#                                 [[256, 512], [512, 512]]]
#         self.encoder_list = ModuleList([
#             ModuleList([
#                 ConvBNReLU(channels[0], channels[1], 3)
#                 for channels in layer_group_channel_list
#             ])
#             for layer_group_channel_list in encoder_channel_list
#         ])

#         decoder_head_channel_list = [[64, 32],
#                                      [128, 64],
#                                      [256, 128],
#                                      [512, 256]]
#         self.decoder_head_list = ModuleList([
#             Conv2dTranspose(channels[0], channels[1], 2, stride=2)
#             for channels in decoder_head_channel_list
#         ])

#         decoder_channel_list = [[[64, 32], [32, 32], [32, 2]],
#                                 [[128, 64], [64, 64]],
#                                 [[256, 128], [128, 128]],
#                                 [[512, 256], [256, 256]]]
#         self.decoder_list = ModuleList([
#             ModuleList([
#                 ConvBNReLU(channels[0], channels[1], 3)
#                 for channels in layer_group_channel_list
#             ])
#             for layer_group_channel_list in decoder_channel_list
#         ])

#         self.up_sample_list = ModuleList([
#             Upsample(scale_factor=scale_factor)
#             for scale_factor in [8, 4, 2]
#         ])

#         self.out_channel_convert_list = ModuleList([
#             Conv2d(in_channels, num_classes, 1)
#             for in_channels in [256, 128, 64, 2]
#         ])

#         self.avg_pool = AvgPool2d(kernel_size=2)
#         self.down_sample = MaxPool2d(kernel_size=2)

#     def encoder(self, x_list):
#         for encoder_group_layer in self.encoder_list[0]:
#             x_list[0] = encoder_group_layer(x_list[0])

#         for i in range(1, 4):
#             x_list[i] = torch.cat([
#                 self.encoder_head_list[i - 1](x_list[i]),
#                 self.down_sample(x_list[i - 1])
#             ], axis=1)
#             for encoder_group_layer in self.encoder_list[i]:
#                 x_list[i] = encoder_group_layer(x_list[i])

#         x = self.down_sample(x_list[-1])
#         for encoder_group_layer in self.encoder_list[-1]:
#             x = encoder_group_layer(x)

#         return x, x_list

#     def decoder(self, x, shortcut_list):
#         o_list = [x]
#         for i in range(3, -1, -1):
#             o = torch.cat([
#                 shortcut_list[i],
#                 self.decoder_head_list[i](o_list[-1])],
#                 axis=1)
#             for decoder_layer_group in self.decoder_list[i]:
#                 o = decoder_layer_group(o)
#             o_list.append(o)
#         o_list.pop(0)

#         for i in range(3):
#             o_list[i] = self.up_sample_list[i](o_list[i])

#         return o_list

#     def forward(self, x):
#         x_list = [x]
#         for _ in range(3):
#             x_list.append(self.avg_pool(x_list[-1]))

#         x, shortcut_list = self.encoder(x_list)
#         o_list = self.decoder(x, shortcut_list)

#         for i in range(4):
#             o_list[i] = self.out_channel_convert_list[i](o_list[i])

#         o_mean = torch.add_n(o_list) / len(o_list)
#         if self.training:
#             o_list.append(o_mean)
#             return o_list
#         else:
#             return [o_mean]
    
