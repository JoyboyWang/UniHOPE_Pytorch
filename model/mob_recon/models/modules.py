# Copyright (c) Xingyu Chen. All Rights Reserved.
"""
 * @file modules.py
 * @author chenxingyu (chenxy.sean@gmail.com)
 * @brief Modules composing MobRecon
 * @version 0.1
 * @date 2022-04-28
 * 
 * @copyright Copyright (c) 2022 chenxingyu
 * 
"""
import torch.nn as nn
import torch
from model.mob_recon.conv.spiralconv import SpiralConv


from collections import OrderedDict


# Basic modules


# Init model weights
def init_weights(m):
    # if type(m) == nn.ConvTranspose2d:
    #     nn.init.normal_(m.weight, std=0.001)
    # elif type(m) == nn.Conv2d or type(m) == nn.Conv1d:
    #     nn.init.normal_(m.weight, std=0.001)
    #     nn.init.constant_(m.bias, 0)
    # elif type(m) == nn.BatchNorm2d or type(m) == nn.BatchNorm1d:
    #     nn.init.constant_(m.weight, 1)
    #     nn.init.constant_(m.bias, 0)
    # elif type(m) == nn.Linear:
    #     nn.init.normal_(m.weight, std=0.01)
    #     nn.init.constant_(m.bias, 0)
    if type(m) == nn.Conv2d or type(m) == nn.Conv1d:
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.BatchNorm2d or type(m) == nn.BatchNorm1d:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class Reorg(nn.Module):
    dump_patches = True

    def __init__(self):
        """Reorg layer to re-organize spatial dim and channel dim
        """
        super(Reorg, self).__init__()

    def forward(self, x):
        ss = x.size()
        out = x.view(ss[0], ss[1], ss[2] // 2, 2, ss[3]).view(ss[0], ss[1], ss[2] // 2, 2, ss[3] // 2, 2). \
            permute(0, 1, 3, 5, 2, 4).contiguous().view(ss[0], -1, ss[2] // 2, ss[3] // 2)
        return out


def conv_layer(channel_in, channel_out, ks=1, stride=1, padding=0, dilation=1, bias=False, bn=True, relu=True, group=1):
    """Conv block

    Args:
        channel_in (int): input channel size
        channel_out (int): output channel size
        ks (int, optional): kernel size. Defaults to 1.
        stride (int, optional): Defaults to 1.
        padding (int, optional): Defaults to 0.
        dilation (int, optional): Defaults to 1.
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.
        group (int, optional): group conv parameter. Defaults to 1.

    Returns:
        Sequential: a block with bn and relu
    """
    _conv = nn.Conv2d
    sequence = [_conv(channel_in, channel_out, kernel_size=ks, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=group)]
    if bn:
        sequence.append(nn.BatchNorm2d(channel_out))
    if relu:
        sequence.append(nn.ReLU())

    return nn.Sequential(*sequence)


def linear_layer(channel_in, channel_out, bias=False, bn=True, relu=True):
    """Fully connected block

    Args:
        channel_in (int): input channel size
        channel_out (_type_): output channel size
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.

    Returns:
        Sequential: a block with bn and relu
    """
    _linear = nn.Linear
    sequence = [_linear(channel_in, channel_out, bias=bias)]

    if bn:
        sequence.append(nn.BatchNorm1d(channel_out))
    if relu:
        sequence.append(nn.Hardtanh(0, 4))

    return nn.Sequential(*sequence)


class mobile_unit(nn.Module):
    dump_patches = True

    def __init__(self, channel_in, channel_out, stride=1, has_half_out=False, num3x3=1):
        """Init a depth-wise sparable convolution

        Args:
            channel_in (int): input channel size
            channel_out (_type_): output channel size
            stride (int, optional): conv stride. Defaults to 1.
            has_half_out (bool, optional): whether output intermediate result. Defaults to False.
            num3x3 (int, optional): amount of 3x3 conv layer. Defaults to 1.
        """
        super(mobile_unit, self).__init__()
        self.stride = stride
        self.channel_in = channel_in
        self.channel_out = channel_out
        if num3x3 == 1:
            self.conv3x3 = nn.Sequential(conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in), )
        else:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=1, padding=1, group=channel_in),
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        self.conv1x1 = conv_layer(channel_in, channel_out)
        self.has_half_out = has_half_out

    def forward(self, x):
        half_out = self.conv3x3(x)
        out = self.conv1x1(half_out)
        if self.stride == 1 and (self.channel_in == self.channel_out):
            out = out + x
        if self.has_half_out:
            return half_out, out
        else:
            return out


def Pool(x, trans, dim=1):
    """Upsample a mesh

    Args:
        x (tensor): input tensor, BxNxD
        trans (tuple): upsample indices and valus, (shape[294], shape[294], shape[294])
        dim (int, optional): upsample axis. Defaults to 1.

    Returns:
        tensor: upsampled tensor, BxN'xD
    """
    # import ipdb; ipdb.set_trace()
    row, col, value = trans[0].to(x.device), trans[1].to(x.device), trans[2].to(x.device)
    value = value.unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out2 = torch.zeros(x.size(0), row.size(0) // 3, x.size(-1)).to(x.device)
    idx = row.unsqueeze(0).unsqueeze(-1).expand_as(out)
    out2 = torch.scatter_add(out2, dim, idx, out) # idx.shape = out.shape
    return out2


def Pool_down(x, trans, dim=1):
    """Downsample a mesh

    Args:
        x (tensor): input tensor, BxNxD
        trans (tuple): upsample indices and valus, (shape[294], shape[294], shape[294])
        dim (int, optional): upsample axis. Defaults to 1.

    Returns:
        tensor: upsampled tensor, BxN'xD
    """
    # import ipdb; ipdb.set_trace()
    row, col, value = trans[0].to(x.device), trans[1].to(x.device), trans[2].to(x.device)
    value = value.unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out2 = torch.zeros(x.size(0), row.size(0), x.size(-1)).to(x.device)
    idx = row.unsqueeze(0).unsqueeze(-1).expand_as(out)
    out2 = torch.scatter_add(out2, dim, idx, out) # idx.shape = out.shape
    return out2


class SpiralDeblock(nn.Module):

    def __init__(self, in_channels, out_channels, indices, meshconv=SpiralConv):
        """Init a spiral conv block

        Args:
            in_channels (int): input feature dim
            out_channels (int): output feature dim
            indices (tensor): neighbourhood of each hand vertex
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(SpiralDeblock, self).__init__()
        self.conv = meshconv(in_channels, out_channels, indices)
        self.relu = nn.ReLU(inplace=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform) # from torch.Size([32, 98, 256]) -> torch.Size([32, 49, 256])
        out = self.relu(self.conv(out))
        return out


class SpiralDeblock2(nn.Module):

    def __init__(self, in_channels, out_channels, indices, meshconv=SpiralConv):
        """Init a spiral conv block without upsampling layer

        Args:
            in_channels (int): input feature dim
            out_channels (int): output feature dim
            indices (tensor): neighbourhood of each hand vertex
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(SpiralDeblock2, self).__init__()
        self.conv = meshconv(in_channels, out_channels, indices)
        self.relu = nn.ReLU(inplace=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = Pool_down(x, down_transform) # from torch.Size([32, 49, 256]) -> torch.Size([32, 98, 256])
        out = self.relu(self.conv(out))
        return out


# Advanced modules
class Reg2DDecode3D(nn.Module):

    def __init__(self, latent_size, out_channels, spiral_indices, up_transform, uv_channel, meshconv=SpiralConv):
        """Init a 3D decoding with sprial convolution

        Args:
            latent_size (int): feature dim of backbone feature
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            uv_channel (int): amount of 2D landmark 
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(Reg2DDecode3D, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.num_vert = [u[0].size(0) // 3 for u in self.up_transform] + [self.up_transform[-1][0].size(0) // 6]
        self.uv_channel = uv_channel
        self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[-1], 1, bn=False, relu=False)
        self.de_layer = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
            else:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
        self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])
        self.upsample = nn.Parameter(torch.ones([self.num_vert[-1], self.uv_channel]) * 0.01, requires_grad=True) # torch.Size([49, 21])

    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, uv, x):
        uv = torch.clamp((uv - 0.5) * 2, -1, 1)  # normalize
        x = self.de_layer_conv(x) # torch.Size([B, 256, 4, 4])
        x = self.index(x, uv).permute(0, 2, 1) # torch.Size([32, 21, 256])
        x = torch.bmm(self.upsample.repeat(x.size(0), 1, 1).to(x.device), x) # feature lifting: torch.Size([32, 49, 256]). Matrix product: [32, 49, 21] x [32, 21, 256]
        num_features = len(self.de_layer) # 4
        for i, layer in enumerate(self.de_layer):  # upsampling + DSConv
            x = layer(x, self.up_transform[num_features - i - 1])
        pred = self.head(x) # x.shape: torch.Size([32, 778, 32])

        return pred

# class Reg2DDecode3D(nn.Module):

#     def __init__(self, latent_size, out_channels, spiral_indices, up_transform, uv_channel, meshconv=SpiralConv):
#         """Init a 3D decoding with sprial convolution

#         Args:
#             latent_size (int): feature dim of backbone feature
#             out_channels (list): feature dim of each spiral layer
#             spiral_indices (list): neighbourhood of each hand vertex
#             up_transform (list): upsampling matrix of each hand mesh level
#             uv_channel (int): amount of 2D landmark 
#             meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
#         """
#         super(Reg2DDecode3D, self).__init__()
#         self.latent_size = latent_size
#         self.out_channels = out_channels
#         self.spiral_indices = spiral_indices
#         self.up_transform = up_transform
#         self.num_vert = [u[0].size(0) // 3 for u in self.up_transform] + [self.up_transform[-1][0].size(0) // 6]
#         self.uv_channel = uv_channel
#         self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[-1], 1, bn=False, relu=False)
#         self.de_layer = nn.ModuleList()
#         for idx in range(len(self.out_channels)):
#             if idx == 0:
#                 self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
#             else:
#                 self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
#         self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])
#         self.upsample = nn.Parameter(torch.ones([self.num_vert[-1], self.uv_channel]) * 0.01, requires_grad=True) # torch.Size([49, 21])

#     def index(self, feat, uv):
#         uv = uv.unsqueeze(2)  # [B, N, 1, 2]
#         samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
#         return samples[:, :, :, 0]  # [B, C, N]

#     def forward(self, uv, x, batch_input=None):
#         uv = torch.clamp((uv - 0.5) * 2, -1, 1)  # normalize
#         x = self.de_layer_conv(x) # torch.Size([B, 256, 4, 4])
#         x = self.index(x, uv).permute(0, 2, 1) # torch.Size([32, 21, 256])
#         x = torch.bmm(self.upsample.repeat(x.size(0), 1, 1).to(x.device), x) # feature lifting: torch.Size([32, 49, 256]). Matrix product: [32, 49, 21] x [32, 21, 256]
        
#         x_copy = x.clone()
        
#         if batch_input is not None:
#             if "hand" in batch_input:
#                 x = batch_input["hand"]
        
#         num_features = len(self.de_layer) # 4
#         for i, layer in enumerate(self.de_layer):  # upsampling + DSConv
#             x = layer(x, self.up_transform[num_features - i - 1])
#         pred = self.head(x) # x.shape: torch.Size([32, 778, 32])

#         return pred, x_copy

class Reg2DDecode3D_MV(nn.Module):

    def __init__(self, latent_size, out_channels, spiral_indices, up_transform, uv_channel, meshconv=SpiralConv):
        """Init a 3D decoding with sprial convolution

        Args:
            latent_size (int): feature dim of backbone feature
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            uv_channel (int): amount of 2D landmark 
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(Reg2DDecode3D_MV, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.num_vert = [u[0].size(0) // 3 for u in self.up_transform] + [self.up_transform[-1][0].size(0) // 6]
        self.uv_channel = uv_channel
        self.de_layer = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
            else:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
        self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])
        self.upsample = nn.Parameter(torch.ones([self.num_vert[-1], self.uv_channel]) * 0.01, requires_grad=True) # torch.Size([49, 21])

    def forward(self, x):  # x: torch.Size([B, 21, 256])
        x = torch.bmm(self.upsample.repeat(x.size(0), 1, 1).to(x.device), x) # torch.Size([32, 49, 256]). Matrix product: [32, 49, 21] x [32, 21, 256]
        num_features = len(self.de_layer) # 4
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])
        pred = self.head(x) # x.shape: torch.Size([32, 778, 32])

        return pred


class Upsample_MV(nn.Module):

    def __init__(self, out_channels, spiral_indices, up_transform, meshconv=SpiralConv, return_kd_dict=False, init_weights=False):
        """Init a 3D decoding with sprial convolution

        Args:
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(Upsample_MV, self).__init__()
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.de_layer = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
            else:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
        self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])
        self.return_kd_dict = return_kd_dict

        if init_weights is True:
            self.init_weights()

    def init_weights(self):
        self.apply(init_weights)

    def forward(self, x): # x: torch.Size([B, 49, 256])
        kd_dict = {} # OrderedDict()

        num_features = len(self.de_layer) # 4
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])
            kd_dict.update({"up_layer_{}".format(i): x})
        pred = self.head(x)  # x.shape: torch.Size([B, 778, 32])

        if self.return_kd_dict:
            return pred, kd_dict
        else:
            return pred
        


class Upsample_MV_test_time(nn.Module):

    def __init__(self, out_channels, spiral_indices, up_transform, meshconv=SpiralConv, return_kd_dict=False, init_weights=False):
        """Init a 3D decoding with sprial convolution

        Args:
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(Upsample_MV_test_time, self).__init__()
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.de_layer = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
            else:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
        self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])
        self.return_kd_dict = return_kd_dict

        if init_weights is True:
            self.init_weights()

    def init_weights(self):
        self.apply(init_weights)

    def forward(self, x): # x: torch.Size([B, 49, 256])
        num_features = len(self.de_layer) # 4
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])
        pred = self.head(x)  # x.shape: torch.Size([B, 778, 32])

        return pred


class Upsample_MV_MS(nn.Module):

    def __init__(self, out_channels, spiral_indices, up_transform, meshconv=SpiralConv, return_kd_dict=False, init_weights=False):
        """Init a 3D decoding with sprial convolution

        Args:
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(Upsample_MV_MS, self).__init__()
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.de_layer = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
            else:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))

        self.reg_layer = nn.ModuleList()
        self.reg_layer.append(nn.Sequential(  # [B, 256, 49] -> [B, 3, 49]
                            nn.Conv1d(256, 128, 1, bias=True), 
                            nn.Conv1d(128, 64, 1, bias=True),
                            nn.Conv1d(64, 32, 1, bias=True),
                            nn.Conv1d(32, 3, 1, bias=True)))
        
        self.reg_layer.append(nn.Sequential(  # [B, 256, 98] -> [B, 3, 98]
                            nn.Conv1d(256, 128, 1, bias=True), 
                            nn.Conv1d(128, 64, 1, bias=True),
                            nn.Conv1d(64, 32, 1, bias=True),
                            nn.Conv1d(32, 3, 1, bias=True)))
        
        self.reg_layer.append(nn.Sequential(  # [B, 128, 196] -> [B, 3, 196]
                            nn.Conv1d(128, 64, 1, bias=True),
                            nn.Conv1d(64, 32, 1, bias=True),
                            nn.Conv1d(32, 3, 1, bias=True)))
        
        self.reg_layer.append(nn.Sequential(  # [B, 64, 392] -> [B, 3, 392]
                            nn.Conv1d(64, 32, 1, bias=True),
                            nn.Conv1d(32, 3, 1, bias=True)))
        

        self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])
        self.return_kd_dict = return_kd_dict

        if init_weights is True:
            self.init_weights()

    def init_weights(self):
        self.apply(init_weights)

    def forward(self, x): # x: torch.Size([B, 49, 256])
        kd_dict = {} # OrderedDict()
        ms_pred = []

        num_features = len(self.de_layer) # 4
        for i, layer in enumerate(self.de_layer):
            ms_pred.append(self.reg_layer[i](x.permute(0, 2, 1)).permute(0, 2, 1))

            x = layer(x, self.up_transform[num_features - i - 1])
            kd_dict.update({"up_layer_{}".format(i): x})


        pred = self.head(x)  # x.shape: torch.Size([B, 778, 32])


        if self.return_kd_dict:
            return pred, ms_pred, kd_dict
        else:
            return pred, ms_pred
        

class Upsample_MV_2(nn.Module):

    def __init__(self, out_channels, spiral_indices, up_transform, meshconv=SpiralConv, init_weights=False):
        """Init a 3D decoding with sprial convolution

        Args:
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(Upsample_MV_2, self).__init__()
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.de_layer = nn.ModuleList()
        for idx in range(1, len(self.out_channels)):
            self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))

        self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])

        if init_weights is True:
            self.init_weights()

    def init_weights(self):
        self.apply(init_weights)

    def forward(self, x):# x: torch.Size([B, 49, 256])
        num_features = len(self.de_layer) # 3
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])
        pred = self.head(x)  # x.shape: torch.Size([B, 778, 32])

        return pred

# no head layer for predicting the vertex coordinates
class Upsample_MV_3(nn.Module):

    def __init__(self, out_channels, spiral_indices, up_transform, meshconv=SpiralConv, return_kd_dict=False, init_weights=False):
        """Init a 3D decoding with sprial convolution

        Args:
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(Upsample_MV_3, self).__init__()
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.de_layer = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
            else:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
        # self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])
        self.return_kd_dict = return_kd_dict

        if init_weights is True:
            self.init_weights()

    def init_weights(self):
        self.apply(init_weights)

    def forward(self, x):# x: torch.Size([B, 49, 256])
        kd_dict = {}

        num_features = len(self.de_layer) # 4
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])
            kd_dict.update({"up_layer_{}".format(i): x})
        
        # x.shape: torch.Size([B, 778, 32])

        if self.return_kd_dict:
            return x, kd_dict
        else:
            return x
