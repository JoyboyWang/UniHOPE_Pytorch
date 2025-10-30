import torch.nn as nn
import torch
from model.h2onet.conv.spiralconv import SpiralConv
from model.mob_recon.models.transformer import *

# Init model weights
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Conv1d:
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.BatchNorm2d or type(m) == nn.BatchNorm1d:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
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
        trans (tuple): upsample indices and valus
        dim (int, optional): upsample axis. Defaults to 1.

    Returns:
        tensor: upsampled tensor, BxN"xD
    """
    row, col, value = trans[0].to(x.device), trans[1].to(x.device), trans[2].to(x.device)
    value = value.unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out2 = torch.zeros(x.size(0), row.size(0) // 3, x.size(-1)).to(x.device)
    idx = row.unsqueeze(0).unsqueeze(-1).expand_as(out)
    out2 = torch.scatter_add(out2, dim, idx, out)
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
        out = Pool(x, up_transform)
        out = self.relu(self.conv(out))
        return out


# Advanced modules
class H2ONet_GlobRotReg(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Linear(512, 256),
                                      nn.ReLU(inplace=True), nn.Linear(256, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        x = j_x + r_x
        kd_dict.update({"rot_reg_latent": x.clone()})

        B, C = x.size(0), x.size(1)
        x = x.view(B, C, -1)  # (B, C, HW)

        x = self.conv_block(x)  # (B, C, HW)
        kd_dict.update({"rot_reg_latent_conv": x.clone()})   # (B, 256, 16)

        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        kd_dict.update({"rot_reg_latent_pre": x.clone()})  # (B, 1024)

        pred_rot = self.fc_block(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot



# add more linear layers in fc_block based on H2ONet_GlobRotReg
class H2ONet_GlobRotReg_2(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_2, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        x = j_x + r_x
        kd_dict.update({"rot_reg_latent": x.clone()})

        B, C = x.size(0), x.size(1)
        x = x.view(B, C, -1)  # (B, C, HW)
        x = self.conv_block(x)  # (B, C, HW)
        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot


# concat j_latent and rot_latent and conv fusion based on H2ONet_GlobRotReg_2
class H2ONet_GlobRotReg_3(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_3, self).__init__()
        self.fusion_block = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        x = torch.cat([j_x, r_x], dim=1)  # torch.Size([B, 2048, 4, 4])
        x = self.fusion_block(x)  # torch.Size([B, 1024, 4, 4])

        kd_dict.update({"rot_reg_latent": x.clone()})

        B, C = x.size(0), x.size(1)
        x = x.view(B, C, -1)  # (B, C, HW)
        x = self.conv_block(x)  # (B, C, HW)
        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot

# cross attention between j_latent and rot_latent based on H2ONet_GlobRotReg_2
class H2ONet_GlobRotReg_4(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_4, self).__init__()
        self.fusion_block = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        B, C = j_x.size(0), j_x.size(1)

        j_x = j_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        r_x = r_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        x, _ = self.fusion_block(j_x, r_x, r_x)  # torch.Size([B, 16, 1024])
        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])

        rot_reg_latent = x.view(B, C, 4, 4)  # torch.Size([B, 1024, 4, 4])
        kd_dict.update({"rot_reg_latent": rot_reg_latent})

        x = self.conv_block(x)  # (B, C, HW)
        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot

# Use EncoderLayer to fuse j_latent and rot_latent based on H2ONet_GlobRotReg_2
class H2ONet_GlobRotReg_5(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_5, self).__init__()
        self.fusion_block = EncoderCrsAttnLayer(
            d_model=1024, 
            d_inner=2048,
            n_head=8,
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        B, C = j_x.size(0), j_x.size(1)

        j_x = j_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        r_x = r_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        x, _ = self.fusion_block(j_x, r_x)  # torch.Size([B, 16, 1024])
        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])

        rot_reg_latent = x.view(B, C, 4, 4)  # torch.Size([B, 1024, 4, 4])
        kd_dict.update({"rot_reg_latent": rot_reg_latent})

        x = self.conv_block(x)  # (B, C, HW)
        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot

# use self-attention(MultiHeadAttention) before conv_block based on H2ONet_GlobRotReg_2
class H2ONet_GlobRotReg_6(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_6, self).__init__()
        # self.self_attn = MultiHeadAttention(
        #     n_head=8, 
        #     d_model=1024, 
        #     d_k=128, 
        #     d_v=128, 
        #     dropout=0.1
        # )
        self.self_attn = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        x = j_x + r_x # torch.Size([B, 1024, 4, 4])
        kd_dict.update({"rot_reg_latent": x.clone()})

        B, C = x.size(0), x.size(1)
        x = x.view(B, C, -1)  # (B, C, HW)  # torch.Size([B, 1024, 16])

        x = x.permute(0, 2, 1)
        x, _ = self.self_attn(x, x, x)  # torch.Size([B, 16, 1024])
        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])
        
        kd_dict.update({"rot_reg_latent_flt": x.clone()})

        x = self.conv_block(x)  # (B, C, HW)
        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot

# use self-attention(EncoderCrsAttnLayer) before conv_block based on H2ONet_GlobRotReg_2
class H2ONet_GlobRotReg_7(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_7, self).__init__()
        self.self_attn = EncoderCrsAttnLayer(
            d_model=1024, 
            d_inner=2048,
            n_head=8,
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        x = j_x + r_x # torch.Size([B, 1024, 4, 4])
        kd_dict.update({"rot_reg_latent": x.clone()})

        B, C = x.size(0), x.size(1)
        x = x.view(B, C, -1)  # (B, C, HW)  # torch.Size([B, 1024, 16])

        x = x.permute(0, 2, 1)
        x, _ = self.self_attn(x, x)  # torch.Size([B, 16, 1024])
        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])

        x = self.conv_block(x)  # (B, C, HW)
        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot

# concat j_latent and rot_latent and conv fusion add to rot_latent based on H2ONet_GlobRotReg_2
class H2ONet_GlobRotReg_8(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_8, self).__init__()
        self.fusion_block = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        fx = torch.cat([j_x, r_x], dim=1)  # torch.Size([B, 2048, 4, 4])
        fx = self.fusion_block(fx)  # torch.Size([B, 1024, 4, 4])
        x = r_x + fx  # torch.Size([B, 1024, 4, 4])

        kd_dict.update({"rot_reg_latent": x.clone()})

        B, C = x.size(0), x.size(1)
        x = x.view(B, C, -1)  # (B, C, HW)
        x = self.conv_block(x)  # (B, C, HW)
        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot

# cross attention between j_latent and rot_latent add to rot_latent based on H2ONet_GlobRotReg_2
class H2ONet_GlobRotReg_9(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_9, self).__init__()
        self.fusion_block = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        B, C = j_x.size(0), j_x.size(1)

        j_x = j_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        r_x = r_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        x, _ = self.fusion_block(j_x, r_x, r_x)  # torch.Size([B, 16, 1024])

        x = r_x + x  # torch.Size([B, 16, 1024])
        
        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])

        rot_reg_latent = x.view(B, C, 4, 4)  # torch.Size([B, 1024, 4, 4])
        kd_dict.update({"rot_reg_latent": rot_reg_latent})

        x = self.conv_block(x)  # (B, C, HW)
        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot

# use self-attention(MultiHeadAttention) before conv_block based on H2ONet_GlobRotReg_2
# n_head=8, 
# d_model=1024, 
# d_k=128, 
# d_v=128, 
# dropout=0.1
class H2ONet_GlobRotReg_10(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_10, self).__init__()
        self.self_attn = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=128, 
            d_v=128, 
            dropout=0.1
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        x = j_x + r_x # torch.Size([B, 1024, 4, 4])
        kd_dict.update({"rot_reg_latent": x.clone()})

        B, C = x.size(0), x.size(1)
        x = x.view(B, C, -1)  # (B, C, HW)  # torch.Size([B, 1024, 16])

        x = x.permute(0, 2, 1)
        x, _ = self.self_attn(x, x, x)  # torch.Size([B, 16, 1024])
        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])

        x = self.conv_block(x)  # (B, C, HW)
        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot

# use self-attention(MultiHeadAttention) before conv_block based on H2ONet_GlobRotReg_2
# n_head=4, 
# d_model=1024, 
# d_k=256, 
# d_v=256, 
# dropout=0
class H2ONet_GlobRotReg_11(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_11, self).__init__()
        self.self_attn = MultiHeadAttention(
            n_head=4, 
            d_model=1024, 
            d_k=256, 
            d_v=256, 
            dropout=0
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        x = j_x + r_x # torch.Size([B, 1024, 4, 4])
        kd_dict.update({"rot_reg_latent": x.clone()})

        B, C = x.size(0), x.size(1)
        x = x.view(B, C, -1)  # (B, C, HW)  # torch.Size([B, 1024, 16])

        x = x.permute(0, 2, 1)
        x, _ = self.self_attn(x, x, x)  # torch.Size([B, 16, 1024])
        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])

        x = self.conv_block(x)  # (B, C, HW)
        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot


# self-attention on j_latent, self-attention on rot_latent and cross attention based on H2ONet_GlobRotReg_2
class H2ONet_GlobRotReg_12(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_12, self).__init__()
        self.self_attn_1 = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.self_attn_2 = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )
        
        self.crs_attn = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        B, C = j_x.size(0), j_x.size(1)

        j_x = j_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        r_x = r_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])

        j_x_slf, _ = self.self_attn_1(j_x, j_x, j_x)  # torch.Size([B, 16, 1024])
        r_x_slf, _ = self.self_attn_2(r_x, r_x, r_x)  # torch.Size([B, 16, 1024])

        x, _ = self.crs_attn(j_x_slf, r_x_slf, r_x_slf)  # torch.Size([B, 16, 1024])

        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])

        rot_reg_latent = x.view(B, C, 4, 4)  # torch.Size([B, 1024, 4, 4])
        kd_dict.update({"rot_reg_latent": rot_reg_latent})

        x = self.conv_block(x)  # (B, C, HW)
        kd_dict.update({"rot_reg_latent_conv": x.clone()})   # (B, 256, 16)

        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        kd_dict.update({"rot_reg_latent_pre": x.clone()})  # (B, 1024)

        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot


# self-attention on j_latent, self-attention on rot_latent and cross attention based on H2ONet_GlobRotReg_2
class H2ONet_GlobRotReg_12_test_time(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_12_test_time, self).__init__()
        self.self_attn_1 = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.self_attn_2 = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )
        
        self.crs_attn = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        B, C = j_x.size(0), j_x.size(1)

        j_x = j_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        r_x = r_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])

        j_x_slf, _ = self.self_attn_1(j_x, j_x, j_x)  # torch.Size([B, 16, 1024])
        r_x_slf, _ = self.self_attn_2(r_x, r_x, r_x)  # torch.Size([B, 16, 1024])

        x, _ = self.crs_attn(j_x_slf, r_x_slf, r_x_slf)  # torch.Size([B, 16, 1024])
        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])
        x = self.conv_block(x)  # (B, C, HW)
        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        return pred_rot
        

# add root joint feature to j_latent and rot_latent based on H2ONet_GlobRotReg_2
class H2ONet_GlobRotReg_13(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_13, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x, root_joint_feat):
        kd_dict = {}

        x = j_x + r_x
        kd_dict.update({"rot_reg_latent": x.clone()})

        B, C = x.size(0), x.size(1)
        x = x.view(B, C, -1)  # (B, C, HW)  # torch.Size([B, 1024, 16])
        x = self.conv_block(x)  # (B, C, HW)  # torch.Size([B, 256, 16])

        x = x + root_joint_feat.permute(0, 2, 1)  # torch.Size([B, 256, 16])

        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot


# add root joint feature to rot_latent based on H2ONet_GlobRotReg_2
class H2ONet_GlobRotReg_14(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_14, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, r_x, root_joint_feat):
        kd_dict = {}

        x = r_x

        B, C = x.size(0), x.size(1)
        x = x.view(B, C, -1)  # (B, C, HW)  # torch.Size([B, 1024, 16])
        x = self.conv_block(x)  # (B, C, HW)  # torch.Size([B, 256, 16])

        x = x + root_joint_feat.permute(0, 2, 1)  # torch.Size([B, 256, 16])

        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot


# add image feature to rot_latent based on H2ONet_GlobRotReg_2
class H2ONet_GlobRotReg_15(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_15, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, r_x, latent):
        kd_dict = {}

        x = r_x

        B, C = x.size(0), x.size(1)
        x = x.view(B, C, -1)  # (B, C, HW)  # torch.Size([B, 1024, 16])
        x = self.conv_block(x)  # (B, C, HW)  # torch.Size([B, 256, 16])

        x = x + latent.view(B, 256, -1)  # torch.Size([B, 256, 16])

        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot


# add image feature to to j_latent and rot_latent based on H2ONet_GlobRotReg_2
class H2ONet_GlobRotReg_16(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_16, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x, latent):
        kd_dict = {}

        x = j_x + r_x

        B, C = x.size(0), x.size(1)
        x = x.view(B, C, -1)  # (B, C, HW)  # torch.Size([B, 1024, 16])
        x = self.conv_block(x)  # (B, C, HW)  # torch.Size([B, 256, 16])

        x = x + latent.view(B, 256, -1)  # torch.Size([B, 256, 16])

        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot


# self-attention(EncoderSlfAttnLayer) on j_latent, self-attention(EncoderSlfAttnLayer) on rot_latent and cross attention((EncoderCrsAttnLayer)) based on H2ONet_GlobRotReg_2
class H2ONet_GlobRotReg_17(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_17, self).__init__()
        self.self_attn_1 = EncoderSlfAttnLayer(
            n_head=8, 
            d_model=1024, 
            d_inner=1024,
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.self_attn_2 = EncoderSlfAttnLayer(
            n_head=8, 
            d_model=1024, 
            d_inner=1024,
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )
        
        self.crs_attn = EncoderCrsAttnLayer(
            n_head=8, 
            d_model=1024, 
            d_inner=1024,
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        B, C = j_x.size(0), j_x.size(1)

        j_x = j_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        r_x = r_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])

        j_x_slf, _ = self.self_attn_1(j_x)  # torch.Size([B, 16, 1024])
        r_x_slf, _ = self.self_attn_2(r_x)  # torch.Size([B, 16, 1024])

        x, _ = self.crs_attn(j_x_slf, r_x_slf)  # torch.Size([B, 16, 1024])

        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])

        rot_reg_latent = x.view(B, C, 4, 4)  # torch.Size([B, 1024, 4, 4])
        kd_dict.update({"rot_reg_latent": rot_reg_latent})

        x = self.conv_block(x)  # (B, C, HW)
        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot


# add feature mapping based on H2ONet_GlobRotReg_12
class H2ONet_GlobRotReg_18(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_18, self).__init__()
        self.self_attn_1 = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.self_attn_2 = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )
        
        self.crs_attn = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.fusion_block = nn.Sequential(
            nn.Conv1d(1024, 1024, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        B, C = j_x.size(0), j_x.size(1)

        j_x = j_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        r_x = r_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])

        j_x_slf, _ = self.self_attn_1(j_x, j_x, j_x)  # torch.Size([B, 16, 1024])
        r_x_slf, _ = self.self_attn_2(r_x, r_x, r_x)  # torch.Size([B, 16, 1024])

        x, _ = self.crs_attn(j_x_slf, r_x_slf, r_x_slf)  # torch.Size([B, 16, 1024])

        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])

        x = self.fusion_block(x)  # torch.Size([B, 1024, 16])

        rot_reg_latent = x.view(B, C, 4, 4)  # torch.Size([B, 1024, 4, 4])
        kd_dict.update({"rot_reg_latent": rot_reg_latent})

        x = self.conv_block(x)  # (B, C, HW)
        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot


# hyperparamter setting 2 based on H2ONet_GlobRotReg_12
class H2ONet_GlobRotReg_19(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_19, self).__init__()
        self.self_attn_1 = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=256, 
            d_v=256, 
            dropout=0
        )

        self.self_attn_2 = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=256, 
            d_v=256, 
            dropout=0
        )
        
        self.crs_attn = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=256, 
            d_v=256, 
            dropout=0
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        B, C = j_x.size(0), j_x.size(1)

        j_x = j_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        r_x = r_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])

        j_x_slf, _ = self.self_attn_1(j_x, j_x, j_x)  # torch.Size([B, 16, 1024])
        r_x_slf, _ = self.self_attn_2(r_x, r_x, r_x)  # torch.Size([B, 16, 1024])

        x, _ = self.crs_attn(j_x_slf, r_x_slf, r_x_slf)  # torch.Size([B, 16, 1024])

        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])

        rot_reg_latent = x.view(B, C, 4, 4)  # torch.Size([B, 1024, 4, 4])
        kd_dict.update({"rot_reg_latent": rot_reg_latent})

        x = self.conv_block(x)  # (B, C, HW)
        kd_dict.update({"rot_reg_latent_conv": x.clone()})   # (B, 256, 16)

        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        kd_dict.update({"rot_reg_latent_pre": x.clone()})  # (B, 1024)

        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot

# hyperparamter setting 3 based on H2ONet_GlobRotReg_12
class H2ONet_GlobRotReg_20(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_20, self).__init__()
        self.self_attn_1 = MultiHeadAttention(
            n_head=16, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.self_attn_2 = MultiHeadAttention(
            n_head=16, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )
        
        self.crs_attn = MultiHeadAttention(
            n_head=16, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        B, C = j_x.size(0), j_x.size(1)

        j_x = j_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        r_x = r_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])

        j_x_slf, _ = self.self_attn_1(j_x, j_x, j_x)  # torch.Size([B, 16, 1024])
        r_x_slf, _ = self.self_attn_2(r_x, r_x, r_x)  # torch.Size([B, 16, 1024])

        x, _ = self.crs_attn(j_x_slf, r_x_slf, r_x_slf)  # torch.Size([B, 16, 1024])

        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])

        rot_reg_latent = x.view(B, C, 4, 4)  # torch.Size([B, 1024, 4, 4])
        kd_dict.update({"rot_reg_latent": rot_reg_latent})

        x = self.conv_block(x)  # (B, C, HW)
        kd_dict.update({"rot_reg_latent_conv": x.clone()})   # (B, 256, 16)

        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        kd_dict.update({"rot_reg_latent_pre": x.clone()})  # (B, 1024)

        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot


# hyperparamter setting 4 based on H2ONet_GlobRotReg_12
class H2ONet_GlobRotReg_21(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_21, self).__init__()
        self.self_attn_1 = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.2
        )

        self.self_attn_2 = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.2
        )
        
        self.crs_attn = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.2
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        B, C = j_x.size(0), j_x.size(1)

        j_x = j_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        r_x = r_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])

        j_x_slf, _ = self.self_attn_1(j_x, j_x, j_x)  # torch.Size([B, 16, 1024])
        r_x_slf, _ = self.self_attn_2(r_x, r_x, r_x)  # torch.Size([B, 16, 1024])

        x, _ = self.crs_attn(j_x_slf, r_x_slf, r_x_slf)  # torch.Size([B, 16, 1024])

        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])

        rot_reg_latent = x.view(B, C, 4, 4)  # torch.Size([B, 1024, 4, 4])
        kd_dict.update({"rot_reg_latent": rot_reg_latent})

        x = self.conv_block(x)  # (B, C, HW)
        kd_dict.update({"rot_reg_latent_conv": x.clone()})   # (B, 256, 16)

        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        kd_dict.update({"rot_reg_latent_pre": x.clone()})  # (B, 1024)

        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot


# Feature mapping based on H2ONet_GlobRotReg_21
class H2ONet_GlobRotReg_22(nn.Module):

    def __init__(self, return_kd_dict=False):
        super(H2ONet_GlobRotReg_22, self).__init__()
        self.self_attn_1 = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.2
        )

        self.self_attn_2 = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.2
        )
        
        self.crs_attn = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.2
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))
        
        
        self.fusion_block = nn.Sequential(
            nn.Conv1d(1024, 1024, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.return_kd_dict = return_kd_dict

    def forward(self, j_x, r_x):
        kd_dict = {}

        B, C = j_x.size(0), j_x.size(1)

        j_x = j_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        r_x = r_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])

        j_x_slf, _ = self.self_attn_1(j_x, j_x, j_x)  # torch.Size([B, 16, 1024])
        r_x_slf, _ = self.self_attn_2(r_x, r_x, r_x)  # torch.Size([B, 16, 1024])

        x, _ = self.crs_attn(j_x_slf, r_x_slf, r_x_slf)  # torch.Size([B, 16, 1024])

        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])

        x = self.fusion_block(x)  # torch.Size([B, 1024, 16])

        rot_reg_latent = x.view(B, C, 4, 4)  # torch.Size([B, 1024, 4, 4])
        kd_dict.update({"rot_reg_latent": rot_reg_latent})

        x = self.conv_block(x)  # (B, C, HW)
        kd_dict.update({"rot_reg_latent_conv": x.clone()})   # (B, 256, 16)


        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        kd_dict.update({"rot_reg_latent_pre": x.clone()})  # (B, 1024)

        pred_rot = self.fc_block_1(x)

        if self.return_kd_dict:
            return pred_rot, kd_dict
        else:
            return pred_rot



class H2ONet_Decoder(nn.Module):

    def __init__(self, cfg, latent_size, out_channels, spiral_indices, up_transform, uv_channel, meshconv=SpiralConv):
        """Init a 3D decoding with sprial convolution

        Args:
            latent_size (int): feature dim of backbone feature
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            uv_channel (int): amount of 2D landmark 
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(H2ONet_Decoder, self).__init__()
        self.cfg = cfg
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
        self.upsample = nn.Parameter(torch.ones([self.num_vert[-1], self.uv_channel]) * 0.01, requires_grad=True)
        self.rot_reg = H2ONet_GlobRotReg()
        self.init_weights()

    def init_weights(self):
        self.rot_reg.apply(init_weights)

    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, uv, x, j_mid, r_mid):
        pred_glob_rot = self.rot_reg(j_mid.detach(), r_mid)
        uv = torch.clamp((uv - 0.5) * 2, -1, 1)
        x = self.de_layer_conv(x)
        x = self.index(x, uv).permute(0, 2, 1)  # (B, N, C)
        x = torch.bmm(self.upsample.repeat(x.size(0), 1, 1).to(x.device), x)
        num_features = len(self.de_layer)
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])

        pred = self.head(x)

        return pred, pred_glob_rot
