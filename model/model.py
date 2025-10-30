import logging
import torch
import torch.nn as nn
from pytorch3d import transforms as p3dt
from torchvision import ops

from common.utils.mano import MANO
from model.hand_occ_net import backbone, transformer, regressor
from model.hand_occ_net.mano_head import batch_rodrigues, rot6d2mat, mat2aa, mano_regShapeHead, mano_regRootPoseHead, mano_regFingerPoseHead

from model.mob_recon.utils.read import spiral_tramsform
from model.mob_recon.utils.utils import *
from model.mob_recon.conv.dsconv import DSConv
from model.mob_recon.models.densestack import DenseStack_Backnone, DenseStack_Backnone_2, DenseStack_Backnone_MS
from model.mob_recon.models.modules import Reg2DDecode3D, Reg2DDecode3D_MV, Upsample_MV, Upsample_MV_3, SpiralDeblock, SpiralDeblock2

from model.h2onet.conv.spiralconv import SpiralConv
from model.h2onet.models.densestack import *
from model.h2onet.models.modules import H2ONet_Decoder

from model.mob_recon.models.modules import conv_layer

from model.mob_recon.models.transformer import *

from model.utils import *
from model.h2onet.models.modules import *

logger = logging.getLogger(__name__)
mano = MANO()

import numpy as np
from collections import defaultdict

import copy
import time

import timm

class HandOccNet(nn.Module):

    def __init__(self, cfg):
        super(HandOccNet, self).__init__()
        self.cfg = cfg
        self.backbone = backbone.FPN(pretrained=True)
        self.FIT = transformer.Transformer(injection=True)  # feature injecting transformer
        self.SET = transformer.Transformer(injection=False)  # self enhancing transformer
        self.regressor = regressor.Regressor()
        self.mano_pose_size = 16 * 3
        self.mano_layer = mano.layer

    def init_weights(self):
        self.FIT.apply(init_weights)
        self.SET.apply(init_weights)
        self.regressor.apply(init_weights)

    def forward(self, input):
        p_feats, s_feats = self.backbone(input["img"])  # primary, secondary feats
        feats = self.FIT(s_feats, p_feats)
        feats = self.SET(feats, feats)

        if "mano_pose" in input:
            gt_mano_params = torch.cat([input["mano_pose"], input["mano_shape"]], dim=1)
            gt_mano_shape = gt_mano_params[:, self.mano_pose_size:]
            gt_mano_pose = gt_mano_params[:, :self.mano_pose_size].contiguous()
            gt_mano_pose_rotmat = batch_rodrigues(gt_mano_pose.view(-1, 3)).view(-1, 16, 3, 3)
            gt_verts, gt_joints = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)

            gt_verts /= 1000
            gt_joints /= 1000
        else:
            gt_mano_params = None

        pred_mano_results, preds_joints_img = self.regressor(feats)

        output = {}
        output["pred_joints3d_cam"] = pred_mano_results["joints3d"]
        output["pred_verts3d_cam"] = pred_mano_results["verts3d"]
        output["pred_mano_pose"] = pred_mano_results["mano_pose"]
        output["pred_mano_shape"] = pred_mano_results["mano_shape"]
        output["pred_joints_img"] = preds_joints_img[0]

        # for eval
        output["pred_joints3d_w_gr"] = pred_mano_results["joints3d"]
        output["pred_verts3d_w_gr"] = pred_mano_results["verts3d"]

        if gt_mano_params is not None:
            output["gt_joints3d_cam"] = gt_joints
            output["gt_verts3d_cam"] = gt_verts
            output["gt_mano_pose"] = gt_mano_pose_rotmat
            output["gt_mano_shape"] = gt_mano_shape

            if "val_mano_pose" in input:
                # for eval
                val_gt_verts, val_gt_joints = self.mano_layer(th_pose_coeffs=input["val_mano_pose"], th_betas=input["mano_shape"])
                output["gt_verts3d_w_gr"], output["gt_joints3d_w_gr"] = val_gt_verts / 1000, val_gt_joints / 1000

            # for loss and metric
            output['joints_img'] = input['joints_img']

        return output


class SemiHand(nn.Module):

    def __init__(self, cfg):

        super(SemiHand, self).__init__()
        from model.semi_hand.networks.backbone import FPN
        from model.semi_hand.networks.hand_head import hand_Encoder, hand_regHead
        from model.semi_hand.networks.object_head import obj_regHead, Pose2DLayer
        from model.semi_hand.networks.mano_head import mano_regHead
        from model.semi_hand.networks.CR import Transformer
        from common.utils.manopth.manopth.manolayer import ManoLayer

        roi_res = 32
        joint_nb = 21
        stacks = 1
        channels = 256
        blocks = 1
        transformer_depth = 1
        transformer_head = 1
        mano_layer = None
        mano_neurons = [1024, 512]
        pretrained = True
        self.mano_layer = mano.layer

        self.mano_pose_size = 16 * 3

        self.out_res = roi_res
        # FPN-Res50 backbone
        self.base_net = FPN(pretrained=pretrained)

        # hand head
        self.hand_head = hand_regHead(roi_res=roi_res, joint_nb=joint_nb, stacks=stacks, channels=channels, blocks=blocks)
        # hand encoder
        self.hand_encoder = hand_Encoder(num_heatmap_chan=joint_nb, num_feat_chan=channels, size_input_feature=(roi_res, roi_res))
        # mano branch
        self.mano_branch = mano_regHead()

        # CR blocks
        self.transformer = Transformer(inp_res=roi_res, dim=channels, depth=transformer_depth, num_heads=transformer_head)

    def forward(self, input):

        imgs = input["img"]
        
        if "bbox" in input:
            bbox_hand = input["bbox"]  # follow XH
        else:
            bbox_hand = input["bbox_hand"]  # follow semihand and hflnet
        
        batch = imgs.shape[0]
        # P2 from FPN Network
        P2 = self.base_net(imgs)[0]
        idx_tensor = torch.arange(batch, device=imgs.device).float().view(-1, 1)
        # get roi boxes
        roi_boxes_hand = torch.cat((idx_tensor, bbox_hand), dim=1)
        # 4 here is the downscale size in FPN network(P2)
        x = ops.roi_align(P2, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0, sampling_ratio=-1)  # hand
        # hand forward
        out_hm, encoding, preds_joints = self.hand_head(x)
        mano_encoding = self.hand_encoder(out_hm, encoding)
        pred_mano_results, gt_mano_results = self.mano_branch(mano_encoding, )

        if "mano_pose" in input:
            gt_mano_params = torch.cat([input["mano_pose"], input["mano_shape"]], dim=1)
            gt_mano_shape = gt_mano_params[:, self.mano_pose_size:]
            gt_mano_pose = gt_mano_params[:, :self.mano_pose_size].contiguous()
            gt_mano_pose_rotmat = batch_rodrigues(gt_mano_pose.view(-1, 3)).view(-1, 16, 3, 3)
            gt_verts, gt_joints = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)

            gt_verts /= 1000
            gt_joints /= 1000
        else:
            gt_mano_params = None

        output = {}
        output["pred_joints3d_cam"] = pred_mano_results["joints3d"]
        output["pred_verts3d_cam"] = pred_mano_results["verts3d"]
        output["pred_mano_pose"] = pred_mano_results["mano_pose"]
        output["pred_mano_shape"] = pred_mano_results["mano_shape"]
        output["pred_joints_img"] = preds_joints[0]

        # for eval
        output["pred_joints3d_w_gr"] = pred_mano_results["joints3d"]
        output["pred_verts3d_w_gr"] = pred_mano_results["verts3d"]

        if gt_mano_params is not None:
            output["gt_joints3d_cam"] = gt_joints
            output["gt_verts3d_cam"] = gt_verts
            output["gt_mano_pose"] = gt_mano_pose_rotmat
            output["gt_mano_shape"] = gt_mano_shape

            # for eval
            output["gt_joints3d_w_gr"] = gt_joints
            output["gt_verts3d_w_gr"] = gt_verts

        output['joints_img'] = input['joints_img']

        return output


class MobRecon_DS(nn.Module):

    def __init__(self, cfg):
        super(MobRecon_DS, self).__init__()
        self.cfg = cfg
        self.backbone = DenseStack_Backnone(latent_size=256, kpts_num=21, pretrain=self.cfg.model.pretrain)
        template_fp = "model/mob_recon/template/template.ply"
        transform_fp = "model/mob_recon/template/transform.pkl"
        spiral_indices, _, up_transform, tmp = spiral_tramsform(transform_fp, template_fp, ds_factors=[2, 2, 2, 2], seq_length=[9, 9, 9, 9], dilation=[1, 1, 1, 1])
        for i in range(len(up_transform)):
            up_transform[i] = (*up_transform[i]._indices(), up_transform[i]._values()) # up_transform is pre-computed
        self.decoder3d = Reg2DDecode3D(latent_size=256,
                                       out_channels=[32, 64, 128, 256],
                                       spiral_indices=spiral_indices,
                                       up_transform=up_transform,
                                       uv_channel=21,
                                       meshconv=DSConv)  # original mobrecon
        
        self.mano_pose_size = 16 * 3
        self.mano_layer = mano.layer
        self.mano_joint_reg = torch.from_numpy(mano.joint_regressor)

    def forward(self, input):
        x = input["img"]  # torch.Size([32, 3, 128, 128])
        if x.size(1) == 6: # contrastive learning
            pred3d_list = []
            pred2d_pt_list = []
            for i in range(2):
                latent, pred2d_pt = self.backbone(x[:, 3 * i:3 * i + 3])
                pred3d = self.decoder3d(pred2d_pt, latent)
                pred3d_list.append(pred3d)
                pred2d_pt_list.append(pred2d_pt)
            pred2d_pt = torch.cat(pred2d_pt_list, -1)
            pred3d = torch.cat(pred3d_list, -1)
        else: # go this way
            latent, pred2d_pt = self.backbone(x)
            pred3d = self.decoder3d(pred2d_pt, latent)

        if "mano_pose" in input:
            gt_mano_params = torch.cat([input["mano_pose"], input["mano_shape"]], dim=1)
            gt_mano_shape = gt_mano_params[:, self.mano_pose_size:]
            gt_mano_pose = gt_mano_params[:, :self.mano_pose_size].contiguous()
            gt_verts, gt_joints = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)

            gt_verts /= 1000
            gt_joints /= 1000
        else:
            gt_mano_params = None

        output = {}
        output["pred_verts3d_cam"] = pred3d
        output["pred_joints3d_cam"] = torch.matmul(self.mano_joint_reg.to(pred3d.device), pred3d) # torch.Size([32, 21, 3])

        output["pred_verts3d_w_gr"] = output["pred_verts3d_cam"]
        output["pred_joints3d_w_gr"] = output["pred_joints3d_cam"]

        output["pred_joints_img"] = pred2d_pt # torch.Size([32, 21, 2])

        if gt_mano_params is not None:
            output["gt_verts3d_cam"] = gt_verts
            output["gt_joints3d_cam"] = gt_joints
            if "val_mano_pose" in input:
                # for eval
                val_gt_verts, val_gt_joints = self.mano_layer(th_pose_coeffs=input["val_mano_pose"], th_betas=input["mano_shape"])
                output["gt_verts3d_w_gr"], output["gt_joints3d_w_gr"] = val_gt_verts / 1000, val_gt_joints / 1000

            output['joints_img'] = input['joints_img']

        return output
    

class H2ONet(nn.Module):

    def __init__(self, cfg):
        super(H2ONet, self).__init__()
        self.cfg = cfg
        self.backbone = H2ONet_Backnone(cfg=self.cfg, latent_size=256, kpts_num=21)
        template_fp = "model/h2onet/template/template.ply"
        transform_fp = "model/h2onet/template/transform.pkl"
        spiral_indices, _, up_transform, tmp = spiral_tramsform(transform_fp, template_fp, ds_factors=[2, 2, 2, 2], seq_length=[9, 9, 9, 9], dilation=[1, 1, 1, 1])
        for i in range(len(up_transform)):
            up_transform[i] = (*up_transform[i]._indices(), up_transform[i]._values())
        self.decoder3d = H2ONet_Decoder(cfg=self.cfg,
                                        latent_size=256,
                                        out_channels=[32, 64, 128, 256],
                                        spiral_indices=spiral_indices,
                                        up_transform=up_transform,
                                        uv_channel=21,
                                        meshconv=SpiralConv)
        self.mano_pose_size = 16 * 3
        self.mano_layer = mano.layer
        self.mano_joint_reg = torch.from_numpy(mano.joint_regressor)
        self.mano_joint_reg = torch.nn.Parameter(self.mano_joint_reg)

    def forward(self, input):
        x = input["img"]
        B = x.size(0)
        latent, j_latent, rot_latent, pred2d_pt = self.backbone(x)
        pred_verts_wo_gr, pred_glob_rot = self.decoder3d(pred2d_pt, latent, j_latent, rot_latent)
        pred_joints_wo_gr = torch.matmul(self.mano_joint_reg.to(pred_verts_wo_gr.device), pred_verts_wo_gr)

        pred_glob_rot_mat = p3dt.rotation_6d_to_matrix(pred_glob_rot)

        pred_root_joint_wo_gr = pred_joints_wo_gr[:, 0, None, ...]
        pred_verts_w_gr = torch.matmul(pred_verts_wo_gr.detach() - pred_root_joint_wo_gr.detach(), pred_glob_rot_mat.permute(0, 2, 1))
        pred_verts_w_gr = pred_verts_w_gr + pred_root_joint_wo_gr
        pred_joints_w_gr = torch.matmul(pred_joints_wo_gr.detach() - pred_root_joint_wo_gr.detach(), pred_glob_rot_mat.permute(0, 2, 1)) + pred_root_joint_wo_gr

        if "mano_pose" in input:
            gt_mano_params = torch.cat([input["mano_pose"], input["mano_shape"]], dim=1)
            gt_mano_shape = gt_mano_params[:, self.mano_pose_size:]
            gt_mano_pose = gt_mano_params[:, :self.mano_pose_size].contiguous()
            gt_verts_w_gr, gt_joints_w_gr = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)
            gt_glob_rot = gt_mano_pose[:, :3].clone()
            gt_glob_rot_mat = p3dt.axis_angle_to_matrix(gt_glob_rot)
            gt_mano_pose[:, :3] = 0
            gt_verts_wo_gr, gt_joints_wo_gr = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)

            gt_verts_w_gr /= 1000
            gt_joints_w_gr /= 1000
            gt_verts_wo_gr /= 1000
            gt_joints_wo_gr /= 1000

        else:
            gt_mano_params = None
            # the gt annotations are provided in the input

        output = {}
        # for evaluation purpose
        output["pred_verts3d_cam"] = pred_verts_w_gr
        output["pred_joints3d_cam"] = pred_joints_w_gr

        output["pred_verts3d_wo_gr"] = pred_verts_wo_gr
        output["pred_joints3d_wo_gr"] = pred_joints_wo_gr
        output["pred_verts3d_w_gr"] = pred_verts_w_gr
        output["pred_joints3d_w_gr"] = pred_joints_w_gr
        output["pred_joints_img"] = pred2d_pt
        output["pred_glob_rot"] = pred_glob_rot
        output["pred_glob_rot_mat"] = pred_glob_rot_mat
        if gt_mano_params is not None:
            output["gt_verts3d_cam"] = gt_verts_w_gr
            output["gt_joints3d_cam"] = gt_joints_w_gr
            output["gt_glob_rot"] = gt_glob_rot
            output["gt_glob_rot_mat"] = gt_glob_rot_mat
            output["gt_verts3d_w_gr"] = gt_verts_w_gr
            output["gt_joints3d_w_gr"] = gt_joints_w_gr
            output["gt_verts3d_wo_gr"] = gt_verts_wo_gr
            output["gt_joints3d_wo_gr"] = gt_joints_wo_gr

            if "val_mano_pose" in input:
                # for eval
                val_gt_verts, val_gt_joints = self.mano_layer(th_pose_coeffs=input["val_mano_pose"], th_betas=input["mano_shape"])
                output["gt_verts3d_w_gr"], output["gt_joints3d_w_gr"] = val_gt_verts / 1000, val_gt_joints / 1000
                
            output['joints_img'] = input['joints_img']
            

        return output
 
 
import hiera
from model.simple_hand.modules import MeshHead, AttentionBlock, IdentityBlock, SepConvBlock
from model.simple_hand.losses import mesh_to_joints

class SimpleHand(nn.Module):
    def __init__(self, cfg, pretrained=None):
        super().__init__()
        self.cfg = cfg
        
        model_cfg = cfg.model
        backbone_cfg = model_cfg.BACKBONE

        self.loss_cfg = model_cfg.LOSSES

        if pretrained is None:
            pretrained=backbone_cfg.pretrain

        if "hiera" in backbone_cfg.model_name:
            self.backbone = hiera.__dict__[backbone_cfg.model_name](pretrained=True, checkpoint="mae_in1k",  drop_path_rate=backbone_cfg.drop_path_rate)
            self.is_hiera = True
        else:
            self.backbone = timm.create_model(backbone_cfg.model_name, pretrained=pretrained, drop_path_rate=backbone_cfg.drop_path_rate)
            self.is_hiera = False
            
        self.avg_pool = nn.AvgPool2d((7, 7), 1)            

        uv_cfg = model_cfg.UV_HEAD
        depth_cfg = model_cfg.DEPTH_HEAD

        self.keypoints_2d_head = nn.Linear(uv_cfg.in_features, uv_cfg.out_features)
        # self.depth_head = nn.Linear(depth_cfg['in_features'], depth_cfg['out_features'])
        
        mesh_head_cfg = copy.deepcopy(model_cfg.MESH_HEAD)
        
        block_types_name = mesh_head_cfg.block_types
        block_types = []
        block_map = {
            "attention": AttentionBlock,
            "identity": IdentityBlock,
            "conv": SepConvBlock,
        }
        
        for name in block_types_name:
            block_types.append(block_map[name])
        mesh_head_cfg['block_types'] = block_types
        
        self.mesh_head = MeshHead(**mesh_head_cfg)        
        
        self.mano_pose_size = 16 * 3
        self.mano_layer = mano.layer
        
        # added by yqwang
        self.mano_joint_reg = torch.from_numpy(mano.joint_regressor)


    def infer(self, image):
        if self.is_hiera:
            x, intermediates = self.backbone(image, return_intermediates=True)
            features = intermediates[-1]
            features = features.permute(0, 3, 1, 2).contiguous()
        else:
            features = self.backbone.forward_features(image)
        
        global_feature = self.avg_pool(features).squeeze(-1).squeeze(-1)
        uv = self.keypoints_2d_head(global_feature)     
        # depth = self.depth_head(global_feature)
        
        vertices = self.mesh_head(features, uv)
        # joints = mesh_to_joints(vertices)
        joints = torch.matmul(self.mano_joint_reg.to(vertices.device), vertices) 

        return {
            "uv": uv,
            # "root_depth": depth,
            "pred_joints3d_cam": joints,
            "pred_verts3d_cam": vertices,            
        }


    def forward(self, input):
        """get training loss

        Args:
            inputs (dict): {
                'img': (B, 1, H, W), 
                "uv": [B, 21, 2],
                "xyz": [B,  21, 3],
                "hand_uv_valid": [B, 21],
                "gamma": [B, 1],    

                "vertices": [B, 778, 3],
                "xyz_valid": [B,  21],
                "verts_valid": [B, 1],
                "hand_valid": [B, 1],
            }     
        """
        # img = input['img']
        img = input['img'].clone()  # avoid affecting the input dict
        img -= 0.5
        output_dict = self.infer(img)
        
        if "mano_pose" in input:
            # get gt vertices from mano
            gt_mano_params = torch.cat([input["mano_pose"], input["mano_shape"]], dim=1)
            gt_mano_shape = gt_mano_params[:, self.mano_pose_size:]
            gt_mano_pose = gt_mano_params[:, :self.mano_pose_size].contiguous()
            gt_verts, gt_joints = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)
            gt_verts /= 1000
            gt_joints /= 1000
            
            output_dict.update({
                "gt_joints3d_cam": gt_joints,
                "gt_verts3d_cam": gt_verts,
            })
        
        return output_dict
    
class SemiHandObject(nn.Module):

    def __init__(self, cfg):

        super(SemiHandObject, self).__init__()
        from model.semi_hand.networks.backbone import FPN
        from model.semi_hand.networks.hand_head import hand_Encoder, hand_regHead
        from model.semi_hand.networks.object_head import obj_regHead, Pose2DLayer
        from model.semi_hand.networks.mano_head import mano_regHead
        from model.semi_hand.networks.CR import Transformer
        from common.utils.manopth.manopth.manolayer import ManoLayer

        roi_res = 32
        joint_nb = 21
        stacks = 1
        channels = 256
        blocks = 1
        transformer_depth = 1
        transformer_head = 1  # the default value of their original code
        mano_layer = None
        mano_neurons = [1024, 512]
        pretrained = True
        reg_object = True
        
        self.mano_layer = mano.layer

        self.mano_pose_size = 16 * 3

        self.out_res = roi_res
        # FPN-Res50 backbone
        self.base_net = FPN(pretrained=pretrained)

        # hand head
        self.hand_head = hand_regHead(roi_res=roi_res, joint_nb=joint_nb, stacks=stacks, channels=channels, blocks=blocks)
        # hand encoder
        self.hand_encoder = hand_Encoder(num_heatmap_chan=joint_nb, num_feat_chan=channels, size_input_feature=(roi_res, roi_res))
        # mano branch
        self.mano_branch = mano_regHead()
        
        # object head
        self.reg_object = reg_object
        self.obj_head = obj_regHead(channels=channels, inter_channels=channels//2, joint_nb=joint_nb)
        self.obj_reorgLayer = Pose2DLayer(joint_nb=joint_nb)

        # CR blocks
        self.transformer = Transformer(inp_res=roi_res, dim=channels, depth=transformer_depth, num_heads=transformer_head)

    def forward(self, input):
        imgs = input["img"]
        bbox_hand = input["bbox_hand"]
        bbox_obj = input["bbox_obj"]
        
        batch = imgs.shape[0]
        
        inter_topLeft = torch.max(bbox_hand[:, :2], bbox_obj[:, :2])
        inter_bottomRight = torch.min(bbox_hand[:, 2:], bbox_obj[:, 2:])
        bbox_inter = torch.cat((inter_topLeft, inter_bottomRight), dim=1)
        msk_inter = ((inter_bottomRight-inter_topLeft > 0).sum(dim=1)) == 2

   
        # P2 from FPN Network
        P2 = self.base_net(imgs)[0]
        idx_tensor = torch.arange(batch, device=imgs.device).float().view(-1, 1)
        # get roi boxes
        roi_boxes_hand = torch.cat((idx_tensor, bbox_hand), dim=1)
        # 4 here is the downscale size in FPN network(P2)
        x = ops.roi_align(P2, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0, sampling_ratio=-1)  # hand
        # hand forward
        out_hm, encoding, preds_joints = self.hand_head(x)
        mano_encoding = self.hand_encoder(out_hm, encoding)
        pred_mano_results, gt_mano_results = self.mano_branch(mano_encoding, )
        
        # obj forward
        if self.reg_object:
            roi_boxes_obj = torch.cat((idx_tensor, bbox_obj), dim=1)
            roi_boxes_inter = torch.cat((idx_tensor, bbox_inter), dim=1)

            y = ops.roi_align(P2, roi_boxes_obj, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)  # obj

            z = ops.roi_align(P2, roi_boxes_inter, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)  # intersection
            z = msk_inter[:, None, None, None] * z
            y = self.transformer(y, z)
            out_fm = self.obj_head(y)
            preds_obj = self.obj_reorgLayer(out_fm)
        else:
            preds_obj = None

        if "mano_pose" in input:
            gt_mano_params = torch.cat([input["mano_pose"], input["mano_shape"]], dim=1)
            gt_mano_shape = gt_mano_params[:, self.mano_pose_size:]
            gt_mano_pose = gt_mano_params[:, :self.mano_pose_size].contiguous()
            gt_mano_pose_rotmat = batch_rodrigues(gt_mano_pose.view(-1, 3)).view(-1, 16, 3, 3)
            gt_verts, gt_joints = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)

            gt_verts /= 1000
            gt_joints /= 1000
        else:
            gt_mano_params = None

        output = {}
        output["pred_joints3d_cam"] = pred_mano_results["joints3d"]
        output["pred_verts3d_cam"] = pred_mano_results["verts3d"]
        output["pred_mano_pose"] = pred_mano_results["mano_pose"]
        output["pred_mano_shape"] = pred_mano_results["mano_shape"]
        output["pred_joints_img"] = preds_joints[0]
        output["preds_joints2d"] = preds_joints[0]
        
        if self.training:
            output["preds_obj"] = preds_obj
        else:
            for i in range(len(preds_obj)):
                output["preds_obj_{}".format(i)] = preds_obj[i]
      
        # for eval
        output["pred_joints3d_w_gr"] = pred_mano_results["joints3d"]
        output["pred_verts3d_w_gr"] = pred_mano_results["verts3d"]

        if gt_mano_params is not None:
            output["gt_joints3d_cam"] = gt_joints
            output["gt_verts3d_cam"] = gt_verts
            output["gt_mano_pose"] = gt_mano_pose_rotmat
            output["gt_mano_shape"] = gt_mano_shape

            # for eval
            output["gt_joints3d_w_gr"] = gt_joints
            output["gt_verts3d_w_gr"] = gt_verts

            output['joints_img'] = input['joints_img']

        return output    


from model.kypt_transformer.common.nets.module import BackboneNet, DecoderNet, DecoderNet_big
from model.kypt_transformer.common.nets.transformer import Transformer as Transformer_2
from model.kypt_transformer.common.nets.position_encoding import build_position_encoding
from model.kypt_transformer.common.utils.preprocessing import PeakDetector
from model.kypt_transformer.common.nets.layer import MLP
from model.kypt_transformer.common.utils.misc import get_tgt_mask, get_src_memory_mask
from model.kypt_transformer.common.nets.loss import ManoMesh
import time
    
class KyptTransformer(nn.Module):

    def __init__(self, cfg):
        super(KyptTransformer, self).__init__()
        
        self.cfg = cfg

        # modules
        self.backbone_net = BackboneNet(self.cfg.model)
        
        if self.cfg.model.use_big_decoder:
            self.decoder_net = DecoderNet_big(self.cfg.model)
        else:
            self.decoder_net = DecoderNet(self.cfg.model)
            
        self.transformer = Transformer_2(
            cfg=self.cfg.model,
            d_model=self.cfg.model.hidden_dim,
            dropout=self.cfg.model.dropout,
            nhead=self.cfg.model.nheads,
            dim_feedforward=self.cfg.model.dim_feedforward,
            num_encoder_layers=self.cfg.model.enc_layers,
            num_decoder_layers=self.cfg.model.dec_layers,
            normalize_before=self.cfg.model.pre_norm,
            return_intermediate_dec=True,
        )
        
        self.position_embedding = build_position_encoding(self.cfg.model)

        start = time.time()
        self.peak_detector = PeakDetector(self.cfg.model)
        self.obj_peak_detector = PeakDetector(self.cfg.model, nearest_neighbor_th=5)
        print('Init of peak detector took %f s'%(time.time()-start))


        output_dim = self.cfg.model.hidden_dim

        if self.cfg.model.position_embedding == 'simpleCat':
            output_dim = output_dim - 32


        # MLP for converting concatenated image features to 256-D features
        self.norm1 = nn.LayerNorm(self.cfg.model.mutliscale_dim)
        self.linear1 = MLP(input_dim=self.cfg.model.mutliscale_dim, hidden_dim=[1024, 512, 256],
                           output_dim=output_dim,
                           num_layers=4, is_activation_last=True)
        self.activation = nn.functional.relu

        self.query_embed = nn.Embedding(self.cfg.model.num_queries, self.cfg.model.hidden_dim)
        self.shape_query_embed = None


        # MLPs for keypoint Classification
        if self.cfg.model.has_object:
            self.linear_class = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim,
                                    (42 + 1 + 1) if self.cfg.model.hand_type == 'both' else (21 + 1 + 1),
                                    4)
        else:
            self.linear_class = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim,
                                    (42+1) if self.cfg.model.hand_type=='both' else (21+1),
                                    4)

        # Pose Regression MLPs
        pose_fan_out = 3
        if self.cfg.model.predict_type == 'angles':
            self.linear_pose = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, pose_fan_out, 3)
        elif self.cfg.model.predict_type == 'vectors':
            if not self.cfg.model.predict_2p5d :
                self.linear_joint_vecs =  MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, 3, 3)
            else:
                self.linear_joint_2p5d = {}
                self.linear_joint_2p5d_px = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, self.cfg.model.output_hm_shape[1], 3)
                self.linear_joint_2p5d_py = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, self.cfg.model.output_hm_shape[2], 3)
                self.linear_joint_2p5d_dep = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, self.cfg.model.output_hm_shape[0], 3)
                self.softmax = nn.Softmax(dim=3)

        self.linear_rel_trans = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, 3, 3)

        if self.cfg.model.has_object:
            self.linear_obj_rel_trans = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, 3, 3)
            if self.cfg.model.predict_obj_left_hand_trans:
                self.linear_obj_left_rel_trans = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, 3,
                                                3)
            self.linear_obj_corner_proj = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, 16, 3)
            if self.cfg.model.use_obj_rot_parameterization:
                self.linear_obj_rot = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, 6, 3)
            else:
                self.linear_obj_rot = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, 3, 3)


            if not self.cfg.model.use_big_decoder:
                self.linear1_obj = MLP(input_dim=(512+256), hidden_dim=[512, 256, 256],
                                   output_dim=output_dim,
                                   num_layers=4, is_activation_last=True)
            else:
                self.linear1_obj = MLP(input_dim=3072, hidden_dim=[1024, 512, 256],
                                       output_dim=output_dim,
                                       num_layers=4, is_activation_last=True)



        # MLPs for predicting Hand shape and camera parameters
        if self.cfg.model.use_2D_loss:
            self.linear_shape = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, 10, 3)
            self.linear_cam = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, 3, 3)

        else:
            self.linear_shape = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, 10, 3)


        # MLP for predicting the hand type after the U-Net decoder
        if self.cfg.model.use_bottleneck_hand_type:
            if self.cfg.model.resnet_type >= 50:
                self.linear_bottleneck_hand_type = MLP(2048, 512, 2, 2)
            else:
                self.linear_bottleneck_hand_type = MLP(512, 512, 2, 2)

        if self.cfg.model.hand_type == 'both':
            self.linear_hand_type = MLP(self.cfg.model.hidden_dim, self.cfg.model.hidden_dim, 3, 2)
            self.hand_type_query_embed = None
        else:
            self.hand_type_query_embed = None

        # Freeze batch norm layers
        self.freeze_stages()
        
        
        self.mano_mesh = ManoMesh(self.cfg.model)
        
        self.ih26m_joint_regressor = np.load(self.cfg.model.joint_regr_np_path)
        self.ih26m_joint_regressor = torch.FloatTensor(self.ih26m_joint_regressor).unsqueeze(0)  # 1 x 21 x 778
        
        self.jointsNormalToManoMap = [20,
                                 7,6,5,
                                 11,10,9,
                                 19,18,17,
                                 15,14,13,
                                 3,2,1,
                                 0,4,8,12,16]

    def freeze_stages(self):

        for name, param in self.backbone_net.named_parameters():
            if 'bn' in name:
                param.requires_grad = False
    

    def sample_obj_seg(self, obj_seg):
        total_area = np.sum(obj_seg>self.cfg.model.intensity_th)
        nms_th = np.sqrt(total_area/(np.pi*self.cfg.model.num_obj_samples))
        peaks, peaks_ind_list = self.obj_peak_detector.detect_peaks_nms(obj_seg, self.cfg.model.num_obj_samples,
                                                                        intensity_th=15)#, nms_th)
        return peaks_ind_list

    def get_input_seq(self, joint_heatmap_out, obj_seg_out, feature_pyramid, pos_embed, input, joint_coord, joint_valid, obj_seg_gt, epoch_cnt):
        heatmap_np = joint_heatmap_out.detach().cpu().numpy()
        if self.cfg.model.has_object:
            obj_seg_np = obj_seg_out.detach().cpu().numpy()

        if epoch_cnt<self.cfg.model.num_epochs_gt_peak_locs:
            use_gt_peak_locs = True
        else:
            use_gt_peak_locs = False

        grids = []
        masks = []
        peak_joints_map_batch = []
        normalizer = np.array([self.cfg.model.output_hm_shape[1] - 1, self.cfg.model.output_hm_shape[2] - 1]) / 2
        for ii in range(heatmap_np.shape[0]):
            if use_gt_peak_locs:
                peaks_ind_list = joint_coord[ii,:,[1,0]].cpu().numpy()
                peak_joints_map = np.arange(0, peaks_ind_list.shape[0]) + 1
                mask1 = np.logical_or(peaks_ind_list[:, 0] < 0, peaks_ind_list[:, 1] < 0)
                mask2 = np.logical_or(peaks_ind_list[:, 0] > self.cfg.model.output_hm_shape[1]-1, peaks_ind_list[:, 1] > self.cfg.model.output_hm_shape[2]-1)
                mask = np.logical_not(np.logical_or(mask1, mask2))
                mask = np.logical_and(mask, joint_valid[ii].cpu().numpy())
                peaks_ind_list = peaks_ind_list[mask]
                peak_joints_map = peak_joints_map[mask]

                if self.cfg.model.has_object:
                    obj_peaks_ind_list = self.sample_obj_seg(obj_seg_gt[ii].cpu().numpy())

                    if len(obj_peaks_ind_list) == 0:
                        # print('Found %d object peaks for %s/%s/' % (len(obj_peaks_ind_list),
                        #                                           str(input['seq_id'][ii]),
                        #                                           str(input['frame'][ii])))
                        input['obj_pose_valid'][ii] *= 0

                    if len(obj_peaks_ind_list) > 0:
                        obj_peaks_ind_list = np.stack(obj_peaks_ind_list, axis=0)
                        mask1 = np.logical_or(obj_peaks_ind_list[:, 0] < 0, obj_peaks_ind_list[:, 1] < 0)
                        mask2 = np.logical_or(obj_peaks_ind_list[:, 0] > self.cfg.model.output_hm_shape[1] - 1,
                                              obj_peaks_ind_list[:, 1] > self.cfg.model.output_hm_shape[2] - 1)
                        mask = np.logical_not(np.logical_or(mask1, mask2))
                        obj_peaks_ind_list = obj_peaks_ind_list[mask]

                        obj_peak_map = np.zeros((len(obj_peaks_ind_list))) + self.cfg.model.obj_cls_index
                        peaks_ind_list = np.concatenate([peaks_ind_list, np.array(obj_peaks_ind_list)], axis=0)
                        peak_joints_map = np.concatenate([peak_joints_map, obj_peak_map], axis=0)
            else:
                peaks, peaks_ind_list = self.peak_detector.detect_peaks_nms(heatmap_np[ii],
                                                                            (self.cfg.model.max_num_peaks-self.cfg.model.num_obj_samples) if self.cfg.model.has_object else self.cfg.model.max_num_peaks)
                peak_joints_map = np.zeros((len(peaks_ind_list)), dtype=np.int)+1
                if self.cfg.model.has_object:
                    obj_peaks_ind_list = self.sample_obj_seg(obj_seg_np[ii])
                    if len(obj_peaks_ind_list) > 0:
                        obj_peak_map = np.zeros((len(obj_peaks_ind_list))) + self.cfg.model.obj_cls_index
                        if len(peaks_ind_list) > 0:
                            peaks_ind_list = np.concatenate([peaks_ind_list, np.array(obj_peaks_ind_list)], axis=0)
                            peak_joints_map = np.concatenate([peak_joints_map, obj_peak_map], axis=0)
                        else:
                            peaks_ind_list = np.array(obj_peaks_ind_list)
                            peak_joints_map = obj_peak_map
                    else:
                        # Corner case when the object is heavily occluded
                        # print('Found %d object peaks for %s/%s/' % (len(obj_peaks_ind_list),
                        #                                      str(input['seq_id'][ii]),
                        #                                      str(input['frame'][ii])))
                        input['obj_pose_valid'][ii] *= 0

            if len(peak_joints_map) != len(peaks_ind_list):
                print(len(peak_joints_map), len(peaks_ind_list))
                assert len(peak_joints_map) == len(peaks_ind_list)

            if len(peaks_ind_list) == 0:
                # Corner case when the object and hand is heavily occluded in the image

                # print('Found %d peaks for %s/%s/%s/%s'%(len(peaks_ind_list), str(input['capture'][ii]),
                #                                      str(input['cam'][ii]), str(input['seq_id'][ii]), str(input['frame'][ii])))
                input['mano_valid'][ii] = False
                input['joint_valid'][ii] *= 0
                input['hand_type_valid'][ii] *= 0
                peaks_pixel_locs_normalized = np.tile(np.array([[-1, -1]]), (self.cfg.model.max_num_peaks,1))
                mask = np.ones((self.cfg.model.max_num_peaks), dtype=np.bool)
                peak_joints_map = np.zeros((self.cfg.model.max_num_peaks,), dtype=np.int)
            else:
                peaks_ind_normalized = (np.array(peaks_ind_list) - normalizer) / normalizer
                assert np.sum(peaks_ind_normalized < -1) == 0 and np.sum(peaks_ind_normalized > 1) == 0

                peaks_pixel_locs_normalized = peaks_ind_normalized[:, [1, 0]]  # in pixel coordinates
                mask = np.ones((peaks_pixel_locs_normalized.shape[0],), dtype=np.bool)

                # fill up the empty slots with some dummy values
                if peaks_pixel_locs_normalized.shape[0] < self.cfg.model.max_num_peaks:
                    dummy_peaks = np.tile(np.array([[-1, -1]]), (self.cfg.model.max_num_peaks - peaks_pixel_locs_normalized.shape[0],1))
                    invalid_mask = np.zeros((self.cfg.model.max_num_peaks - peaks_pixel_locs_normalized.shape[0],), dtype=np.bool)
                    peak_joints_map = np.concatenate([peak_joints_map, invalid_mask.astype(np.int)], axis=0)

                    peaks_pixel_locs_normalized = np.concatenate([peaks_pixel_locs_normalized, dummy_peaks], axis=0)
                    mask = np.concatenate([mask, invalid_mask], axis=0)


            grids.append(peaks_pixel_locs_normalized)
            masks.append(mask)
            peak_joints_map_batch.append(peak_joints_map)

        peak_joints_map_batch = torch.from_numpy(np.stack(peak_joints_map_batch, 0)).to(pos_embed.device) # N x max_num_peaks
        grids = np.stack(grids, 0)  # N x max_num_peaks x 2
        grids_unnormalized_np = grids*normalizer[[1,0]] + normalizer[[1,0]] # in pixel coordinates space
        masks_np = np.stack(masks, 0)  # N x max_num_peaks
        masks = torch.from_numpy(masks_np).bool().to(pos_embed.device) # N x max_num_peaks


        # Get the positional embeddings
        positions = nn.functional.grid_sample(pos_embed,
                                              torch.from_numpy(np.expand_dims(grids, 1)).float().to(pos_embed.device),
                                              mode='nearest', align_corners=True).squeeze(2) # N x hidden_dim x max_num_peaks


        # Sample the CNN features
        multiscale_features = []
        grids_tensor = torch.from_numpy(np.expand_dims(grids, 1)).float().to(feature_pyramid[self.cfg.model.mutliscale_layers[0]].device)
        for layer_name in self.cfg.model.mutliscale_layers:
            # N x C x 1 x max_num_peaks
            multiscale_features.append(torch.nn.functional.grid_sample(feature_pyramid[layer_name],
                                                                       grids_tensor,
                                                                         align_corners=True))

        multiscale_features = torch.cat(multiscale_features, dim=1).squeeze(2) # N x C1 x  max_num_peaks
        multiscale_features = multiscale_features.permute(0, 2, 1) # N x max_num_peaks x C1


        if self.cfg.model.has_object:
            input_seq_hands = self.linear1(multiscale_features)  # N x max_num_peaks x hidden_dim

            # For object use CNN features from deeper layers only as they have higer receptive field
            if not self.cfg.model.use_big_decoder:
                input_seq_object = self.linear1_obj(multiscale_features[:,:,-(512+256):])  # N x max_num_peaks x hidden_dim
            else:
                input_seq_object = self.linear1_obj(multiscale_features[:, :, -3072:])  # N x max_num_peaks x hidden_dim

            input_seq = input_seq_hands*(peak_joints_map_batch.unsqueeze(2)!=self.cfg.model.obj_cls_index)\
                        + input_seq_object*(peak_joints_map_batch.unsqueeze(2)==self.cfg.model.obj_cls_index)
            input_seq = input_seq.permute(0, 2, 1) # N x hidden_dim x max_num_peaks
        else:
            input_seq = self.linear1(multiscale_features).permute(0, 2, 1) # N x hidden_dim x max_num_peaks

        return input_seq, masks, positions, grids_unnormalized_np, masks_np, peak_joints_map_batch

    def forward(self, input, epoch_cnt=1e8):  # TODO: if the result is far away from being good, consider epoch
        input_img = input['img']
        input_mask = input['mask']
        batch_size = input_img.shape[0]

        img_feat, enc_skip_conn_layers = self.backbone_net(input_img)
        feature_pyramid, decoder_out = self.decoder_net(img_feat, enc_skip_conn_layers)


        joint_heatmap_out = decoder_out[:,0]


        if self.cfg.model.has_object:
            obj_seg_out = decoder_out[:,1]
            obj_seg_gt = input['obj_seg']  # if self.training else None
            # obj_kps_coord_gt = input['obj_kps_coord']
        else:
            obj_seg_out = None
            obj_seg_gt = None
            # obj_kps_coord_gt = None

        # Get the positional embeddings
        pos = self.position_embedding(nn.functional.interpolate(input_img, (self.cfg.model.output_hm_shape[2], self.cfg.model.output_hm_shape[1])),
                                nn.functional.interpolate(input_mask, (self.cfg.model.output_hm_shape[2], self.cfg.model.output_hm_shape[1])))

        # Get the input tokens
        input_seq, masks, positions, joint_loc_pred_np, mask_np, peak_joints_map_batch \
            = self.get_input_seq(joint_heatmap_out, obj_seg_out, feature_pyramid, pos, input,
                                 input['joint_coord'], input['joint_valid'], obj_seg_gt, epoch_cnt)

        if self.cfg.model.use_bottleneck_hand_type:
            bottleneck_hand_type_feat = F.avg_pool2d(img_feat, (img_feat.shape[2], img_feat.shape[3])).view(-1, img_feat.shape[1])
            bottleneck_hand_type = torch.sigmoid(self.linear_bottleneck_hand_type(bottleneck_hand_type_feat))



        # Concatenate positional and appearance embeddings
        if self.cfg.model.position_embedding == 'simpleCat':
            input_seq = torch.cat([input_seq, positions], dim=1)
            positions = torch.zeros_like(input_seq).to(input_seq.device)



        # Define attention masks
        tgt_key_padding_mask = None
        if self.cfg.model.hand_type == 'both':
            # define attention masks on the queries. This is irrelevant when using only 1 cross-attention layer.
            tgt_mask = get_tgt_mask(self.cfg.model).to(input_seq.device)

        if self.cfg.model.has_object:
            _, memory_mask = get_src_memory_mask(peak_joints_map_batch, self.cfg.model)
            memory_mask = memory_mask.to(input_seq.device)
            src_mask = None
        else:
            src_mask = None
            memory_mask = None


        transformer_out, hand_type, memory, encoder_out, attn_wts_all_layers = self.transformer(src=input_seq, mask=torch.logical_not(masks),
                                                                                       query_embed=self.query_embed.weight,
                                                                                       pos_embed=positions,
                                                                                       tgt_mask=tgt_mask if self.cfg.model.use_tgt_mask else None,
                                                                                       tgt_key_padding_mask = tgt_key_padding_mask,
                                                                                       src_mask=src_mask,
                                                                                       memory_mask=memory_mask)


        # Make all the predictions
        if self.cfg.model.hand_type == 'both':
            if self.cfg.model.predict_type == 'angles':
                pose = self.linear_pose(transformer_out[:, :(self.cfg.model.num_joint_queries_per_hand*2)])  # 6 x 32 x N x 3(9)
                shape = self.linear_shape(transformer_out[:, self.cfg.model.shape_indx])  # 6 x N x 10

            elif self.cfg.model.predict_type == 'vectors':
                if self.cfg.model.predict_2p5d:
                    joint_px_hm = self.linear_joint_2p5d_px(transformer_out[:, :self.cfg.model.num_joint_queries_per_hand*2]) # 6 x 42 x N x 128
                    joint_py_hm = self.linear_joint_2p5d_py(transformer_out[:, :self.cfg.model.num_joint_queries_per_hand * 2])  # 6 x 42 x N x 128
                    joint_dep_hm = self.linear_joint_2p5d_dep(transformer_out[:, :self.cfg.model.num_joint_queries_per_hand * 2])  # 6 x 42 x N x 128
                    joint_2p5d_hm = torch.cat([joint_px_hm.unsqueeze(3),
                                               joint_py_hm.unsqueeze(3), joint_dep_hm.unsqueeze(3)], dim=3) # 6 x 42 x N x 3 x 128
                else:
                    joint_vecs = self.linear_joint_vecs(transformer_out[:, :self.cfg.model.num_joint_queries_per_hand*2]) # 6 x 40 x N x 3
            else:
                raise NotImplementedError

            rel_trans = self.linear_rel_trans(transformer_out[:, self.cfg.model.shape_indx])

            if self.cfg.model.has_object:
                obj_rot = self.linear_obj_rot(transformer_out[:, self.cfg.model.obj_rot_indx])  # 6 x N x 3
                obj_trans = self.linear_obj_rel_trans(transformer_out[:, self.cfg.model.obj_trans_indx])
                if self.cfg.model.predict_obj_left_hand_trans:
                    obj_trans_left = self.linear_obj_left_rel_trans(transformer_out[:, self.cfg.model.obj_trans_indx])
                else:
                    obj_trans_left = None
                obj_corner_proj = self.linear_obj_corner_proj(transformer_out[:, self.cfg.model.obj_trans_indx]) # 6 x N x 16


        if self.cfg.model.use_2D_loss:
            cam_param = self.linear_cam(transformer_out[:, self.cfg.model.shape_indx]) # 6 x N x 3
        else:
            cam_param = None

        if self.cfg.model.enc_layers>0:
            joint_class = self.linear_class(encoder_out) # 6 x max_num_peaks x N x 22(43)


        # Put all the outputs in a dict
        out = {}
        
        if self.training:
            out['rel_trans_out'] = rel_trans[-1]
            out['attn_weights_out'] = attn_wts_all_layers[0]
            if 'inv_trans' in input:
                out['inv_trans_out'] = input['inv_trans']
            out['joint_heatmap_out'] = joint_heatmap_out
            if self.cfg.model.predict_type == 'angles':
                out['pose_out'] = pose  # torch.Size([6, 19, 10, 3])
                out['shape_out'] = shape # torch.Size([6, 10, 10])

            if self.cfg.model.enc_layers > 0:
                out['joint_class_out'] = joint_class[-1].permute(1,0,2)# N x max_num_peaks
            out['seq_mask_out'] = torch.from_numpy(mask_np).to(transformer_out.device)
            out['joint_loc_pred_out'] = torch.from_numpy(joint_loc_pred_np).to(transformer_out.device)
            out['cam_param_out'] = cam_param[-1]
            if self.cfg.model.hand_type == 'both':
                if self.cfg.model.use_bottleneck_hand_type:
                    out['hand_type_out'] = bottleneck_hand_type  # N x 2
                else:
                    out['hand_type_out'] = torch.argmax(hand_type[-1,0], dim=1) # N

            if self.cfg.model.has_object:
                # out['obj_rot_out'] = obj_rot[-1]
                # out['obj_trans_out'] = obj_trans[-1]
                out['obj_corner_proj_out'] = obj_corner_proj[-1]
                # out['obj_seg_gt_out'] = obj_seg_gt
                out['obj_seg_pred_out'] = obj_seg_out
                
            # added by yqwang
            out['obj_corner_proj'] = obj_corner_proj
            out['rel_trans'] = rel_trans
            out['cam_param'] = cam_param
            out['joint_loc_pred_np'] = joint_loc_pred_np
            out['obj_rot'] = obj_rot
            out['obj_trans'] = obj_trans
            out['obj_trans_left'] = obj_trans_left
            
            if self.cfg.model.predict_type == 'vectors': 
                if self.cfg.model.predict_2p5d:
                    out['joint_2p5d_hm'] = joint_2p5d_hm
                else:
                    out['joint_vecs'] = joint_vecs
                
            out['mask_np'] = mask_np
            out['joint_class'] = joint_class
            out['peak_joints_map_batch'] = peak_joints_map_batch
            out['transformer_out'] = transformer_out
            out['hand_type'] = hand_type
        
        else:  # inference
            if "mano_pose" in input:
                # get hand gts
                gt_pose = input['mano_pose'][:, :48]
                gt_shape = input['mano_shape'][:, :10]
            
                mano_out = self.mano_mesh.mano_layer['right'](global_orient=gt_pose[:, :3].cpu(),
                                                            hand_pose=gt_pose[:, 3:48].cpu(),
                                                            betas=gt_shape.cpu())
                # import ipdb; ipdb.set_trace()
                gt_verts = mano_out.vertices.cuda()
                # gt_joints = mano_out.joints.cuda()
                # print(gt_verts.shape, gt_joints.shape)
                # gt_verts /= 1000
                # gt_joints /= 1000
                
                out['gt_verts3d_cam'] = gt_verts
                # out['gt_joints3d_cam'] = gt_joints  
                # out['gt_joints3d_cam'] = input['joint_cam'][:, :21] / 1000
            
                # original
                out['gt_joints3d_cam'] = torch.matmul(self.ih26m_joint_regressor.cuda(), gt_verts)  # torch.Size([B, 21, 3])
                # added by yqwang
                out['gt_joints3d_cam'] = out['gt_joints3d_cam'][:, self.jointsNormalToManoMap]
            
            # print(out['gt_joints3d_cam'].shape)  
            
            # get hand predictions
            # print(pose.shape, shape.shape)  # torch.Size([6, 19, 10, 3]), torch.Size([6, 10, 10])
            # pred_verts, pred_joints = self.mano_layer(th_pose_coeffs=pose, th_betas=shape)
            # pred_verts /= 1000
            # pred_joints /= 1000
            
            # out['pred_verts3d_cam'] = pred_verts
            # out['pred_joints3d_cam'] = pred_joints
            # import ipdb; ipdb.set_trace() # self.
            joints_pred = self.mano_mesh.get_mano_mesh(pose, shape, rel_trans, input['root_valid'], cam_param)
            out['pred_verts3d_cam'] = joints_pred['mesh_right'][-1].cuda()  # 778
            out['pred_joints3d_cam'] = joints_pred['joints_right'][-1].cuda()  # 21
            
            if self.cfg.model.has_object:
                out['obj_rot_out'] = obj_rot[-1]
                out['obj_trans_out'] = obj_trans[-1]

        return out
    
    
from torchvision import ops
from model.hfl_net_hor.backbone_share import FPN
from model.hfl_net_hor.hand_head import hand_Encoder, hand_regHead
from model.hfl_net_hor.object_head import obj_regHead, Pose2DLayer, Pose2DLayerConf
from model.hfl_net_hor.mano_head import mano_regHead
from model.hfl_net_hor.CR import Transformer

class HFLNet(nn.Module):
    def __init__(self, cfg):
        roi_res=32
        joint_nb=21
        stacks=1
        channels=256
        blocks=1
        transformer_depth=1
        transformer_head=8
        
        mano_layer = mano.layer
        
        mano_neurons=[1024, 512]
        coord_change_mat=None
        reg_object=True
        pretrained=True

        super(HFLNet, self).__init__()

        self.out_res = roi_res

        # FPN-Res50 backbone
        self.base_net = FPN(pretrained=pretrained)

        # hand head
        self.hand_head = hand_regHead(roi_res=roi_res, joint_nb=joint_nb,
                                      stacks=stacks, channels=channels, blocks=blocks)
        # hand encoder
        self.hand_encoder = hand_Encoder(num_heatmap_chan=joint_nb, num_feat_chan=channels,
                                         size_input_feature=(roi_res, roi_res))
        # mano branch
        self.mano_branch = mano_regHead(mano_layer, feature_size=self.hand_encoder.num_feat_out,
                                        mano_neurons=mano_neurons, coord_change_mat=coord_change_mat)
        # object head
        self.reg_object = reg_object
        self.obj_head = obj_regHead(channels=channels, inter_channels=channels//2, joint_nb=joint_nb)
        self.obj_reorgLayer = Pose2DLayer(joint_nb=joint_nb)

        # CR blocks
        self.transformer_obj = Transformer(inp_res=roi_res, dim=channels, depth=transformer_depth, num_heads=transformer_head)
        self.transformer_hand = Transformer(inp_res=roi_res, dim=channels*2,depth=transformer_depth, num_heads=transformer_head)

        self.hand_head.apply(self.init_weights)
        self.hand_encoder.apply(self.init_weights)
        self.mano_branch.apply(self.init_weights)
        self.transformer_obj.apply(self.init_weights)
        self.transformer_hand.apply(self.init_weights)
        self.obj_head.apply(self.init_weights)


    def init_weights(self, m):
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            nn.init.constant_(m.bias,0)


    def net_forward(self, imgs, bbox_hand, bbox_obj,mano_params=None, roots3d=None, return_feat=False):
        batch = self.new_method(imgs)

        inter_topLeft = torch.max(bbox_hand[:, :2], bbox_obj[:, :2])
        inter_bottomRight = torch.min(bbox_hand[:, 2:], bbox_obj[:, 2:])
        bbox_inter = torch.cat((inter_topLeft, inter_bottomRight), dim=1)
        msk_inter = ((inter_bottomRight-inter_topLeft > 0).sum(dim=1)) == 2
        # P2 from FPN Network
        P2_h,P2_o = self.base_net(imgs)
        idx_tensor = torch.arange(batch, device=imgs.device).float().view(-1, 1)
        # get roi boxes
        roi_boxes_hand = torch.cat((idx_tensor, bbox_hand), dim=1)
        # 4 here is the downscale size in FPN network(P2)
        x_hand = ops.roi_align(P2_h, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0,
                          sampling_ratio=-1)  # hand

        x_obj = ops.roi_align(P2_o, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0,
                          sampling_ratio=-1)  # hand

        # obj forward
        if self.reg_object:
            roi_boxes_obj = torch.cat((idx_tensor, bbox_obj), dim=1)
            roi_boxes_inter = torch.cat((idx_tensor, bbox_inter), dim=1)

            y = ops.roi_align(P2_o, roi_boxes_obj, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)  # obj

            z_x = ops.roi_align(P2_h, roi_boxes_inter, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)  # intersection

            z_x = msk_inter[:, None, None, None] * z_x

            hand_obj = torch.cat([x_hand,x_obj.detach()],dim=1)
            hand_obj = self.transformer_hand(hand_obj,hand_obj)

            y = self.transformer_obj(y, z_x.detach())

            out_fm = self.obj_head(y)
            preds_obj = self.obj_reorgLayer(out_fm)
        else:
            preds_obj = None

        hand = hand_obj[:,0:256,:,:]
        #hand forward

        out_hm, encoding, preds_joints = self.hand_head(hand)

        mano_encoding = self.hand_encoder(out_hm, encoding)

        pred_mano_results, gt_mano_results = self.mano_branch(mano_encoding, mano_params=mano_params, roots3d=roots3d)

        # features are in the dict
        if return_feat:
            hand_feats = {"P2_h": P2_h, "x_hand": x_hand, "hand": hand, "out_hm": out_hm[0], "encoding": encoding[0], "mano_encoding": mano_encoding}
            return preds_joints, pred_mano_results, gt_mano_results, preds_obj, hand_feats
        else:
            return preds_joints, pred_mano_results, gt_mano_results, preds_obj
        
    def new_method(self, imgs):
        batch = imgs.shape[0]
        return batch

    def forward(self, input):
        imgs = input["img"]
        # print(imgs.shape)
        
        bbox_hand = input["bbox_hand"]
        bbox_obj = input["bbox_obj"]
        mano_params=input.get("mano_param", None)
        # roots3d=input["root_joint_flip"]
        joints_img = input.get("joints_img", None)
        
        output = {}
        
        if self.training:
            if imgs.shape[1] == 6:  # contrastive learning
                preds_joints_lst = []
                pred_mano_results_lst = defaultdict(list)
                gt_mano_results_lst = defaultdict(list)
                preds_obj_lst = []
                hand_feats_dict_lst = defaultdict(list)
                joints_img_lst = []
                
                for i in range(2):
                    preds_joints, pred_mano_results, gt_mano_results, preds_obj, hand_feats_dict = self.net_forward(imgs[:, i*3:(i+1)*3], bbox_hand[:, i*4:(i+1)*4], bbox_obj[:, i*4:(i+1)*4],
                                                                                            mano_params=mano_params[:, i*58:(i+1)*58], return_feat=True)
                    preds_joints_lst.append(preds_joints)  # list
                    preds_obj_lst.append(preds_obj) # list
                    
                    for k, v in pred_mano_results.items():
                        pred_mano_results_lst[k].append(v) # dict
                    
                    for k, v in gt_mano_results.items():
                        gt_mano_results_lst[k].append(v) # dict
                    
                    for k, v in hand_feats_dict.items():
                        hand_feats_dict_lst[k].append(v)
                        
                    joints_img_lst.append(joints_img[:, i*21:(i+1)*21])
                    
                # TODO: change concat strategy for some tensors used for object loss
                for k, v in pred_mano_results_lst.items():
                    pred_mano_results_lst[k] = torch.cat(v, 1)
                pred_mano_results = pred_mano_results_lst
                
                for k, v in gt_mano_results_lst.items():
                    gt_mano_results_lst[k] = torch.cat(v, 1)
                gt_mano_results = gt_mano_results_lst
                
                preds_joints = []
                for i in range(len(preds_joints_lst[0])):  # for each value
                    preds_joints.append(torch.cat([preds_joints_lst[0][i], preds_joints_lst[1][i]], 1))
                    
                preds_obj = []
                for i in range(len(preds_obj_lst[0])):  # for each value
                    preds_obj.append(torch.cat([preds_obj_lst[0][i], preds_obj_lst[1][i]], 1))
                    
                for k, v in hand_feats_dict_lst.items():
                    hand_feats_dict_lst[k] = torch.cat(v, 1)
                output.update(hand_feats_dict_lst)
                
                joints_img_lst = torch.cat(joints_img_lst, 1)
                output["joints_img"] = joints_img_lst
            else:
                preds_joints, pred_mano_results, gt_mano_results, preds_obj, hand_feats_dict = self.net_forward(imgs, bbox_hand, bbox_obj,
                                                                                           mano_params=mano_params, return_feat=True)
            output["preds_joints2d"] = preds_joints
            output["pred_mano_results"] = pred_mano_results
            output["gt_mano_results"] = gt_mano_results
            output["preds_obj"] = preds_obj
            
            # # NOTE: for calculating metric in validation set
            output["pred_verts3d_cam"] = pred_mano_results["verts3d"]
            output["pred_joints3d_cam"] = pred_mano_results["joints3d"]
            output["gt_verts3d_cam"] = gt_mano_results["verts3d"]
            output["gt_joints3d_cam"] = gt_mano_results["joints3d"]
        else:
            preds_joints, pred_mano_results, gt_mano_results, preds_obj = self.net_forward(imgs, bbox_hand, bbox_obj,
                                                                             mano_params=mano_params)
            output["preds_joints2d"] = preds_joints[0]
            # output["pred_mano_results"] = pred_mano_results
            # output["preds_obj"] = preds_obj
            for i in range(len(preds_obj)):
                output["preds_obj_{}".format(i)] = preds_obj[i]
            
            output["pred_verts3d_cam"] = pred_mano_results["verts3d"]
            output["pred_joints3d_cam"] = pred_mano_results["joints3d"]
            
            if gt_mano_results is not None:
                output["gt_verts3d_cam"] = gt_mano_results["verts3d"]
                output["gt_joints3d_cam"] = gt_mano_results["joints3d"]
        
        return output
    

def denormalize_joints_img(normed_joints2d, bbox, img_shape):
    #
    B = bbox.shape[0]
    bbox = bbox.view(B, 2, 2)
    
    # convert to img coord
    joints2d = normed_joints2d * (bbox[:, 1:, :] - bbox[:, :1, :]) + bbox[:, :1, :]
    
    # then normalize with img size
    img_mid = torch.tensor(img_shape).cuda() / 2.0 # tensor([64., 64.]
    
    joints2d = torch.clamp((joints2d - img_mid) / img_mid, -1, 1)
    return joints2d

# UniHOPE network
class UniHOPENet(nn.Module):
    def __init__(self, cfg):
        roi_res=32
        joint_nb=21
        stacks=1
        channels=256
        blocks=1
        transformer_depth=1
        transformer_head=8
        
        mano_layer = mano.layer
        
        mano_neurons=[1024, 512]
        coord_change_mat=None
        reg_object=True
        pretrained=True

        super(UniHOPENet, self).__init__()

        self.out_res = roi_res

        # FPN-Res50 backbone
        self.base_net = FPN(pretrained=pretrained)

        # hand head
        self.hand_head = hand_regHead(roi_res=roi_res, joint_nb=joint_nb,
                                      stacks=stacks, channels=channels, blocks=blocks)
        # hand encoder
        self.hand_encoder = hand_Encoder(num_heatmap_chan=joint_nb, num_feat_chan=channels,
                                         size_input_feature=(roi_res, roi_res))
        # mano branch
        self.mano_branch = mano_regHead(mano_layer, feature_size=self.hand_encoder.num_feat_out,
                                        mano_neurons=mano_neurons, coord_change_mat=coord_change_mat)
        # object head
        self.reg_object = reg_object
        self.obj_head = obj_regHead(channels=channels, inter_channels=channels//2, joint_nb=joint_nb)
        self.obj_reorgLayer = Pose2DLayer(joint_nb=joint_nb)
        
        # object classification head
        if cfg.data.input_img_shape[0] == 128:
            conv_spatial_dim = 32
        elif cfg.data.input_img_shape[0] == 256:
            conv_spatial_dim = 64
        
        obj_conf_layer_legacy = cfg.model.get("obj_conf_layer_legacy", True)
        if obj_conf_layer_legacy:  # legacy version, occupying too much memory space
            self.obj_conf_layer = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=True),
                nn.Flatten(), 
                nn.Linear(64*conv_spatial_dim*conv_spatial_dim, 1024),
                nn.Linear(1024, 2))  # with object or not
        else:  # reduce the model size
            self.obj_conf_layer = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=True),
                nn.Flatten(), 
                nn.Linear(64*conv_spatial_dim*conv_spatial_dim, 2))

        # occlusion prediction head
        self.occ_cls_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=True),
            nn.Flatten(), 
            nn.Linear(64*conv_spatial_dim*conv_spatial_dim, 2))  # occluded or non-occluded
        
        # use 1d convolutional layer for fpn feature
        self.fpn_feature_mapper = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True)
        )
        # use attention for high-level features
        self.hand_feature_mapper = SlfMultiHeadAttention(
            n_head=8,
            d_model=256, 
            d_k=256, 
            d_v=256
        )
        self.mano_feature_mapper = SlfMultiHeadAttention(
            n_head=8,
            d_model=1024, 
            d_k=256, 
            d_v=256
        )
       
        # CR blocks
        self.transformer_obj = Transformer(inp_res=roi_res, dim=channels, depth=transformer_depth, num_heads=transformer_head)
        self.transformer_hand = Transformer(inp_res=roi_res, dim=channels*2,depth=transformer_depth, num_heads=transformer_head)

        self.hand_head.apply(self.init_weights)
        self.hand_encoder.apply(self.init_weights)
        self.mano_branch.apply(self.init_weights)
        self.transformer_obj.apply(self.init_weights)
        self.transformer_hand.apply(self.init_weights)
        self.obj_head.apply(self.init_weights)
        
        self.init_layer_weight = cfg.train.get("init_layer_weight", False)
        if self.init_layer_weight:
            self.obj_conf_layer.apply(self.init_weights)
            # self.occ_cls_layer.apply(self.init_weights)
            self.fpn_feature_mapper.apply(self.init_weights)
            self.hand_feature_mapper.apply(self.init_weights)
            self.mano_feature_mapper.apply(self.init_weights)
            
        self.train_part = cfg.train.get("train_part", [0, 1]) # train on which part of the dataset, [0], [1], [0, 1]

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)

    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]

    def net_forward(self, imgs, bbox_hand, bbox_obj,mano_params=None, roots3d=None, return_feat=False, batch_input=None, aug_rot_mat=None, feature_mapping=False):
        batch = self.new_method(imgs)

        inter_topLeft = torch.max(bbox_hand[:, :2], bbox_obj[:, :2])
        inter_bottomRight = torch.min(bbox_hand[:, 2:], bbox_obj[:, 2:])
        bbox_inter = torch.cat((inter_topLeft, inter_bottomRight), dim=1)
        msk_inter = ((inter_bottomRight-inter_topLeft > 0).sum(dim=1)) == 2
        # P2 from FPN Network
        P2_h,P2_o, base_feat_dict = self.base_net(imgs, return_feat=True)
        
        idx_tensor = torch.arange(batch, device=imgs.device).float().view(-1, 1)
        # get roi boxes
        roi_boxes_hand = torch.cat((idx_tensor, bbox_hand), dim=1)
        # 4 here is the downscale size in FPN network(P2)
        x_hand = ops.roi_align(P2_h, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0,
                          sampling_ratio=-1)  # hand

        x_obj = ops.roi_align(P2_o, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0,
                          sampling_ratio=-1)  # hand
        
        # convert P2_o to probability
        # P2_o.shape: torch.Size([B, 256, 32, 32])
        pred_obj_conf = self.obj_conf_layer(P2_o)  # [B, 2]
        
        # P2_h.shape: torch.Size([B, 256, 32, 32])
        pred_hand_occ = self.occ_cls_layer(P2_h)  # [B, 2]

        # obj forward
        if self.reg_object:
            roi_boxes_obj = torch.cat((idx_tensor, bbox_obj), dim=1)  # roi_boxes_obj might be None if object is not present
            roi_boxes_inter = torch.cat((idx_tensor, bbox_inter), dim=1)

            y = ops.roi_align(P2_o, roi_boxes_obj, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)  # obj
            
            z_x = ops.roi_align(P2_h, roi_boxes_inter, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)  # intersection

            z_x = msk_inter[:, None, None, None] * z_x
            
            # no matter train or test, use the prediction as object mask
            obj_mask = torch.argmax(pred_obj_conf, dim=-1).bool()  # whether holding object
            
            hand_occ_mask = torch.argmax(pred_hand_occ, dim=-1).bool()  # whether hand is occluded
            
            obj_nonocc_mask = torch.logical_and(~hand_occ_mask, obj_mask)  # holding object, but hand is not occluded. We only fuse such object feature
            
            x_hand_obj = x_hand.clone()
            
            # for fusing object feature, still use predicted object mask
            x_hand_obj[obj_nonocc_mask] = x_obj[obj_nonocc_mask].detach()
            
            hand_obj = torch.cat([x_hand,x_hand_obj],dim=1)
            hand_obj = self.transformer_hand(hand_obj,hand_obj)

            y = self.transformer_obj(y, z_x.detach())  # torch.Size([64, 256, 32, 32])

            out_fm = self.obj_head(y)  # torch.Size([B, 63, 32, 32])
            preds_obj = self.obj_reorgLayer(out_fm)
        else:
            preds_obj = None

        hand = hand_obj[:,0:256,:,:]

        out_hm, encoding, preds_joints = self.hand_head(hand)

        mano_encoding = self.hand_encoder(out_hm, encoding)
        
        pred_mano_results, gt_mano_results, mano_feat_dict = self.mano_branch(mano_encoding.clone(), mano_params=mano_params, roots3d=roots3d, return_feat=True)
        
        # index the image feature with predicted 2d joint
        uv = preds_joints[0].clone()
        # normalize
        uv = torch.clamp((uv - 0.5) * 2, -1, 1) # torch.Size([B, 256, 32, 32])
        # since joint_uv is relative to the bbox_hand, so we use uv to index the image area corresponding to bbox_hand
        joint_img_feat = self.index(x_hand, uv) # torch.Size([B, 256, 21]) 
        
        # index the base feature with predicted 2d joint
        c3_h = base_feat_dict['c3_h']  # torch.Size([B, 512, 16, 16])
        # first convert joint2d to (0-1) in image coord
        img_uv = denormalize_joints_img(preds_joints[0], bbox_hand, imgs.shape[2:])
        # then index the FPN feature
        joint_base_feat = self.index(c3_h, img_uv)  # torch.Size([B, 512, 21])
        
        if feature_mapping:  # first sample, do feature mapping
            feature_mapping_mask = torch.ones(imgs.shape[0], dtype=torch.bool, device=imgs.device)
        else:  # second sample, will not do feature mapping
            feature_mapping_mask = None
            
         # feature mapping for grasping hands
        if feature_mapping_mask is not None and torch.count_nonzero(feature_mapping_mask) > 0:
            # adaptation for joint fpn feature
            joint_base_feat[feature_mapping_mask] = self.fpn_feature_mapper(joint_base_feat[feature_mapping_mask])
            # adaptation for joint hand feature
            joint_img_feat[feature_mapping_mask] = self.hand_feature_mapper(joint_img_feat[feature_mapping_mask].permute(0, 2, 1)).permute(0, 2, 1)
            # adaptation for mano feature
            mano_encoding[feature_mapping_mask] = self.mano_feature_mapper(mano_encoding[feature_mapping_mask].unsqueeze(1)).squeeze(1)  # torch.Size([B, 1024])

        # features are in the dict
        if return_feat:
            hand_feats = {}
            hand_feats.update(mano_feat_dict)
            hand_feats.update(base_feat_dict)
            hand_feats.update({"P2_h": P2_h, "P2_o": P2_o, "c3_h": c3_h, "x_hand": x_hand, "hand": hand, "out_hm": out_hm[0], "encoding": encoding[0], "mano_encoding": mano_encoding, "x_obj": x_obj, 'joint_base_feat': joint_base_feat, "joint_img_feat": joint_img_feat})
            return preds_joints, pred_mano_results, gt_mano_results, preds_obj, pred_obj_conf, pred_hand_occ, hand_feats
        else:
            return preds_joints, pred_mano_results, gt_mano_results, preds_obj, pred_obj_conf, pred_hand_occ

    def new_method(self, imgs):
        batch = imgs.shape[0]
        return batch

    def forward(self, input):
        imgs = input["img"]
        
        bbox_hand = input["bbox_hand"]
        bbox_obj = input["bbox_obj"]  # if no object present, it will be empty
        mano_params=input.get("mano_param", None)
        roots3d = input["root_joint_flip"]
        gt_grasping = input["gt_grasping"]
        joints_img = input.get("joints_img", None)
        aug_rot_mat = input["aug_rot_mat"]
        
        output = {}
        if self.training:
            if imgs.shape[1] == 6:  # contrastive learning
                preds_joints_lst = []
                pred_mano_results_lst = defaultdict(list)
                gt_mano_results_lst = defaultdict(list)
                preds_obj_lst = []
                pred_obj_conf_lst = []
                hand_feats_dict_lst = defaultdict(list)
                joints_img_lst = []
                pred_hand_occ_lst = []
                
                # for i in range(2):
                for i in self.train_part:  # only go through parts we're interested in to speed up the foreward process
                    aug_rot_mat_i = None
                    
                    preds_joints, pred_mano_results, gt_mano_results, preds_obj, pred_obj_conf, pred_hand_occ, hand_feats_dict = self.net_forward(imgs[:, i*3:(i+1)*3], bbox_hand[:, i*4:(i+1)*4], bbox_obj[:, i*4:(i+1)*4],
                                                                                        mano_params=mano_params[:, i*58:(i+1)*58], return_feat=True, batch_input=input,
                                                                                        aug_rot_mat=aug_rot_mat_i,
                                                                                        feature_mapping=not bool(i))
                    preds_joints_lst.append(preds_joints)  # list
                    preds_obj_lst.append(preds_obj) # list
                    pred_obj_conf_lst.append(pred_obj_conf) # Tensor
                    pred_hand_occ_lst.append(pred_hand_occ)
                    
                    for k, v in pred_mano_results.items():
                        pred_mano_results_lst[k].append(v) # dict
                    
                    for k, v in gt_mano_results.items():
                        gt_mano_results_lst[k].append(v) # dict
                    
                    for k, v in hand_feats_dict.items():
                        hand_feats_dict_lst[k].append(v)
                        
                    joints_img_lst.append(joints_img[:, i*21:(i+1)*21])
                    
                for k, v in pred_mano_results_lst.items():
                    pred_mano_results_lst[k] = torch.cat(v, 1)
                pred_mano_results = pred_mano_results_lst
                
                for k, v in gt_mano_results_lst.items():
                    gt_mano_results_lst[k] = torch.cat(v, 1)
                gt_mano_results = gt_mano_results_lst
                
                preds_joints = []
                for i in range(len(preds_joints_lst[0])):  # for each value
                    preds_joints_lst_lst = []
                    for j in range(len(preds_joints_lst)):  # for each part
                        preds_joints_lst_lst.append(preds_joints_lst[j][i])
                        
                    preds_joints.append(torch.cat(preds_joints_lst_lst, 1))
                    
                preds_obj = []
                for i in range(len(preds_obj_lst[0])):  # for each value
                    preds_obj_lst_lst = []
                    for j in range(len(preds_obj_lst)):  # for each part
                        preds_obj_lst_lst.append(preds_obj_lst[j][i])
                        
                    preds_obj.append(torch.cat(preds_obj_lst_lst, 1))
                    
                pred_obj_conf = torch.cat(pred_obj_conf_lst, 1)
                pred_hand_occ = torch.cat(pred_hand_occ_lst, 1)
                
                # merge the feature output
                for k, v in hand_feats_dict_lst.items():
                    hand_feats_dict_lst[k] = torch.cat(v, 1)
                output.update(hand_feats_dict_lst)
                
                joints_img_lst = torch.cat(joints_img_lst, 1)
                output["joints_img"] = joints_img_lst
            else:
                preds_joints, pred_mano_results, gt_mano_results, preds_obj, pred_obj_conf, pred_hand_occ = self.net_forward(imgs, bbox_hand, bbox_obj,
                                                                                            mano_params=mano_params)
                
            output["preds_joints2d"] = preds_joints
            output["pred_mano_results"] = pred_mano_results
            output["gt_mano_results"] = gt_mano_results
            output["preds_obj"] = preds_obj
            # additional object classifier
            output["pred_obj_conf"] = pred_obj_conf
            # additional occlusion classifier
            output["pred_hand_occ"] = pred_hand_occ
            
            output["pred_verts3d_cam"] = pred_mano_results["verts3d"]
            output["pred_joints3d_cam"] = pred_mano_results["joints3d"]
            output["gt_verts3d_cam"] = gt_mano_results["verts3d"]
            output["gt_joints3d_cam"] = gt_mano_results["joints3d"]
        else:  # test
            preds_joints, pred_mano_results, gt_mano_results, preds_obj, pred_obj_conf, pred_hand_occ = self.net_forward(imgs, bbox_hand, bbox_obj,
                                                                            mano_params=mano_params,
                                                                            feature_mapping=True,
                                                                            aug_rot_mat=aug_rot_mat)
            output["preds_joints2d"] = preds_joints[0]
            for i in range(len(preds_obj)):
                output["preds_obj_{}".format(i)] = preds_obj[i]
            
            output["pred_verts3d_cam"] = pred_mano_results["verts3d"]
            output["pred_joints3d_cam"] = pred_mano_results["joints3d"]
            
            if gt_mano_results is not None:
                output["gt_verts3d_cam"] = gt_mano_results["verts3d"]
                output["gt_joints3d_cam"] = gt_mano_results["joints3d"]
            # additional object classifier
            output["pred_obj_conf"] = pred_obj_conf
            output["pred_hand_occ"] = pred_hand_occ
            output["pred_grasping"] = torch.argmax(pred_obj_conf, dim=-1)
        return output
    
    
# this classifier is the same as used in UniHOPENet
class GraspClassifier(nn.Module):
    def __init__(self, cfg):
        super(GraspClassifier, self).__init__()
        
        pretrained=True
        
        # FPN-Res50 backbone
        self.base_net = FPN(pretrained=pretrained)
        
        # TODO: make it compatible in configuration
        input_img_shape = (128, 128)
        
        # object classification head
        if input_img_shape[0] == 128:
            conv_spatial_dim = 32
        elif input_img_shape[0] == 256:
            conv_spatial_dim = 64
            
        self.obj_conf_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=True),
            nn.Flatten(), 
            nn.Linear(64*conv_spatial_dim*conv_spatial_dim, 2)
        )
    
    def forward(self, imgs):
        P2_h,P2_o = self.base_net(imgs)
        
        res = self.obj_conf_layer(P2_o)  # [B, 2]
        
        output = {"pred_obj_conf": res}
        return output


# build submodules for A+B baseline
def build_sub_model(model_name, cfg):
    if model_name == "hand_occ_net":
        model = HandOccNet(cfg)
    elif model_name == "semi_hand":
        model = SemiHand(cfg)
    elif model_name == "mobrecon":
        model = MobRecon_DS(cfg)
    elif model_name == "h2onet":
        model = H2ONet(cfg)
    elif model_name == "hflnet":
        model = HFLNet(cfg)
    elif model_name == "classifier":
        model = GraspClassifier(cfg)
    else:
        raise NotImplementedError
    
    return model

class TwoInOneBaselineClassifier(nn.Module):
    def __init__(self, cfg):
        super(TwoInOneBaselineClassifier, self).__init__()
        
        self.cfg = cfg
        
        # build model
        self.cls_model = build_sub_model(cfg.model.cls_model.name, cfg.model.cls_model)
        self.h_model = build_sub_model(cfg.model.h_model.name, cfg.model.h_model)
        self.ho_model = build_sub_model(cfg.model.ho_model.name, cfg.model.ho_model)
    
    def forward(self, input):
        if "img_cls" in input:
            img_cls = input["img_cls"]
        else:
            img_cls = input["img"]
            
        cls_out = self.cls_model(img_cls)
        res = cls_out["pred_obj_conf"]  # [B, 2]
        
        graspings = torch.argmax(res, dim=-1).bool()  # use predicted grasping, [B]
        
        B = graspings.shape[0]
        device = graspings.device
        
        ho_mask = graspings
        h_mask = ~ho_mask
        
        ho_num = torch.count_nonzero(ho_mask)
        h_num = B - ho_num
        
        pred_verts3d_cam = torch.zeros(B, 778, 3).to(device)
        pred_joints3d_cam = torch.zeros(B, 21, 3).to(device)
        gt_verts3d_cam = torch.zeros(B, 778, 3).to(device)
        gt_joints3d_cam = torch.zeros(B, 21, 3).to(device)
        
        if ho_num > 0:
            ho_input = {}
            # split the input
            for k, v in input.items():
                if type(v) != list:
                    ho_input[k] = v[ho_mask]
                    
            if "img_ho" in ho_input:
                ho_input["img"] = ho_input["img_ho"]
            
            ho_out = self.ho_model(ho_input)  # B1
            # merge the output
            pred_verts3d_cam[ho_mask] = ho_out["pred_verts3d_cam"]
            pred_joints3d_cam[ho_mask] = ho_out["pred_joints3d_cam"]
            
            if "gt_verts3d_cam" in ho_out:
                gt_verts3d_cam[ho_mask] = ho_out["gt_verts3d_cam"]
                gt_joints3d_cam[ho_mask] = ho_out["gt_joints3d_cam"]
        if h_num > 0:
            h_input = {}
            for k, v in input.items():
                if type(v) != list:
                    h_input[k] = v[h_mask]
                    
            # if "img_1" in h_input:  # has another image resolution fo ho model
            #     h_input["img"] = h_input["img_1"]  # update the original size image to resized ones
            if "img_h" in h_input:
                h_input["img"] = h_input["img_h"]
            
            h_out = self.h_model(h_input)  # B2
            # merge the output
            pred_verts3d_cam[h_mask] = h_out["pred_verts3d_cam"]
            pred_joints3d_cam[h_mask] = h_out["pred_joints3d_cam"]
            
            if "gt_verts3d_cam" in h_out:
                gt_verts3d_cam[h_mask] = h_out["gt_verts3d_cam"]
                gt_joints3d_cam[h_mask] = h_out["gt_joints3d_cam"]
     
        # import ipdb; ipdb.set_trace()
        outs = {
            "pred_verts3d_cam": pred_verts3d_cam,
            "pred_joints3d_cam": pred_joints3d_cam,
            "pred_grasping": graspings
        }
        if "mano_pose" in input:
            outs.update({
                "gt_verts3d_cam": gt_verts3d_cam,
                "gt_joints3d_cam": gt_joints3d_cam
            })
        
        if not self.training: # testing
            out_preds_obj_0 = torch.zeros(B, 32, 32, 21).to(device)
            out_preds_obj_1 = torch.zeros(B, 32, 32, 21).to(device)
            out_preds_obj_2 = torch.zeros(B, 32, 32, 21).to(device)
                
            if ho_num > 0:
                # only store object output when using HO model
                out_preds_obj_0[ho_mask] = ho_out["preds_obj_0"]
                out_preds_obj_1[ho_mask] = ho_out["preds_obj_1"]
                out_preds_obj_2[ho_mask] = ho_out["preds_obj_2"]
                
            outs.update({
                "preds_obj_0": out_preds_obj_0,
                "preds_obj_1": out_preds_obj_1,
                "preds_obj_2": out_preds_obj_2,
            })
            
        return outs
    