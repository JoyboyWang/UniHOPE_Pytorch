import torch
from torch import nn
from torchvision import ops
from model.semi_hand.networks.backbone import FPN
from model.semi_hand.networks.hand_head import hand_Encoder, hand_regHead
from model.semi_hand.networks.object_head import obj_regHead, Pose2DLayer
from model.semi_hand.networks.mano_head import mano_regHead
from model.semi_hand.networks.CR import Transformer
from model.semi_hand.networks.loss import Joint2DLoss, ManoLoss, ObjectLoss


class HOModel(nn.Module):

    def __init__(self,
                 honet,
                 mano_lambda_verts3d=None,
                 mano_lambda_joints3d=None,
                 mano_lambda_manopose=None,
                 mano_lambda_manoshape=None,
                 mano_lambda_regulshape=None,
                 mano_lambda_regulpose=None,
                 lambda_joints2d=None,
                 lambda_objects=None):

        super(HOModel, self).__init__()
        self.honet = honet
        # supervise when provide mano params
        self.mano_loss = ManoLoss(lambda_verts3d=mano_lambda_verts3d,
                                  lambda_joints3d=mano_lambda_joints3d,
                                  lambda_manopose=mano_lambda_manopose,
                                  lambda_manoshape=mano_lambda_manoshape)
        self.joint2d_loss = Joint2DLoss(lambda_joints2d=lambda_joints2d)
        # supervise when provide hand joints
        self.mano_joint_loss = ManoLoss(lambda_joints3d=mano_lambda_joints3d, lambda_regulshape=mano_lambda_regulshape, lambda_regulpose=mano_lambda_regulpose)
        # object loss
        self.object_loss = ObjectLoss(obj_reg_loss_weight=lambda_objects)

    def forward(self, imgs, bbox_hand, bbox_obj, joints_uv=None, joints_xyz=None, mano_params=None, roots3d=None, obj_p2d_gt=None, obj_mask=None, obj_lossmask=None):
        
