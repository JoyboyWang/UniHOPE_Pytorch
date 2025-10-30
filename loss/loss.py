import torch
import torch.nn as nn
from torch.nn import functional as F
from common.tool import tensor_gpu
from common.utils.mano import MANO
from model.mob_recon.utils.utils import *
from model.hand_occ_net.mano_head import batch_rodrigues, rot6d2mat, mat2aa
from pytorch3d import transforms as p3dt
from common.utils.transforms import *

from model.kypt_transformer.common.nets.loss import *
import copy

mano = MANO()


class CoordLoss(nn.Module):

    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D=None):
        loss = torch.abs(coord_out - coord_gt) * valid
        if is_3D is not None:
            loss_z = loss[:, :, 2:] * is_3D[:, None, None].float()
            loss = torch.cat((loss[:, :, :2], loss_z), 2)

        return loss


class ParamLoss(nn.Module):

    def __init__(self):
        super(ParamLoss, self).__init__()

    def forward(self, param_out, param_gt, valid):
        loss = torch.abs(param_out - param_gt) * valid
        return loss


class NormalVectorLoss(nn.Module):

    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt, is_valid=None):
        face = torch.LongTensor(self.face).cuda()

        v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
        v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
        v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
        v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

        # valid_mask = valid[:, face[:, 0], :] * valid[:, face[:, 1], :] * valid[:, face[:, 2], :]

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))  # * valid_mask
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))  # * valid_mask
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))  # * valid_mask
        loss = torch.cat((cos1, cos2, cos3), 1)
        if is_valid is not None:
            loss *= is_valid
        return torch.mean(loss)


class EdgeLengthLoss(nn.Module):

    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt, is_valid=None):
        face = torch.LongTensor(self.face).cuda()

        d1_out = torch.sqrt(torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :])**2, 2, keepdim=True))
        d2_out = torch.sqrt(torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :])**2, 2, keepdim=True))
        d3_out = torch.sqrt(torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :])**2, 2, keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :])**2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :])**2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :])**2, 2, keepdim=True))

        # valid_mask_1 = valid[:, face[:, 0], :] * valid[:, face[:, 1], :]
        # valid_mask_2 = valid[:, face[:, 0], :] * valid[:, face[:, 2], :]
        # valid_mask_3 = valid[:, face[:, 1], :] * valid[:, face[:, 2], :]

        diff1 = torch.abs(d1_out - d1_gt)  # * valid_mask_1
        diff2 = torch.abs(d2_out - d2_gt)  # * valid_mask_2
        diff3 = torch.abs(d3_out - d3_gt)  # * valid_mask_3
        loss = torch.cat((diff1, diff2, diff3), 1)
        if is_valid is not None:
            loss *= is_valid
        return torch.mean(loss)


class L1Loss(nn.Module):
    """L1Loss.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ... match_costs.match_cost import L1Cost
        >>> import torch
        >>> self = L1Cost()
        >>> bbox_pred = torch.rand(1, 4)
        >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(bbox_pred, gt_bboxes, factor)
        tensor([[1.6172, 1.6422]])
    """

    def __call__(self,
                 pred_bboxes,
                 gt_bboxes,
                 **kwargs):
        bbox_cost = F.l1_loss(pred_bboxes, gt_bboxes, reduction='none')
        return bbox_cost


def mse_loss_with_mask(a, b, mask=None):
    if mask is not None:
        if torch.count_nonzero(mask) == 0:
            return torch.tensor(0.0).cuda()
            # return torch.tensor(0.0, requires_grad=True).cuda()
        else:
            return F.mse_loss(a[mask], b[mask])
    else:
        return F.mse_loss(a, b)

class Joint2DLoss(nn.Module):
    def __init__(self, lambda_joints2d):
        super(Joint2DLoss,self).__init__()
        self.lambda_joints2d = lambda_joints2d

    def __call__(self, preds, gts, mask=None):
        final_loss = 0
        joint_losses = {}
        if type(preds) == list:
            num_stack = len(preds)
            for i, pred in enumerate(preds):
                # joints2d_loss = self.lambda_joints2d * F.mse_loss(pred, gts)
                joints2d_loss = self.lambda_joints2d * mse_loss_with_mask(pred, gts, mask)
                
                final_loss += joints2d_loss
                if i == num_stack-1:
                    joint_losses["hm_joints2d_loss"] = joints2d_loss.detach().cpu()
            final_loss /= num_stack
        else:
            # joints2d_loss = self.lambda_joints2d * F.mse_loss(preds, gts)
            joints2d_loss = self.lambda_joints2d * mse_loss_with_mask(preds, gts, mask)
            final_loss = joints2d_loss
            joint_losses["hm_joints2d_loss"] = joints2d_loss.detach().cpu()
        return final_loss, joint_losses
    
    
class Joint2DLossHalf(nn.Module):
    def __init__(self, lambda_joints2d):
        super(Joint2DLossHalf,self).__init__()
        self.lambda_joints2d = lambda_joints2d

    def __call__(self, preds, gts, mask=None, index=0):
        final_loss = 0
        joint_losses = {}
        if type(preds) == list:
            num_stack = len(preds)
            for i, pred in enumerate(preds):
                # joints2d_loss = self.lambda_joints2d * F.mse_loss(pred[:, :21], gts[:, :21])
                joints2d_loss = self.lambda_joints2d * mse_loss_with_mask(pred[:, 21*index:21*(index+1)], gts[:, 21*index:21*(index+1)], mask)
                final_loss += joints2d_loss
                if i == num_stack-1:
                    joint_losses["hm_joints2d_loss"] = joints2d_loss.detach().cpu()
            final_loss /= num_stack
        else:
            # joints2d_loss = self.lambda_joints2d * F.mse_loss(preds[:, :21], gts[:, :21])
            joints2d_loss = self.lambda_joints2d * mse_loss_with_mask(preds[:, 21*index:21*(index+1)], gts[:, 21*index:21*(index+1)], mask)
            final_loss = joints2d_loss
            joint_losses["hm_joints2d_loss"] = joints2d_loss.detach().cpu()
        return final_loss, joint_losses


class ManoLoss(nn.Module):
    def __init__(self, lambda_verts3d=None, lambda_joints3d=None,
                 lambda_manopose=None, lambda_manoshape=None,
                 lambda_regulshape=None, lambda_regulpose=None):
        super(ManoLoss,self).__init__()
        
        self.lambda_verts3d = lambda_verts3d
        self.lambda_joints3d = lambda_joints3d
        self.lambda_manopose = lambda_manopose
        self.lambda_manoshape = lambda_manoshape
        self.lambda_regulshape = lambda_regulshape
        self.lambda_regulpose = lambda_regulpose

    def __call__(self, preds, gts, mask=None):
        final_loss = 0
        mano_losses = {}
        if type(preds) == list:
            num_stack = len(preds)
            for i, pred in enumerate(preds):
                if self.lambda_verts3d is not None and "verts3d" in gts:
                    # mesh3d_loss = self.lambda_verts3d * F.mse_loss(pred["verts3d"], gts["verts3d"])
                    mesh3d_loss = self.lambda_verts3d * mse_loss_with_mask(pred["verts3d"], gts["verts3d"], mask)
                    final_loss += mesh3d_loss
                    if i == num_stack - 1:
                        mano_losses["mano_mesh3d_loss"] = mesh3d_loss.detach().cpu()
                if self.lambda_joints3d is not None and "joints3d" in gts:
                    # joints3d_loss = self.lambda_joints3d * F.mse_loss(pred["joints3d"], gts["joints3d"])
                    joints3d_loss = self.lambda_joints3d * mse_loss_with_mask(pred["joints3d"], gts["joints3d"], mask)
                    final_loss += joints3d_loss
                    if i == num_stack - 1:
                        mano_losses["mano_joints3d_loss"] = joints3d_loss.detach().cpu()
                if self.lambda_manopose is not None and "mano_pose" in gts:
                    # pose_param_loss = self.lambda_manopose * F.mse_loss(pred["mano_pose"], gts["mano_pose"])
                    pose_param_loss = self.lambda_manopose * mse_loss_with_mask(pred["mano_pose"], gts["mano_pose"], mask)
                    final_loss += pose_param_loss
                    if i == num_stack - 1:
                        mano_losses["manopose_loss"] = pose_param_loss.detach().cpu()
                if self.lambda_manoshape is not None and "mano_shape" in gts:
                    # shape_param_loss = self.lambda_manoshape * F.mse_loss(pred["mano_shape"], gts["mano_shape"])
                    shape_param_loss = self.lambda_manoshape * mse_loss_with_mask(pred["mano_shape"], gts["mano_shape"], mask)
                    final_loss += shape_param_loss
                    if i == num_stack - 1:
                        mano_losses["manoshape_loss"] = shape_param_loss.detach().cpu()
                if self.lambda_regulshape:
                    # shape_regul_loss = self.lambda_regulshape * F.mse_loss(pred["mano_shape"], torch.zeros_like(pred["mano_shape"]))
                    shape_regul_loss = self.lambda_regulshape * mse_loss_with_mask(pred["mano_shape"], torch.zeros_like(pred["mano_shape"]), mask)
                    final_loss += shape_regul_loss
                    if i == num_stack - 1:
                        mano_losses["regul_manoshape_loss"] = shape_regul_loss.detach().cpu()
                if self.lambda_regulpose:
                    raise NotImplementedError  # this is incorrect for augmented samples
                    # pose_regul_loss = self.lambda_regulpose * F.mse_loss(pred["mano_pose"][:, 3:], torch.zeros_like(pred["mano_pose"][:, 3:]))
                    pose_regul_loss = self.lambda_regulpose * mse_loss_with_mask(pred["mano_pose"][:, 3:], torch.zeros_like(pred["mano_pose"][:, 3:]), mask)
                    final_loss += pose_regul_loss
                    if i == num_stack - 1:
                        mano_losses["regul_manopose_loss"] = pose_regul_loss.detach().cpu()
            final_loss /= num_stack
            mano_losses["mano_total_loss"] = final_loss.detach().cpu()
        else:
            if self.lambda_verts3d is not None and "verts3d" in gts:
                # mesh3d_loss = self.lambda_verts3d * F.mse_loss(preds["verts3d"], gts["verts3d"])
                mesh3d_loss = self.lambda_verts3d * mse_loss_with_mask(preds["verts3d"], gts["verts3d"], mask)
                final_loss += mesh3d_loss
                mano_losses["mano_mesh3d_loss"] = mesh3d_loss.detach().cpu()
            if self.lambda_joints3d is not None and "joints3d" in gts:
                # joints3d_loss = self.lambda_joints3d * F.mse_loss(preds["joints3d"], gts["joints3d"])
                joints3d_loss = self.lambda_joints3d * mse_loss_with_mask(preds["joints3d"], gts["joints3d"], mask)
                final_loss += joints3d_loss
                mano_losses["mano_joints3d_loss"] = joints3d_loss.detach().cpu()
            if self.lambda_manopose is not None and "mano_pose" in gts:
                # pose_param_loss = self.lambda_manopose * F.mse_loss(preds["mano_pose"], gts["mano_pose"])
                pose_param_loss = self.lambda_manopose * mse_loss_with_mask(preds["mano_pose"], gts["mano_pose"], mask)
                final_loss += pose_param_loss
                mano_losses["manopose_loss"] = pose_param_loss.detach().cpu()
            if self.lambda_manoshape is not None and "mano_shape" in gts:
                # shape_param_loss = self.lambda_manoshape * F.mse_loss(preds["mano_shape"], gts["mano_shape"])
                shape_param_loss = self.lambda_manoshape * mse_loss_with_mask(preds["mano_shape"], gts["mano_shape"], mask)
                final_loss += shape_param_loss
                mano_losses["manoshape_loss"] = shape_param_loss.detach().cpu()
            if self.lambda_regulshape:
                # shape_regul_loss = self.lambda_regulshape * F.mse_loss(preds["mano_shape"], torch.zeros_like(preds["mano_shape"]))
                shape_regul_loss = self.lambda_regulshape * mse_loss_with_mask(preds["mano_shape"], torch.zeros_like(preds["mano_shape"]), mask)
                final_loss += shape_regul_loss
                mano_losses["regul_manoshape_loss"] = shape_regul_loss.detach().cpu()
            if self.lambda_regulpose:
                raise NotImplementedError  # this is incorrect for augmented samples
                # pose_regul_loss = self.lambda_regulpose * F.mse_loss(preds["mano_pose"][:, 3:], torch.zeros_like(preds["mano_pose"][:, 3:]))
                pose_regul_loss = self.lambda_regulpose * mse_loss_with_mask(preds["mano_pose"][:, 3:], torch.zeros_like(preds["mano_pose"][:, 3:]), mask)
                final_loss += pose_regul_loss
                mano_losses["regul_manopose_loss"] = pose_regul_loss.detach().cpu()
            mano_losses["mano_total_loss"] = final_loss.detach().cpu()
        return final_loss, mano_losses
    

class ManoLossHalf(nn.Module):
    def __init__(self, lambda_verts3d=None, lambda_joints3d=None,
                 lambda_manopose=None, lambda_manoshape=None,
                 lambda_regulshape=None, lambda_regulpose=None):
        super(ManoLossHalf,self).__init__()
        
        self.lambda_verts3d = lambda_verts3d
        self.lambda_joints3d = lambda_joints3d
        self.lambda_manopose = lambda_manopose
        self.lambda_manoshape = lambda_manoshape
        self.lambda_regulshape = lambda_regulshape
        self.lambda_regulpose = lambda_regulpose

    def __call__(self, preds, gts, mask=None, index=0):
        final_loss = 0
        mano_losses = {}
        if type(preds) == list:
            num_stack = len(preds)
            for i, pred in enumerate(preds):
                if self.lambda_verts3d is not None and "verts3d" in gts:
                    # mesh3d_loss = self.lambda_verts3d * F.mse_loss(pred["verts3d"][:, :778], gts["verts3d"][:, :778])
                    mesh3d_loss = self.lambda_verts3d * mse_loss_with_mask(pred["verts3d"][:, 778*index:778*(index+1)], gts["verts3d"][:, 778*index:778*(index+1)], mask)
                    final_loss += mesh3d_loss
                    if i == num_stack - 1:
                        mano_losses["mano_mesh3d_loss"] = mesh3d_loss.detach().cpu()
                if self.lambda_joints3d is not None and "joints3d" in gts:
                    # joints3d_loss = self.lambda_joints3d * F.mse_loss(pred["joints3d"][:, :21], gts["joints3d"][:, :21])
                    joints3d_loss = self.lambda_joints3d * mse_loss_with_mask(pred["joints3d"][:, 21*index:21*(index+1)], gts["joints3d"][:, 21*index:21*(index+1)], mask)
                    final_loss += joints3d_loss
                    if i == num_stack - 1:
                        mano_losses["mano_joints3d_loss"] = joints3d_loss.detach().cpu()
                if self.lambda_manopose is not None and "mano_pose" in gts:
                    # pose_param_loss = self.lambda_manopose * F.mse_loss(pred["mano_pose"][:, :16], gts["mano_pose"][:, :16])
                    pose_param_loss = self.lambda_manopose * mse_loss_with_mask(pred["mano_pose"][:, 16*index:16*(index+1)], gts["mano_pose"][:, 16*index:16*(index+1)], mask)
                    final_loss += pose_param_loss
                    if i == num_stack - 1:
                        mano_losses["manopose_loss"] = pose_param_loss.detach().cpu()
                if self.lambda_manoshape is not None and "mano_shape" in gts:
                    # shape_param_loss = self.lambda_manoshape * F.mse_loss(pred["mano_shape"][:, :10], gts["mano_shape"][:, :10])
                    shape_param_loss = self.lambda_manoshape * mse_loss_with_mask(pred["mano_shape"][:, 10*index:10*(index+1)], gts["mano_shape"][:, 10*index:10*(index+1)], mask)
                    final_loss += shape_param_loss
                    if i == num_stack - 1:
                        mano_losses["manoshape_loss"] = shape_param_loss.detach().cpu()
                if self.lambda_regulshape:
                    # shape_regul_loss = self.lambda_regulshape * F.mse_loss(pred["mano_shape"][:, :10], torch.zeros_like(pred["mano_shape"][:, :10]))
                    shape_regul_loss = self.lambda_regulshape * mse_loss_with_mask(pred["mano_shape"][:, 10*index:10*(index+1)], torch.zeros_like(pred["mano_shape"][:, 10*index:10*(index+1)]), mask)
                    final_loss += shape_regul_loss
                    if i == num_stack - 1:
                        mano_losses["regul_manoshape_loss"] = shape_regul_loss.detach().cpu()
                if self.lambda_regulpose:
                    # pose_regul_loss = self.lambda_regulpose * F.mse_loss(pred["mano_pose"][:, 3:16], torch.zeros_like(pred["mano_pose"][:, 3:16]))
                    pose_regul_loss = self.lambda_regulpose * mse_loss_with_mask(pred["mano_pose"][:, 16*index + 3 : 16*(index+1)], torch.zeros_like(pred["mano_pose"][:, 16*index + 3 : 16*(index+1)]), mask)
                    final_loss += pose_regul_loss
                    if i == num_stack - 1:
                        mano_losses["regul_manopose_loss"] = pose_regul_loss.detach().cpu()
            final_loss /= num_stack
            mano_losses["mano_total_loss"] = final_loss.detach().cpu()
        else:
            if self.lambda_verts3d is not None and "verts3d" in gts:
                # mesh3d_loss = self.lambda_verts3d * F.mse_loss(preds["verts3d"][:, :778], gts["verts3d"][:, :778])
                mesh3d_loss = self.lambda_verts3d * mse_loss_with_mask(preds["verts3d"][:, 778*index:778*(index+1)], gts["verts3d"][:, 778*index:778*(index+1)], mask)
                final_loss += mesh3d_loss
                mano_losses["mano_mesh3d_loss"] = mesh3d_loss.detach().cpu()
            if self.lambda_joints3d is not None and "joints3d" in gts:
                # joints3d_loss = self.lambda_joints3d * F.mse_loss(preds["joints3d"][:, :21], gts["joints3d"][:, :21])
                joints3d_loss = self.lambda_joints3d * mse_loss_with_mask(preds["joints3d"][:, 21*index:21*(index+1)], gts["joints3d"][:, 21*index:21*(index+1)], mask)
                final_loss += joints3d_loss
                mano_losses["mano_joints3d_loss"] = joints3d_loss.detach().cpu()
            if self.lambda_manopose is not None and "mano_pose" in gts:
                # pose_param_loss = self.lambda_manopose * F.mse_loss(preds["mano_pose"][:, :16], gts["mano_pose"][:, :16])
                pose_param_loss = self.lambda_manopose * mse_loss_with_mask(preds["mano_pose"][:, 16*index:16*(index+1)], gts["mano_pose"][:, 16*index:16*(index+1)], mask)
                final_loss += pose_param_loss
                mano_losses["manopose_loss"] = pose_param_loss.detach().cpu()
            if self.lambda_manoshape is not None and "mano_shape" in gts:
                # shape_param_loss = self.lambda_manoshape * F.mse_loss(preds["mano_shape"][:, :10], gts["mano_shape"][:, :10])
                shape_param_loss = self.lambda_manoshape * mse_loss_with_mask(preds["mano_shape"][:, 10*index:10*(index+1)], gts["mano_shape"][:, 10*index:10*(index+1)], mask)
                final_loss += shape_param_loss
                mano_losses["manoshape_loss"] = shape_param_loss.detach().cpu()
            if self.lambda_regulshape:
                # shape_regul_loss = self.lambda_regulshape * F.mse_loss(preds["mano_shape"][:, :10], torch.zeros_like(preds["mano_shape"][:, :10]))
                shape_regul_loss = self.lambda_regulshape * mse_loss_with_mask(preds["mano_shape"][:, 10*index:10*(index+1)], torch.zeros_like(preds["mano_shape"][:, 10*index:10*(index+1)]), mask)
                final_loss += shape_regul_loss
                mano_losses["regul_manoshape_loss"] = shape_regul_loss.detach().cpu()
            if self.lambda_regulpose:
                # pose_regul_loss = self.lambda_regulpose * F.mse_loss(preds["mano_pose"][:, 3:16], torch.zeros_like(preds["mano_pose"][:, 3:16]))
                pose_regul_loss = self.lambda_regulpose * mse_loss_with_mask(preds["mano_pose"][:, 16*index + 3 : 16*(index+1)], torch.zeros_like(preds["mano_pose"][:, 16*index + 3 : 16*(index+1)]), mask)
                final_loss += pose_regul_loss
                mano_losses["regul_manopose_loss"] = pose_regul_loss.detach().cpu()
            mano_losses["mano_total_loss"] = final_loss.detach().cpu()
        return final_loss, mano_losses
    

class ObjectLoss(nn.Module):
    def __init__(self, obj_reg_loss_weight, obj_conf_loss_weight=None, obj_loss_func=None):
        super(ObjectLoss,self).__init__()
        
        if obj_conf_loss_weight is None and obj_reg_loss_weight is not None:
            obj_conf_loss_weight = obj_reg_loss_weight / 5
        if obj_loss_func is None:
            obj_loss_func = torch.nn.L1Loss()
        self.obj_loss_func = obj_loss_func
        self.obj_conf_loss_weight = obj_conf_loss_weight
        self.obj_reg_loss_weight = obj_reg_loss_weight

    def __call__(self, obj_p2d_gt, obj_mask, obj_pred, obj_lossmask=None):
        obj_losses = {}
        # get predictions for output
        reg_px = obj_pred[0]
        reg_py = obj_pred[1]
        reg_conf = obj_pred[2]
        mask_front = obj_mask.repeat(21, 1, 1, 1).permute(1, 2, 3, 0).contiguous()
        #print(mask_front.shape)
        reg_py = reg_py * mask_front
        reg_px = reg_px * mask_front
        reg_label = obj_p2d_gt.repeat(32, 32, 1, 1, 1).permute(2, 0, 1, 3, 4).contiguous()
        reg_label_x = reg_label[:, :, :, :, 0]
        reg_label_y = reg_label[:, :, :, :, 1]
        reg_label_x = reg_label_x * mask_front
        reg_label_y = reg_label_y * mask_front

        # confidence regression result
        bias = torch.sqrt((reg_py - reg_label_y) ** 2 + (reg_px - reg_label_x) ** 2)
        conf_target = torch.exp(-1 * bias) * mask_front
        conf_target = conf_target.detach()

        if obj_lossmask is None:
            reg_loss = self.obj_loss_func(reg_px, reg_label_x) + self.obj_loss_func(reg_py, reg_label_y)
            conf_loss = self.obj_loss_func(reg_conf, conf_target)
        else:
            obj_lossmask = obj_lossmask.view(-1, 1, 1, 1)
            reg_px = reg_px * obj_lossmask
            reg_py = reg_py * obj_lossmask
            reg_label_x = reg_label_x * obj_lossmask
            reg_label_y = reg_label_y * obj_lossmask
            reg_conf = reg_conf * obj_lossmask
            conf_target = conf_target * obj_lossmask
            reg_loss = self.obj_loss_func(reg_px, reg_label_x) + self.obj_loss_func(reg_py, reg_label_y)
            conf_loss = self.obj_loss_func(reg_conf, conf_target)
            if obj_lossmask.sum() != 0:
                reg_loss *= reg_px.shape[0] / obj_lossmask.sum()
                conf_loss *= reg_px.shape[0] / obj_lossmask.sum()

        reg_loss = self.obj_reg_loss_weight * reg_loss
        conf_loss = self.obj_conf_loss_weight * conf_loss
        obj_losses["obj_reg_loss"] = reg_loss.detach().cpu()
        obj_losses["obj_conf_loss"] = conf_loss.detach().cpu()
        final_loss = reg_loss + conf_loss
        return final_loss, obj_losses
    
    
class GraspingObjectLoss(nn.Module):
    def __init__(self, obj_reg_loss_weight, obj_conf_loss_weight=None, obj_loss_func=None):
        super(GraspingObjectLoss,self).__init__()
        
        if obj_conf_loss_weight is None and obj_reg_loss_weight is not None:
            obj_conf_loss_weight = obj_reg_loss_weight / 5
        if obj_loss_func is None:
            obj_loss_func = torch.nn.L1Loss()
        self.obj_loss_func = obj_loss_func
        self.obj_conf_loss_weight = obj_conf_loss_weight
        self.obj_reg_loss_weight = obj_reg_loss_weight

    def __call__(self, obj_p2d_gt, obj_mask, obj_pred, obj_grasping_mask, obj_lossmask=None): # obj_p2d_gt: torch.Size([16, 21, 2])
        obj_losses = {}
        # get predictions for output
        reg_px = obj_pred[0]  # torch.Size([B, 32, 32, 21])
        reg_py = obj_pred[1]  # torch.Size([B, 32, 32, 21])
        reg_conf = obj_pred[2]  # torch.Size([B, 32, 32, 21])
        mask_front = obj_mask.repeat(21, 1, 1, 1).permute(1, 2, 3, 0).contiguous() # torch.Size([B, 32, 32, 21])
        #print(mask_front.shape)
        reg_py = reg_py * mask_front
        reg_px = reg_px * mask_front
        reg_label = obj_p2d_gt.repeat(32, 32, 1, 1, 1).permute(2, 0, 1, 3, 4).contiguous()  # torch.Size([B, 32, 32, 21, 2])
        reg_label_x = reg_label[:, :, :, :, 0]
        reg_label_y = reg_label[:, :, :, :, 1]
        reg_label_x = reg_label_x * mask_front
        reg_label_y = reg_label_y * mask_front

        # confidence regression result
        bias = torch.sqrt((reg_py - reg_label_y) ** 2 + (reg_px - reg_label_x) ** 2)
        conf_target = torch.exp(-1 * bias) * mask_front
        conf_target = conf_target.detach()

        if obj_lossmask is None:
            grasping_num = torch.count_nonzero(obj_grasping_mask)
            if grasping_num > 0:
                obj_grasping_mask = obj_grasping_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                reg_px_masked = torch.masked_select(reg_px, obj_grasping_mask)
                reg_label_x_masked = torch.masked_select(reg_label_x, obj_grasping_mask)
                reg_py_masked = torch.masked_select(reg_py, obj_grasping_mask)
                reg_label_y_masked = torch.masked_select(reg_label_y, obj_grasping_mask)
                reg_conf_masked = torch.masked_select(reg_conf, obj_grasping_mask)
                conf_target_masked = torch.masked_select(conf_target, obj_grasping_mask)
                
                reg_loss = self.obj_loss_func(reg_px_masked, reg_label_x_masked) + self.obj_loss_func(reg_py_masked, reg_label_y_masked)
                conf_loss = self.obj_loss_func(reg_conf_masked, conf_target_masked)
            else:
                reg_loss = torch.tensor(0).cuda()
                conf_loss = torch.tensor(0).cuda()
        else:
            obj_lossmask = obj_lossmask.view(-1, 1, 1, 1)
            reg_px = reg_px * obj_lossmask
            reg_py = reg_py * obj_lossmask
            reg_label_x = reg_label_x * obj_lossmask
            reg_label_y = reg_label_y * obj_lossmask
            reg_conf = reg_conf * obj_lossmask
            conf_target = conf_target * obj_lossmask
            reg_loss = self.obj_loss_func(reg_px, reg_label_x) + self.obj_loss_func(reg_py, reg_label_y)
            conf_loss = self.obj_loss_func(reg_conf, conf_target)
            if obj_lossmask.sum() != 0:
                reg_loss *= reg_px.shape[0] / obj_lossmask.sum()
                conf_loss *= reg_px.shape[0] / obj_lossmask.sum()

        reg_loss = self.obj_reg_loss_weight * reg_loss
        conf_loss = self.obj_conf_loss_weight * conf_loss
        obj_losses["obj_reg_loss"] = reg_loss.detach().cpu()
        obj_losses["obj_conf_loss"] = conf_loss.detach().cpu()
        final_loss = reg_loss + conf_loss
        return final_loss, obj_losses


class CrosssEntropyLoss(nn.Module):
    def __init__(self, weight):
        super(CrosssEntropyLoss,self).__init__()
        self.weight = weight
        
    def __call__(self, pred_logits, gt_label, mask):
        pos_num = torch.count_nonzero(mask)
        if pos_num > 0:
            pred_logits_masked = pred_logits[mask]
            gt_label_masked = gt_label[mask]
            loss = F.cross_entropy(pred_logits_masked, gt_label_masked)
        else:
            loss = torch.tensor(0).cuda()
            
        return self.weight*loss


# only create it once to avoid warning from smplx
mano_mesh = None

def compute_loss(cfg, input, output, epoch = -1):
    loss = {}

    if cfg.loss.name == "hand_occ_net":
        loss["mano_verts"] = 1e4 * F.mse_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        loss["mano_joints"] = 1e4 * F.mse_loss(output["pred_joints3d_cam"], output["gt_joints3d_cam"])
        loss["mano_pose"] = 10 * F.mse_loss(output["pred_mano_pose"], output["gt_mano_pose"])
        loss["mano_shape"] = 0.1 * F.mse_loss(output["pred_mano_shape"], output["gt_mano_shape"])
        loss["joints_img"] = 100 * F.mse_loss(output["pred_joints_img"], output["joints_img"])

    elif cfg.loss.name == "semi_hand":
        loss["mano_verts"] = 1e4 * F.mse_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        loss["mano_joints"] = 1e4 * F.mse_loss(output["pred_joints3d_cam"], output["gt_joints3d_cam"])
        loss["mano_pose"] = 10 * F.mse_loss(output["pred_mano_pose"], output["gt_mano_pose"])
        loss["mano_shape"] = 0.1 * F.mse_loss(output["pred_mano_shape"], output["gt_mano_shape"])
        loss["joints_img"] = 1e2 * F.mse_loss(output["pred_joints_img"], output["joints_img"])
        loss["shape_regul"] = 1e2 * F.mse_loss(output["pred_mano_shape"], torch.zeros_like(output["pred_mano_shape"]))
        loss["pose_regul"] = 1 * F.mse_loss(output["pred_mano_pose"][:, 3:], torch.zeros_like(output["pred_mano_pose"][:, 3:]))
        
    elif cfg.loss.name == "semi_hand_object":
        loss["mano_verts"] = 1e4 * F.mse_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        loss["mano_joints"] = 1e4 * F.mse_loss(output["pred_joints3d_cam"], output["gt_joints3d_cam"])
        loss["mano_pose"] = 10 * F.mse_loss(output["pred_mano_pose"], output["gt_mano_pose"])
        loss["mano_shape"] = 0.1 * F.mse_loss(output["pred_mano_shape"], output["gt_mano_shape"])
        loss["joints_img"] = 1e2 * F.mse_loss(output["pred_joints_img"], output["joints_img"])
        loss["shape_regul"] = 1e2 * F.mse_loss(output["pred_mano_shape"], torch.zeros_like(output["pred_mano_shape"]))
        loss["pose_regul"] = 1 * F.mse_loss(output["pred_mano_pose"][:, 3:], torch.zeros_like(output["pred_mano_pose"][:, 3:]))
        
        # add object loss
        obj_mask = input["obj_mask"] # torch.Size([B, 32*2, 32])
        obj_p2d_gt = input["obj_p2d"]  # torch.Size([B, 21*2, 2])
        preds_obj = output['preds_obj']
        
        lambda_objects = 5e2
        object_loss = ObjectLoss(obj_reg_loss_weight=lambda_objects)
        obj_total_loss, obj_losses = object_loss(obj_p2d_gt, obj_mask, preds_obj)
        loss["obj_total_loss"] = obj_total_loss
        
    elif cfg.loss.name == "mobrecon":
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        loss["verts_loss"] = F.l1_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        loss["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], output["joints_img"])
        loss["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        loss["edge_loss"] = edge_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])

    elif cfg.loss.name == "h2onet":
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        loss["verts_wo_gr_loss"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])

        if cfg.train.w_gr is True:  # train with global rotation
            loss["verts_w_gr_loss"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
            loss["joints_w_gr_loss"] = F.l1_loss(output["pred_joints3d_w_gr"], output["gt_joints3d_w_gr"])
            loss["glob_rot_loss"] = F.mse_loss(output["pred_glob_rot_mat"], output["gt_glob_rot_mat"])

        loss["joints_wo_gr_loss"] = F.l1_loss(output["pred_joints3d_wo_gr"], output["gt_joints3d_wo_gr"])
        loss["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], output["joints_img"])
        loss["normal_wo_gr_loss"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        loss["edge_wo_gr_loss"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])

    elif cfg.loss.name == "hflnet":
        mano_lambda_verts3d = cfg.loss.weight.get('mano_lambda_verts3d', 1e4)
        mano_lambda_joints3d = cfg.loss.weight.get('mano_lambda_joints3d', 1e4)
        mano_lambda_manopose = cfg.loss.weight.get('mano_lambda_manopose', 10)
        mano_lambda_manoshape = cfg.loss.weight.get('mano_lambda_manoshape', 0.1)
        lambda_joints2d = cfg.loss.weight.get('lambda_joints2d', 1e2)
        lambda_objects = cfg.loss.weight.get('lambda_objects', 5e2)
        
        mano_loss = ManoLoss(lambda_verts3d=mano_lambda_verts3d,
                                  lambda_joints3d=mano_lambda_joints3d,
                                  lambda_manopose=mano_lambda_manopose,
                                  lambda_manoshape=mano_lambda_manoshape)
        
        joint2d_loss = Joint2DLoss(lambda_joints2d=lambda_joints2d)
        
        # object loss
        object_loss = ObjectLoss(obj_reg_loss_weight=lambda_objects)
        
        joints_uv = input['joints_img']
        obj_mask = input["obj_mask"]
        obj_p2d_gt = input["obj_p2d"]
        
        gt_mano_results = output['gt_mano_results']
        pred_mano_results = output['pred_mano_results']
        preds_joints2d = output['preds_joints2d']
        preds_obj = output['preds_obj']
        
        mano_total_loss, mano_losses = mano_loss(pred_mano_results, gt_mano_results)
        joint2d_loss, joint2d_losses = joint2d_loss(preds_joints2d, joints_uv)
        obj_total_loss, obj_losses = object_loss(obj_p2d_gt, obj_mask, preds_obj)
        loss["mano_total_loss"] = mano_total_loss
        loss["joint2d_loss"] = joint2d_loss
        loss["obj_total_loss"] = obj_total_loss
        
    elif cfg.loss.name == "hflnet_grasping_objconf": # only has obj loss when hand grasping object
        mano_lambda_verts3d = cfg.loss.weight.get('mano_lambda_verts3d', 1e4)
        mano_lambda_joints3d = cfg.loss.weight.get('mano_lambda_joints3d', 1e4)
        mano_lambda_manopose = cfg.loss.weight.get('mano_lambda_manopose', 10)
        mano_lambda_manoshape = cfg.loss.weight.get('mano_lambda_manoshape', 0.1)
        lambda_joints2d = cfg.loss.weight.get('lambda_joints2d', 1e2)
        lambda_objects = cfg.loss.weight.get('lambda_objects', 5e2)
        
        # by default, we'll compute all loss for both samples.
        contrastive = cfg.data.get("contrastive", False)
        
        train_part = cfg.train.get("train_part", [0, 1])
        if not contrastive:  # only use the first sample for computing loss when data loader is not contrastive
            train_part = [0]
        
        # only use the half sample for computing loss
        mano_loss = ManoLossHalf(lambda_verts3d=mano_lambda_verts3d,
                                lambda_joints3d=mano_lambda_joints3d,
                                lambda_manopose=mano_lambda_manopose,
                                lambda_manoshape=mano_lambda_manoshape)
        
        joint2d_loss = Joint2DLossHalf(lambda_joints2d=lambda_joints2d)
        
        # object loss
        object_loss = GraspingObjectLoss(obj_reg_loss_weight=lambda_objects)
        
        # out obj classification loss
        out_obj_conf_weight = cfg.loss.weight.get('out_obj_conf_weight', 10)
        classificaition_loss = CrosssEntropyLoss(weight=out_obj_conf_weight)
        
        hand_occ_cls_weight = cfg.loss.weight.get('hand_occ_cls_weight', 1)
        occ_classificaition_loss = CrosssEntropyLoss(weight=hand_occ_cls_weight)
        
        obj_cls_weight = cfg.loss.weight.get('obj_cls_weight', 1)
        obj_cls_classificaition_loss = CrosssEntropyLoss(weight=obj_cls_weight)
        
        if 'joints_img' in output:
            joints_uv = output['joints_img']  # torch.Size([B, 21, 2])  # NOTE: specially handled for speeding up the training process when only using half samples
        else:
            joints_uv = input['joints_img']
            
        obj_mask = input["obj_mask"] # torch.Size([B, 32*2, 32])
        obj_p2d_gt = input["obj_p2d"]  # torch.Size([B, 21*2, 2])
        is_filtered = input['is_filtered']  # torch.Size([B])
        
        # get the batch size and the number of sub-samples in each sample
        B = joints_uv.shape[0]
        if contrastive:
            L = 2
        else:
            L = 1
        
        gt_mano_results = output['gt_mano_results']  # might be None
        pred_mano_results = output['pred_mano_results']
        preds_joints2d = output['preds_joints2d']
        preds_obj = output['preds_obj']
        
        hand_loss_mask = torch.zeros((B, L), dtype=torch.bool).cuda()
        hand_loss_mask[: , 0] = True
        
        if L > 1:  # the generated samples will also be used to compute hand loss
            hand_loss_mask[: , 1] = True
            
        mano_total_loss_contr = 0
        joint2d_loss_contr = 0
        for i in range(len(train_part)):
            if gt_mano_results is not None:
                mano_total_loss, mano_losses = mano_loss(pred_mano_results, gt_mano_results, mask=hand_loss_mask[:, train_part[i]], index=i)
            else:
                joints3d = input.get("joints_coord_cam", None)
                if joints3d is not None:  # has gt 3d joints, but no mano param
                    mano_total_loss, mano_losses = mano_loss(pred_mano_results, {"joints3d": joints3d}, mask=hand_loss_mask[:, train_part[i]], index=i)
                else:  # no supervison on 3d hand
                    mano_total_loss = torch.tensor(0.0).cuda()
                
            joint2d_total_loss, joint2d_losses = joint2d_loss(preds_joints2d, joints_uv, mask=hand_loss_mask[:, train_part[i]], index=i)
            
            mano_total_loss_contr += mano_total_loss
            joint2d_loss_contr += joint2d_total_loss
            
        # we use both samples for computing loss, divided by two
        mano_total_loss_contr /= float(len(train_part))
        joint2d_loss_contr /= float(len(train_part))
            
        loss["mano_total_loss"] = mano_total_loss_contr
        loss["joint2d_loss"] = joint2d_loss_contr
        
        # object loss
        obj_total_loss_contr = 0
        # object interaction classification loss
        out_obj_conf_loss_contr = 0
        # occlusion label classification loss
        hand_occ_cls_loss_contr = torch.tensor(0.0).cuda() # , requires_grad=True
        
        obj_cls_loss_contr = torch.tensor(0.0).cuda()
        
        gt_grasping = input["gt_grasping"]
        gt_cls = gt_grasping.to(torch.int64)  # (B, 2) 0: no-obj, 1: obj
        pred_obj_conf = output['pred_obj_conf']  # (B, 2*2)
        
        if 'pred_hand_occ' in output:
            pred_hand_occ = output['pred_hand_occ']  # (B, 2*2)
            gt_hand_occ_cls = input["gt_occ_cls"].to(torch.int64)  # (B, 2) 0: non-occluded, 1: occluded
        
        gt_grasping_mask = gt_grasping.bool()
        
        # used for computing object loss
        reg_obj_mask = torch.zeros_like(hand_loss_mask)
        # for grasping ones, only use the 1st sample to calculate object loss
        reg_obj_mask[:, 0][gt_grasping_mask[:, 0]] = True
            
        
        # used for computing classification loss
        reg_cls_mask = torch.zeros_like(hand_loss_mask)
         # For all the pairs, only use the 1st sample
        reg_cls_mask[:, 0] = True
            
        # only compute additonal loss and contrastive loss for real-augmented samples.
        for i in range(len(train_part)):
            obj_total_loss, obj_losses = object_loss(obj_p2d_gt[:, train_part[i]*21:(train_part[i]+1)*21],
                                                        obj_mask[:, train_part[i]*32:(train_part[i]+1)*32], 
                                                        [preds_obj[0][:, i*32:(i+1)*32], preds_obj[1][:, i*32:(i+1)*32], preds_obj[2][:, i*32:(i+1)*32]], 
                                                        reg_obj_mask[:, train_part[i]])
            
            # if two contrastive sample, the loss is zero. If normal sample, just calculate half
            obj_total_loss_contr += obj_total_loss
            
            out_obj_conf_loss_contr += classificaition_loss(pred_obj_conf[:, i*2:(i+1)*2], gt_cls[:, train_part[i]], reg_cls_mask[:, train_part[i]]) #...
            
            if 'pred_hand_occ' in output:
                hand_occ_cls_loss_contr += occ_classificaition_loss(pred_hand_occ[:, i*2:(i+1)*2], gt_hand_occ_cls[:, train_part[i]], reg_cls_mask[:, train_part[i]])
                
        loss["obj_total_loss"] = obj_total_loss_contr
        loss["out_obj_conf_loss"] = out_obj_conf_loss_contr
        loss["hand_occ_cls_loss"] = hand_occ_cls_loss_contr
        loss["obj_cls_loss"] = obj_cls_loss_contr
        
        if contrastive:
            # feature-level contrastive loss
            # by default, we only compute contrastive loss for grasping sample pairs
            contrastive_mask = gt_grasping_mask[:, 0]
            # filter out the bad case for pairwise distillation
            contrastive_mask = torch.logical_and(contrastive_mask, ~(is_filtered.bool()))
            
            if torch.count_nonzero(contrastive_mask) > 0:
                loss_func = cfg.loss.get('loss_func', 'l1_loss')
                if loss_func == 'l1_loss':
                    loss_func = F.l1_loss
                elif loss_func == 'mse_loss':
                    loss_func = F.mse_loss
                elif loss_func == 'smooth_l1_loss':
                    loss_func = F.smooth_l1_loss
                else:
                    raise NotImplementedError
                    
                # accumulate all the features
                for k, v in cfg.loss.weight.items():
                    # feat = output[k]
                    feat = output[k][contrastive_mask]
                    d = feat.shape[1] // 2
                    
                    # no backward for generated feature
                    loss[k] = v * loss_func(feat[:, :d], feat[:, d:].detach(), reduction='none').mean(dim=tuple(range(1, feat.dim())))
            else:
                for k, v in cfg.loss.weight.items():
                    loss[k] = torch.tensor(0.0).cuda()
    
    elif cfg.loss.name == "grasp_vit_classifier":
        pred_cls = output["pred_obj_conf"]  # (B, C)
        gt_cls = input["gt_grasping"].squeeze(-1).to(torch.int64)  # (B) 0: non-grasping, 1: grasping
        loss["cls_loss"] = F.cross_entropy(pred_cls, gt_cls) # gt_labels: torch.Size([B, 2])
    
    elif cfg.loss.name == "kypt_transformer":
        # Get all the losses for all the predictions
        loss = {}
        
        def render_gaussian_heatmap(joint_coord, joint_valid ):
            x = torch.arange(cfg.model.output_hm_shape[2])
            y = torch.arange(cfg.model.output_hm_shape[1])
            yy,xx = torch.meshgrid(y,x)
            xx = xx[None,None,:,:].cuda().float(); yy = yy[None,None,:,:].cuda().float()
            
            if cfg.model.hand_type == 'both':
                joint_coord1 = joint_coord # N x 42 x 3
                joint_valid1 = joint_valid
            elif cfg.model.hand_type == 'right':
                joint_coord1 = joint_coord[:,:21]  # N x 21 x 3
                joint_valid1 = joint_valid[:,:21]
            elif cfg.model.hand_type == 'left':
                joint_coord1 = joint_coord[:,21:]  # N x 21 x 3
                joint_valid1 = joint_valid[:, 21:]

            x = joint_coord1[:,:,0,None,None]; y = joint_coord1[:,:,1,None,None]
            heatmap = torch.exp(-(((xx-x)/cfg.model.sigma)**2)/2 -(((yy-y)/cfg.model.sigma)**2)/2) * joint_valid1[:,:,None,None] # N x 42 x h x w
            heatmap = torch.sum(heatmap, 1)
            heatmap = heatmap * 255

            return heatmap
        
        # define loss functions
        joint_heatmap_loss = JointHeatmapLoss()
        obj_seg_loss = ObjSegLoss()
        pose_loss = PoseLoss(cfg.model)
        rel_trans_loss = RelTransLoss()
        joints_loss = JointLoss(cfg.model)
        vertex_loss = VertexLoss(cfg.model)
        shape_reg = ShapeRegularize()
        joint_class_loss = JointClassificationLoss(cfg.model)
        hand_type_loss = HandTypeLoss()
        cam_param_loss = CameraParamLoss(cfg.model)
        shape_loss = ShapeLoss()
        if cfg.model.use_bottleneck_hand_type:
            bottleneck_hand_type_loss = BottleNeckHandTypeLoss()
        
        global mano_mesh
        if mano_mesh is None:
            mano_mesh = ManoMesh(cfg.model)  
        
        if cfg.model.predict_type == 'vectors':
            joint_vecs_loss = JointVectorsLoss(cfg.model)  # never been used
            if cfg.model.predict_2p5d:
                joint_2p5d_loss = Joints2p5dLoss(cfg.model)

        obj_pose_loss = ObjPoseLoss(cfg.model)
        
        # fetch input/output
        obj_corner_proj = output['obj_corner_proj']
        rel_trans = output['rel_trans']
        cam_param = output['cam_param']
        mask_np = output['mask_np']
        joint_class = output['joint_class']
        peak_joints_map_batch = output['peak_joints_map_batch']
        transformer_out = output['transformer_out']
        hand_type = output['hand_type']
        obj_rot = output['obj_rot']
        obj_trans = output['obj_trans']
        obj_trans_left = output['obj_trans_left']
        
        pose = output['pose_out']
        shape = output['shape_out']
        joint_heatmap_out = output['joint_heatmap_out']
        obj_seg_out = output['obj_seg_pred_out']
        # obj_kps_coord_gt = input['obj_kps_coord']
        bottleneck_hand_type = output['hand_type_out']
        
        # import ipdb; ipdb.set_trace()
        
        if cfg.model.predict_type == 'angles':
            joints_pred = mano_mesh.get_mano_mesh(pose, shape, rel_trans, input['root_valid'], cam_param)
        elif cfg.model.predict_type == 'vectors':
            joints_pred = {}
            if cfg.model.predict_2p5d:
                joint_2p5d_hm = output['joint_2p5d_hm']
                loss['joint_2p5d_hm'] = joint_2p5d_loss(joint_2p5d_hm, input['joint_coord'], input['joint_valid'])
                # out['joint_2p5d_out'] = torch.argmax(joint_2p5d_hm[-1], dim=-1).permute(1,0,2) # N x 42 x 3
            else:
                joint_vecs = output['joint_vecs']
                
                loss['joint_vec'], loss['joints_loss'], loss['joints2d_loss'], joints_pred['joints_right'], joints_pred['joints_left'] = \
                joint_vecs_loss(joint_vecs, input['joint_cam_no_trans'], input['joint_valid'], input['joint_coord'][:,:,:2],
                                     cam_param, rel_trans)
                # out['joint_3d_right_out'] = joints_pred['joints_right']
                # out['joint_3d_left_out'] = joints_pred['joints_left']
                joints_pred['joints2d_right'] = joints_pred['joints_right'][:, :, :, :2] * cam_param[:, :, :1].unsqueeze(2) + cam_param[:, :,
                                                                                                  1:].unsqueeze(2)
                joints_pred['joints2d_left'] = joints_pred['joints_left'][:, :, :, :2] * cam_param[:, :, :1].unsqueeze(2) + cam_param[:, :,
                                                                                                    1:].unsqueeze(2)


        target_joint_heatmap = render_gaussian_heatmap(input['joint_coord'], input['joint_valid'])
        loss['joint_heatmap'] = joint_heatmap_loss(joint_heatmap_out, target_joint_heatmap, input['joint_valid'], input['hm_valid'])

        if cfg.model.has_object:
            target_obj_kps_heatmap = input['obj_seg']
            # out['obj_kps_gt_out'] = target_obj_kps_heatmap
            loss['obj_seg'] = obj_seg_loss(obj_seg_out, target_obj_kps_heatmap)
            # target_obj_kps_3d = input['obj_kps_3d']
        else:
            target_obj_kps_heatmap = None
            target_obj_kps_3d = None

        if cfg.model.enc_layers>0:
            loss['cls'], row_inds_batch_list, asso_inds_batch_list \
                = joint_class_loss(output['joint_loc_pred_np'], input['joint_coord'][:,:,:2],
                                        mask_np, joint_class.permute(0,2,1,3), input['joint_valid'], peak_joints_map_batch,
                                        input['joint_cam_no_trans'],
                                        target_obj_kps_heatmap.cpu().numpy() if cfg.model.has_object else None,
                                        # obj_kps_coord_gt.cpu().numpy() if cfg.model.has_object else None,
                                        # target_obj_kps_3d.cpu().numpy() if cfg.model.has_object else None
                                        )
        else:
            loss['cls'] = torch.zeros((1,)).to(transformer_out.device)


        if cfg.model.use_bottleneck_hand_type:
            loss['hand_type'] = bottleneck_hand_type_loss(bottleneck_hand_type, input['hand_type'], input['hand_type_valid'])
        else:
            if cfg.model.hand_type == 'both':
                loss['hand_type'] = hand_type_loss(hand_type, input['hand_type'],
                                                            input['hand_type_valid'])

        if cfg.model.predict_type == 'angles':
            loss['pose'] = pose_loss(pose, input['mano_pose'], input['mano_valid'])
            loss['shape_reg'] = shape_reg(shape)
            loss['shape_loss'] = shape_loss(input['mano_shape'], shape, input['mano_valid'])
            loss['joints_loss'], loss['joints2d_loss'] = joints_loss(rel_trans, input['joint_cam_no_trans'],
                                                   input['joint_valid'], input['joint_coord'][:,:,:2], input['root_valid'],
                                                                          joints_pred, cam_param)  # TODO: joint_cam comes from gt, and it  might be replaced as the joints regressed by mano
            # original zero vertex loss
            loss['vertex_loss'] = torch.zeros((1,)).to(pose.device)
            
            # my implementation
            # loss['vertex_loss'] = vertex_loss(input['verts'], input['mano_valid'], joints_pred)


        if cfg.model.hand_type == 'both':
            loss['rel_trans'] = rel_trans_loss(rel_trans, input['rel_trans_hands_rTol'],
                                                    input['root_valid'])



        if cfg.model.use_2D_loss and ((not cfg.model.predict_2p5d) or (cfg.model.predict_type=='angles')) :
            loss['cam_scale'], loss['cam_trans'], cam_param_gt = cam_param_loss(input['joint_cam_no_trans'], input['joint_valid'],
                                                                       input['joint_coord'][:, :, :2], cam_param,
                                                                       input['root_valid'], rel_trans)

        if cfg.model.has_object:
            # import ipdb; ipdb.set_trace()
            loss['obj_corners'], loss['obj_rot'], loss['obj_trans'],\
            loss['obj_corners_proj'], loss['obj_weak_proj'] = obj_pose_loss(obj_rot, obj_trans, obj_trans_left, rel_trans, input['obj_rot'],  # input['obj_rot'] is axis angle
                                                                                         input['rel_obj_trans'],
                                                                                         input['rel_trans_hands_rTol'],
                                                                                         input['obj_bb_rest'],
                                                                                         input['obj_pose_valid'],
                                                                                         obj_corner_proj,
                                                                                        #  output['obj_corner_proj'],
                                                                                         input['obj_corners_coord'],
                                                                                         input['obj_id'], input['root_valid'], cam_param, cam_param_gt[0])
            
        
        
        loss['joint_heatmap'] *= cfg.model.hm_weight
        if cfg.model.has_object:
            loss['obj_seg'] *= cfg.model.obj_hm_weight

            loss['obj_rot'] *= cfg.model.obj_rot_weight
            loss['obj_trans'] *= cfg.model.obj_trans_weight
            loss['obj_corners'] *= cfg.model.obj_corner_weight
            loss['obj_corners_proj'] *= cfg.model.obj_corner_proj_weight
            loss['obj_weak_proj'] *= cfg.model.obj_weak_proj_weight


        if cfg.model.hand_type == 'both':
            loss['rel_trans'] *= cfg.model.rel_trans_weight
            loss['hand_type'] *= cfg.model.hand_type_weight

        if cfg.model.predict_type == 'angles':
            loss['pose'] *= cfg.model.pose_weight
            loss['shape_reg'] *= cfg.model.shape_reg_weight
            loss['vertex_loss'] *= cfg.model.vertex_weight
            loss['shape_loss'] *= cfg.model.shape_weight
            loss['joints_loss'] *= cfg.model.joint_weight
            loss['joints2d_loss'] *= cfg.model.joint_2d_weight
        elif cfg.model.predict_type == 'vectors':
            if cfg.model.predict_2p5d:
                loss['joint_2p5d_hm'] *= cfg.model.joint_2p5d_weight
            else:
                loss['joint_vec'] *= cfg.model.joint_vec_weight
                loss['joints_loss'] *= cfg.model.joint_weight
                loss['joints2d_loss'] *= cfg.model.joint_2d_weight

        loss['cls'] *= cfg.model.cls_weight

        if cfg.model.use_2D_loss and ((not cfg.model.predict_2p5d) or (cfg.model.predict_type=='angles')) :
            loss['cam_trans'] *= cfg.model.cam_trans_weight
            loss['cam_scale'] *= cfg.model.cam_scale_weight
        
    elif cfg.loss.name == "simple_hand":
        
        def l1_loss_valid(pred, gt, weight=None, valid=None):
            """l1 loss

            Args:
                pred : [B, J, C]
                gt :  [B, J, C]
                weight (optional): [B, J]. Defaults to None.
                valid (optional):  [B]. Defaults to None.
            Returns:
                l1 loss: float
            """
            res = (torch.abs(pred - gt)).sum(dim=2)
            if weight is not None:
                res = res * weight
            res = res.mean(-1)
            if valid is not None:
                res = res * valid
            # return loss.mean()
            # print(loss[0], loss[1])
            return res
        
        uv_pred = output['uv']
        # root_depth_pred = output['root_depth']
        joints_pred = output["pred_joints3d_cam"]
        vertices_pred = output['pred_verts3d_cam']

        uv_pred = uv_pred.reshape(-1, 21, 2).contiguous()
        joints_pred = joints_pred.reshape(-1, 21, 3).contiguous()
        # root_depth_pred = root_depth_pred.reshape(-1, 1).contiguous()

        uv_gt = input['uv']
        # joints_gt = input['xyz']
        # joints_gt = input['xyz'] - input['xyz'][:, :1]
        joints_gt = output['gt_joints3d_cam'] - input['root_joint_flip'].unsqueeze(1)
        # root_depth_gt = input['gamma'].reshape(-1, 1).contiguous()
        hand_uv_valid = input['uv_valid']
        hand_xyz_valid = input['xyz_valid'] # N, 1
        # vertices_gt = output['gt_verts3d_cam']
        # vertices_gt = output['gt_verts3d_cam'] - input['xyz'][:, :1]
        vertices_gt = output['gt_verts3d_cam'] - input['root_joint_flip'].unsqueeze(1)
        
        uv_loss = l1_loss_valid(uv_pred, uv_gt, hand_uv_valid)
        joints_loss = l1_loss_valid(joints_pred, joints_gt, valid=hand_xyz_valid)
        vertices_loss = l1_loss_valid(vertices_pred, vertices_gt, valid=hand_xyz_valid)

        # root_depth_loss = (torch.abs(root_depth_pred- root_depth_gt)).mean()
        # root_depth_loss = root_depth_loss.mean()
        
        loss["uv_loss"] = uv_loss * 1.0
        loss["joints_loss"] = joints_loss * 10.0
        # "root_depth_loss": root_depth_loss * 1.0,
        loss["vertices_loss"] = vertices_loss * 10.0
    
    else:
        raise NotImplementedError
    
    # Common operation
    for k in loss:
        loss[k] = loss[k].mean()  # get mean for each loss
    
    loss["total"] = sum(loss[k] for k in loss)
    
    if cfg.base.verbose:
        print(loss)

    # nan check
    if torch.isnan(loss["total"]):
        print(loss)
        raise NotImplementedError

    return loss


def compute_metric(cfg, input, output):
    metric = {}
    # only compute these loss in metric when gt hand is available
    if "hand_occ_net" in cfg.loss.name and "gt_verts3d_cam" in output:
        metric["mano_verts"] = cfg.loss.lambda_mano_verts * F.mse_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        metric["mano_joints"] = cfg.loss.lambda_mano_joints * F.mse_loss(output["pred_joints3d_cam"], output["gt_joints3d_cam"])
        metric["mano_pose"] = cfg.loss.lambda_mano_pose * F.mse_loss(output["pred_mano_pose"], output["gt_mano_pose"])
        metric["mano_shape"] = cfg.loss.lambda_mano_shape * F.mse_loss(output["pred_mano_shape"], output["gt_mano_shape"])
        metric["joints_img"] = cfg.loss.lambda_joints_img * F.mse_loss(output["pred_joints_img"], output["joints_img"])
        metric["score"] = metric["mano_verts"] + metric["mano_joints"] + metric["mano_pose"] + metric["mano_shape"] + metric["joints_img"]

    elif cfg.loss.name == "semi_hand" and "gt_verts3d_cam" in output:
        metric["mano_verts"] = 1e4 * F.mse_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        metric["mano_joints"] = 1e4 * F.mse_loss(output["pred_joints3d_cam"], output["gt_joints3d_cam"])
        metric["mano_pose"] = 10 * F.mse_loss(output["pred_mano_pose"], output["gt_mano_pose"])
        metric["mano_shape"] = 0.1 * F.mse_loss(output["pred_mano_shape"], output["gt_mano_shape"])
        metric["joints_img"] = 1e2 * F.mse_loss(output["pred_joints_img"], output["joints_img"])
        metric["shape_regul"] = 1e2 * F.mse_loss(output["pred_mano_shape"], torch.zeros_like(output["pred_mano_shape"]))
        metric["pose_regul"] = 1 * F.mse_loss(output["pred_mano_pose"][:, 3:], torch.zeros_like(output["pred_mano_pose"][:, 3:]))
        metric["score"] = metric["mano_verts"] + metric["mano_joints"] + metric["mano_pose"] + metric["mano_shape"] + metric["joints_img"]

    elif cfg.loss.name == "mobrecon" and "gt_verts3d_cam" in output:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        metric["verts_loss"] = F.l1_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        metric["joint_img_loss"] = F.l1_loss(output["pred_joints_img"], output["joints_img"])
        metric["normal_loss"] = 0.1 * normal_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        metric["edge_loss"] = edge_loss(output["pred_verts3d_cam"], output["gt_verts3d_cam"])
        metric["score"] = metric["verts_loss"] + metric["joint_img_loss"] + metric["normal_loss"] + metric["edge_loss"]

    elif "h2onet" in cfg.loss.name and "gt_verts3d_w_gr" in output:
        normal_loss = NormalVectorLoss(mano.face)
        edge_loss = EdgeLengthLoss(mano.face)
        metric["verts_wo_gr"] = F.l1_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["verts_w_gr"] = F.l1_loss(output["pred_verts3d_w_gr"], output["gt_verts3d_w_gr"])
        metric["joints_wo_gr"] = F.l1_loss(output["pred_joints3d_wo_gr"], output["gt_joints3d_wo_gr"])
        metric["joints_w_gr"] = F.l1_loss(output["pred_joints3d_w_gr"], output["gt_joints3d_w_gr"])
        metric["glob_rot_loss"] = F.mse_loss(output["pred_glob_rot_mat"], output["gt_glob_rot_mat"])
        metric["joint_img"] = F.l1_loss(output["pred_joints_img"], output["joints_img"])
        metric["normal"] = 0.1 * normal_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["edge"] = edge_loss(output["pred_verts3d_wo_gr"], output["gt_verts3d_wo_gr"])
        metric["score"] = metric["verts_w_gr"] + metric["verts_wo_gr"]

    elif cfg.loss.name == "grasp_vit_classifier":
        gt_cls = input["gt_grasping"].squeeze(-1).to(torch.int64)
        metric['obj_grasp_acc'] = torch.sum(torch.argmax(output["pred_obj_conf"], dim=-1) == gt_cls) / gt_cls.shape[0]
        metric['score'] = 1 - metric['obj_grasp_acc']

    elif "hflnet_objconf" in cfg.model.name:
        # gt_cls = input["gt_grasping"].squeeze(-1) #.to(torch.int64)
        test_part = cfg.test.get("test_part", 0)
        gt_cls = input["gt_grasping"][:, test_part].to(torch.int64)
        
        # metric for object grasping classification
        metric['obj_grasp_acc'] = torch.sum(torch.argmax(output["pred_obj_conf"], dim=-1) == gt_cls) / gt_cls.shape[0]
        
        # metric for hand occlusion classification
        if "pred_hand_occ" in output:
            gt_hand_occ_cls = input["gt_occ_cls"][:, test_part].to(torch.int64)
            metric['hand_occ_cls_acc'] = torch.sum(torch.argmax(output["pred_hand_occ"], dim=-1) == gt_hand_occ_cls) / gt_hand_occ_cls.shape[0]
        
        # metric for 2d joint
        preds_joints2d = output['preds_joints2d']
        if 'joints_img' in output:
            joints_uv = output['joints_img']  # torch.Size([B, 21, 2])
        elif 'joints_img' in input:
            joints_uv = input['joints_img'][:, :21]  # torch.Size([B, 21, 2])
        else:
            joints_uv = None
        
        if joints_uv is not None:
            joint2d_loss_func = Joint2DLoss(lambda_joints2d=1.0)
            joint2d_loss, joint2d_losses = joint2d_loss_func(preds_joints2d, joints_uv)
            metric['joint2d_loss'] = joint2d_loss
            
    if "gt_verts3d_cam" in output:  # only cal the metric when the ground truth is available
        # pred
        verts_out = copy.deepcopy(output["pred_verts3d_cam"])
        joints_out = copy.deepcopy(output["pred_joints3d_cam"])
        # root centered. Here we assume the root is zero indexed
        verts_out -= joints_out[:, 0, None]
        joints_out -= joints_out[:, 0, None]
        
        # gt
        verts_gt = copy.deepcopy(output["gt_verts3d_cam"])
        joints_gt = copy.deepcopy(output["gt_joints3d_cam"])  # NOTE: obtained from mano coefficients
        # joints_gt = input["joints_coord_cam"]
        
        # root centered
        verts_gt -= joints_gt[:, 0, None]
        joints_gt -= joints_gt[:, 0, None]

        # root align, won't affect point
        gt_root_joint_cam = input["root_joint_flip"].unsqueeze(1)
        verts_out += gt_root_joint_cam
        joints_out += gt_root_joint_cam
        verts_gt += gt_root_joint_cam
        joints_gt += gt_root_joint_cam
        
        # align predictions
        joints_out_aligned = torch_align_w_scale(joints_gt, joints_out)
        verts_out_aligned = torch_align_w_scale(verts_gt, verts_out)

        # m to mm
        joints_out *= 1000
        joints_out_aligned *= 1000
        joints_gt *= 1000
        verts_out *= 1000
        verts_out_aligned *= 1000
        verts_gt *= 1000
        metric['MPJPE'] = torch.sqrt(torch.sum((joints_out - joints_gt)**2, 2)).mean()
        metric['PA-MPJPE'] = torch.sqrt(torch.sum((joints_out_aligned - joints_gt)**2, 2)).mean()
        metric['MPVPE'] = torch.sqrt(torch.sum((verts_out - verts_gt)**2, 2)).mean()
        metric['PA-MPVPE'] = torch.sqrt(torch.sum((verts_out_aligned - verts_gt)**2, 2)).mean()
    
        if 'w_gr' not in cfg.train or cfg.train['w_gr'] is True:  # with global rotation
            metric['score'] = metric['MPJPE'] + metric['PA-MPJPE'] + metric['MPVPE'] + metric['PA-MPVPE']
        else:  # 
            metric['score'] = metric['PA-MPJPE'] + metric['PA-MPVPE']
    
    return metric
