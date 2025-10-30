import torch
import torch.nn as nn
from torch.nn import functional as F
from common.utils.mano import MANO
from model.hand_occ_net.hand_head import hand_regHead, hand_Encoder, hand_Encoder_2
from model.hand_occ_net.mano_head import mano_regHead, mano_regShapeHead, mano_regPoseHead, mano_regRootPoseHead, mano_regFingerPoseHead, batch_rodrigues, rot6d2mat, mat2aa

mano = MANO()

class Regressor(nn.Module):

    def __init__(self):
        super(Regressor, self).__init__()
        self.hand_regHead = hand_regHead()
        self.hand_Encoder = hand_Encoder()
        self.mano_regHead = mano_regHead()

    def forward(self, feats):
        out_hm, encoding, preds_joints_img = self.hand_regHead(feats)  # torch.Size([16, 21, 32, 32]), torch.Size([16, 256, 32, 32]), torch.Size([16, 21, 2])
        mano_encoding = self.hand_Encoder(out_hm, encoding)  # torch.Size([16, 1024])
        pred_mano_results = self.mano_regHead(mano_encoding)  # dict_keys(['verts3d', 'joints3d', 'mano_shape', 'mano_pose', 'mano_pose_aa'])

        return pred_mano_results, preds_joints_img


class Regressor_2(nn.Module):

    def __init__(self):
        super(Regressor_2, self).__init__()
        self.hand_regHead = hand_regHead()

        # share the same encoder
        self.hand_Encoder = hand_Encoder()

        # decouple regressor for shape, finger pose and root pose
        self.mano_regShapeHead = mano_regShapeHead()
        self.mano_regRootPoseHead = mano_regRootPoseHead()
        self.mano_regFingerPoseHead = mano_regFingerPoseHead()

        self.mano_layer = mano.layer

    def forward(self, feats):
        out_hm, encoding, preds_joints_img = self.hand_regHead(feats)  # torch.Size([16, 21, 32, 32]), torch.Size([16, 256, 32, 32]), torch.Size([16, 21, 2])
        mano_encoding = self.hand_Encoder(out_hm, encoding)  # torch.Size([16, 1024])

        pred_mano_shape, kd_dict_shape = self.mano_regShapeHead(mano_encoding)
        pred_mano_shape = pred_mano_shape['mano_shape']
        root_pose_6d, kd_dict_root = self.mano_regRootPoseHead(mano_encoding)
        finger_pose_6d, kd_dict_finger = self.mano_regFingerPoseHead(mano_encoding)
        # combine root pose and finger pose
        pose_6d = torch.cat((root_pose_6d, finger_pose_6d), dim=1)  # torch.Size([B, 96])
        pred_mano_pose = rot6d2mat(pose_6d.view(-1, 6)).view(-1, 16, 3, 3).contiguous()  # torch.Size([B, 16, 3, 3])
        pred_mano_pose_aa = mat2aa(pred_mano_pose.view(-1, 3, 3)).contiguous().view(-1, 16 * 3)  # torch.Size([B, 48])

        # predict vertex and joint coordinates
        pred_verts, pred_joints = self.mano_layer(th_pose_coeffs=pred_mano_pose_aa, th_betas=pred_mano_shape)
        pred_verts /= 1000
        pred_joints /= 1000

        pred_mano_results = {
            'verts3d': pred_verts,
            'joints3d': pred_joints,
            'mano_shape': pred_mano_shape,
            'mano_pose': pred_mano_pose,
            'mano_pose_aa': pred_mano_pose_aa}

        # features to be distillated
        kd_dict = {}
        kd_dict.update(kd_dict_shape)
        kd_dict.update(kd_dict_root)
        kd_dict.update(kd_dict_finger)
        return pred_mano_results, preds_joints_img, kd_dict


class Regressor_MV_1(nn.Module):

    def __init__(self):
        super(Regressor_MV_1, self).__init__()
        self.hand_regHead = hand_regHead(channels=256*8)
        self.hand_Encoder = hand_Encoder(num_feat_chan=256*8)
        self.mano_regHead = mano_regHead(feature_size=1024*8)

    def forward(self, feats):
        out_hm, encoding, preds_joints_img = self.hand_regHead(feats)  # torch.Size([16, 21, 32, 32]), torch.Size([16, 256, 32, 32]), torch.Size([16, 21, 2])
        mano_encoding = self.hand_Encoder(out_hm, encoding)  # torch.Size([16, 1024])
        pred_mano_results = self.mano_regHead(mano_encoding)  # dict_keys(['verts3d', 'joints3d', 'mano_shape', 'mano_pose', 'mano_pose_aa'])

        return pred_mano_results, preds_joints_img


class Regressor_MV_2(nn.Module):

    def __init__(self):
        super(Regressor_MV_2, self).__init__()
        self.hand_Encoder = hand_Encoder()
        self.mano_regHead = mano_regHead()

    def forward(self, out_hm, encoding):
        mano_encoding = self.hand_Encoder(out_hm, encoding)  # torch.Size([16, 1024])
        pred_mano_results = self.mano_regHead(mano_encoding)  # dict_keys(['verts3d', 'joints3d', 'mano_shape', 'mano_pose', 'mano_pose_aa'])

        return pred_mano_results


class Pose_Regressor_MV(nn.Module):

    def __init__(self):
        super(Pose_Regressor_MV, self).__init__()
        self.hand_Encoder = hand_Encoder()
        self.mano_regHead = mano_regPoseHead()

    def forward(self, out_hm, encoding):
        mano_encoding = self.hand_Encoder(out_hm, encoding)  # torch.Size([16, 1024])
        pred_mano_results, kd_dict = self.mano_regHead(mano_encoding)  # dict_keys(['verts3d', 'joints3d', 'mano_shape', 'mano_pose', 'mano_pose_aa'])

        return pred_mano_results, kd_dict


class Shape_Regressor_MV(nn.Module):

    def __init__(self):
        super(Shape_Regressor_MV, self).__init__()
        self.hand_Encoder = hand_Encoder()
        self.mano_regHead = mano_regShapeHead()

    def forward(self, out_hm, encoding):
        mano_encoding = self.hand_Encoder(out_hm, encoding)  # torch.Size([16, 1024])
        pred_mano_results, kd_dict = self.mano_regHead(mano_encoding)  # dict_keys(['mano_shape'])

        # TODO: features to be distillated

        return pred_mano_results, kd_dict

# fusion feature as input based on Shape_Regressor_MV
class Shape_Regressor_MV_2(nn.Module):

    def __init__(self):
        super(Shape_Regressor_MV_2, self).__init__()
        self.hand_Encoder = hand_Encoder_2()
        self.mano_regHead = mano_regShapeHead()

    def forward(self, x):
        mano_encoding = self.hand_Encoder(x)  # torch.Size([16, 1024])
        pred_mano_results, kd_dict = self.mano_regHead(mano_encoding)  # dict_keys(['mano_shape'])

        # TODO: features to be distillated

        return pred_mano_results, kd_dict


class Pose_Root_Regressor_MV(nn.Module):

    def __init__(self):
        super(Pose_Root_Regressor_MV, self).__init__()
        self.hand_Encoder = hand_Encoder()
        self.mano_regRootHead = mano_regRootPoseHead()

    def forward(self, out_hm, encoding):
        mano_encoding = self.hand_Encoder(out_hm, encoding)  # torch.Size([16, 1024])
        pred_mano_results, kd_dict = self.mano_regRootHead(mano_encoding)  # dict_keys(['mano_root_pose', 'mano_root_pose_aa'])

        return pred_mano_results, kd_dict


# fusion feature as input based on Pose_Root_Regressor_MV
class Pose_Root_Regressor_MV_2(nn.Module):

    def __init__(self):
        super(Pose_Root_Regressor_MV_2, self).__init__()
        self.hand_Encoder = hand_Encoder_2()
        self.mano_regRootHead = mano_regRootPoseHead()

    def forward(self, x):
        mano_encoding = self.hand_Encoder(x)  # torch.Size([16, 1024])
        pred_mano_results, kd_dict = self.mano_regRootHead(mano_encoding)  # dict_keys(['mano_root_pose', 'mano_root_pose_aa'])

        return pred_mano_results, kd_dict

class Pose_Finger_Regressor_MV(nn.Module):

    def __init__(self):
        super(Pose_Finger_Regressor_MV, self).__init__()
        self.hand_Encoder = hand_Encoder()
        self.mano_regFingerHead = mano_regFingerPoseHead()

    def forward(self, out_hm, encoding):
        mano_encoding = self.hand_Encoder(out_hm, encoding)  # torch.Size([16, 1024])
        pred_mano_results, kd_dict = self.mano_regFingerHead(mano_encoding)  # dict_keys(['mano_root_pose', 'mano_root_pose_aa'])

        return pred_mano_results, kd_dict


# fusion feature as input based on Pose_Finger_Regressor_MV
class Pose_Finger_Regressor_MV_2(nn.Module):

    def __init__(self):
        super(Pose_Finger_Regressor_MV_2, self).__init__()
        self.hand_Encoder = hand_Encoder_2()
        self.mano_regFingerHead = mano_regFingerPoseHead()

    def forward(self, x):
        mano_encoding = self.hand_Encoder(x)  # torch.Size([16, 1024])
        pred_mano_results, kd_dict = self.mano_regFingerHead(mano_encoding)  # dict_keys(['mano_root_pose', 'mano_root_pose_aa'])

        return pred_mano_results, kd_dict
