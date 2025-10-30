#from unittest.case import doModuleCleanups
from torch.utils import data
import random
import numpy as np
from numpy.linalg import inv
from PIL import Image, ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from common.utils import ho3d_hor_util, dex_ycb_hor_util, dataset_hor_util
from common.utils.transforms import *
import json
import os
from torchvision.transforms import functional
from common.utils.manopth.manopth.manolayer import ManoLayer
import copy
import os.path as osp
Image.MAX_IMAGE_PIXELS = None
from common.utils.metric import *
from common.utils.vis import *

mano_layer = ManoLayer(flat_hand_mean=False,
                           side="right", mano_root=osp.join("./common/utils/manopth", "mano", "models"), use_pca=False)
mano_layerl = ManoLayer(flat_hand_mean=False,
                           side="left", mano_root=osp.join("./common/utils/manopth", "mano", "models"), use_pca=False)

mano_right_face = mano_layer.th_faces.numpy()

import math

has_nori = True # by default use nori
try:
    import nori2 as nori
except ImportError:
    print("nori2 is not installed in this machine. Use local version instead.")
    has_nori = False
import pickle

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = '0.0'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.1'

# common dataloader for UniHOPENet, HFL-Net, MobRecon, HandOccNet .. and other hand reconstruction methods
class DEX_YCB_HOPE(data.Dataset):
    def __init__(self, cfg, transform, mode,
                dataset_root = "./data/DexYCB", 
                max_rot=np.pi, scale_jittering=0.2, center_jittering=0.1,
                hue=0.15, saturation=0.5, contrast=0.5, brightness=0.5, blur_radius=0.5) -> None:
        self.cfg = cfg
        
        self.transform = transform
        self.dataset_root = dataset_root
        
        self.data_split = cfg.data.get("data_split", "s0")
        
        self.use_data_aug_h = cfg.data.get("use_data_aug_h", True)  # data augmentation for the original hand
        self.use_data_aug_ho = cfg.data.get("use_data_aug_ho", True)  # data augmentation for the original hand-object
        self.use_data_aug_h_gen = cfg.data.get("use_data_aug_h_gen", False)  # data augmentation for the generated hand
        self.pair_same_data_aug = cfg.data.get("pair_same_data_aug", False)  # by default we use different data augmentation for one pair of hand images
        
        self.filter_rule = cfg.data.get("filter_rule", "score")  # [score, occ, and, or, all-gen]
        self.score_threshold = cfg.data.get("score_threshold", 0)
        self.occ_threshold = cfg.data.get("occ_threshold", 0.8)
        
        self.mode = mode
        self.inp_res = cfg.data.input_img_shape
        self.joint_root_id = 0
        
        #object information
        self.obj_mesh = dex_ycb_hor_util.load_objects_dex_ycb(self.dataset_root)
        self.obj_bbox3d = dataset_hor_util.get_bbox21_3d_from_dict(self.obj_mesh)
        self.obj_diameters = dataset_hor_util.get_diameter(self.obj_mesh)
        
        self.load_nori = cfg.data.get("load_nori", False) and has_nori  # by default use the local version
        print("Load from Nori: ", self.load_nori)
        
        self.contrastive = self.cfg.data.get("contrastive", False)
        
        self.bbox_hand_normalize_2d_joint = self.cfg.data.get("bbox_hand_normalize_2d_joint", True)  # ho, h

        if self.load_nori:
            self.nori_root = os.path.join(self.dataset_root, "nori")
            self.fetcher = nori.Fetcher()
        
        if self.mode == "train":
            self.hue = hue
            self.contrast = contrast
            self.brightness = brightness
            self.saturation = saturation
            self.blur_radius = blur_radius
            self.scale_jittering = scale_jittering
            self.center_jittering = center_jittering
            self.max_rot = max_rot
            
            if self.load_nori: # load from nori
                nori_path = os.path.join(self.nori_root,f"correct_{self.data_split}_train.npy")
                with open(nori_path, 'rb') as f:
                    self.nori_dict = np.load(f, allow_pickle=True).item()
                print(nori_path)
                self.sample_list_processed = list(self.nori_dict.keys())
                
            else:  # load from local
                data_path = os.path.join(self.dataset_root, f"dex_ycb_{self.data_split}_train_data.json")
                
                with open(data_path, 'r', encoding='utf-8') as f:
                    self.sample_dict = json.load(f)
                self.sample_list = sorted(self.sample_dict.keys(),key=lambda x:int(x[3:]))
                self.sample_list_processed = []
                #筛选bbox expansion_factor=1.5后仍然在图片的样本.与22年的数据处理对齐
                for sample in self.sample_list:
                    joint_2d = np.array(self.sample_dict[sample]["joint_2d"],dtype=np.float32).squeeze()
                    hand_bbox = dex_ycb_hor_util.get_bbox(joint_2d, np.ones_like(joint_2d[:,0]), expansion_factor=1.5)
                    hand_bbox = dex_ycb_hor_util.process_bbox(hand_bbox,640,480,expansion_factor=1.0)
                    if hand_bbox is None:
                        continue
                    else:
                        self.sample_list_processed.append(sample)
            
            if self.contrastive: # Load generated images
                if self.load_nori:
                    gen_version = cfg.data.get("gen_version", 1)
                    gen_part = cfg.data.get("gen_part", 1)
                    
                    # HandRefienr model
                    if gen_part == 1:
                        # generated with adaptive control strengths
                        gen_nori_path = os.path.join(self.nori_root, f"correct_{self.data_split}_train_name_handrefiner_inpaint_condition_v{gen_version}.npy")
                    else:
                        # fixed control strength
                        gen_nori_path = os.path.join(self.nori_root, f"correct_{self.data_split}_train_name_handrefiner_inpaint_condition_v{gen_version}_p{gen_part}.npy")
                    
                    with open(gen_nori_path, 'rb') as f:
                        self.gen_nori_dict = np.load(f, allow_pickle=True).item()
                    print(gen_nori_path)
                else:
                    # TODO: load from local, which is not implemented yet
                    raise NotImplementedError
            
        else:  # test mode
            if self.load_nori: # load from nori
                nori_path = os.path.join(self.nori_root,f"correct_{self.data_split}_test.npy")
                print(nori_path)
                with open(nori_path, 'rb') as f:
                    self.nori_dict = np.load(f, allow_pickle=True).item()
                # self.sample_list_processed = list(self.nori_dict.values())
                self.sample_list_processed = list(self.nori_dict.keys())
                
            else: # load from local
                self.hand_bbox_list = []
                data_path = os.path.join(self.dataset_root, f"dex_ycb_{self.data_split}_test_data.json")
                    
                with open(data_path, 'r', encoding='utf-8') as f:
                    self.sample_dict = json.load(f)
                self.sample_list = sorted(self.sample_dict.keys(),key=lambda x:int(x[3:]))
                #筛选bbox expansion_factor=1.5后仍然在图片的样本.与22年的数据处理对齐
                idx = 0
                for sample in self.sample_list:
                    idx = idx + 1
                    joint_2d = np.array(self.sample_dict[sample]["joint_2d"],dtype=np.float32).squeeze()
                    hand_bbox = dex_ycb_hor_util.get_bbox(joint_2d, np.ones_like(joint_2d[:,0]), expansion_factor=1.5)
                    hand_bbox = dex_ycb_hor_util.process_bbox(hand_bbox,640,480,expansion_factor=1.0)
                    if hand_bbox is None:
                        hand_bbox = np.array([0,0,640-1, 480-1], dtype=np.float32)
                    self.hand_bbox_list.append(hand_bbox)
                self.sample_list_processed = self.sample_list 
                    
        # load occ ratio for both train and test, if the file exists
        occ_path = os.path.join(self.dataset_root, f'hand_occ_{self.data_split}_{self.mode}.npy')
        if os.path.exists(occ_path):
            self.occ_ratio_dict = np.load(occ_path, allow_pickle=True).item()
            print(occ_path)
        else:
            self.occ_ratio_dict = {k: 1.0 for k in self.sample_list_processed}  # all samples are not occluded
            print("Create fake occlusion ratio.")
        
        
        input_mode = cfg.data.get("input_mode", None)
        if input_mode == "ho":
            # all regarded as grasping, crop hand and object together
            self.grasp_gt_dict = {k: True for k in self.sample_list_processed}
        elif input_mode == "h":
            # all regarded as non-grasping, crop hand only
            self.grasp_gt_dict = {k: False for k in self.sample_list_processed}
        elif input_mode is None:
            # load grasping dict if use for both train and test
            grasp_gt_path = os.path.join(self.dataset_root, f"grasp_{self.data_split}_{self.mode}.npy")
                
            self.grasp_gt_dict = np.load(grasp_gt_path, allow_pickle=True).item()
            print(grasp_gt_path)
        
        print("Dataset len: ", len(self.sample_list_processed))
        
        # prepare for object evaluation
        mesh_dict, diameter_dict = dex_ycb_hor_util.filter_test_object_dexycb(self.obj_mesh, self.obj_diameters)
        self.mesh_dict = mesh_dict
        self.diameter_dict = diameter_dict
        
        self.unseen_objects = dataset_hor_util.get_unseen_test_object()
        
        self.obj_eval_mode = cfg.metric.get("obj_eval_mode", "gt")  # [gt, pred-gt, all]
        
        self.REP_res_dict = {}
        self.ADD_res_dict= {}
        for k in mesh_dict.keys():
            self.REP_res_dict[k] = []
            self.ADD_res_dict[k] = []
            
        self.rep_threshold = cfg.metric.get("rep_threshold", 5)
        self.add_threshold = cfg.metric.get("add_threshold", 0.1)
    
    def random_center_jittering(self, scale):
        # Randomly jitter center
        center_offsets = (self.center_jittering * scale * np.random.uniform(low=-1, high=1, size=2))
        return center_offsets

    def random_scale_jittering(self):
        # Scale jittering
        scale_jittering = self.scale_jittering * np.random.randn() + 1
        scale_jittering = np.clip(scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
        return scale_jittering
    
    def random_rot(self):
        rot_factor = 30
        rot = np.clip(np.random.randn(), -2.0,
                2.0) * rot_factor if random.random() <= 0.6 else 0
        rot = rot*self.max_rot/180
        return rot
    
    # Img blurring and color jitter
    def random_blur(self):
        blur_radius = random.random() * self.blur_radius
        gaussian_blur = ImageFilter.GaussianBlur(blur_radius)
        return gaussian_blur
                
    def data_aug(self, img, mano_param, joints_uv, K, gray, p2d, joints_3d, obj_pose, root_joint, root_joint_flip, grasping, K_vis):
        if grasping:
            return self.data_aug_fuse(img, mano_param, joints_uv, K, gray, p2d, joints_3d, obj_pose, root_joint, root_joint_flip, K_vis)
        else:
            return self.data_aug_hand(img, mano_param, joints_uv, K, gray, p2d, joints_3d, obj_pose, root_joint, root_joint_flip, K_vis)
    
    # data augmentation of hand input
    def data_aug_hand(self, img, mano_param, joints_uv, K, gray, p2d, joints_3d, obj_pose, root_joint, root_joint_flip, K_vis):
        crop_hand = dataset_hor_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        center, scale = dataset_hor_util.bbox_center_scale(crop_hand, img.size)

        rot = 0
        if self.use_data_aug_h:
            # Randomly jitter center
            center_offsets = (self.center_jittering * scale * np.random.uniform(low=-1, high=1, size=2))
            center = center + center_offsets

            # Scale jittering
            scale_jittering = self.scale_jittering * np.random.randn() + 1
            scale_jittering = np.clip(scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
            scale = scale * scale_jittering

            rot_factor = 30
            rot = np.clip(np.random.randn(), -2.0,
                    2.0) * rot_factor if random.random() <= 0.6 else 0
            rot = rot*self.max_rot/180
        
        
        affinetrans, post_rot_trans, rot_mat = dataset_hor_util.get_affine_transform(center, scale,
                                                                              self.inp_res, rot=rot,
                                                                              K=K)
        # 注意！
        mano_param[:3] = dataset_hor_util.rotation_angle(mano_param[:3], rot_mat, coord_change_mat=np.eye(3))
        
        joints_3d = joints_3d.dot(rot_mat.T)
        
        root_joint = root_joint.dot(rot_mat.T)
        
        root_joint_flip = root_joint_flip.dot(rot_mat.T)
        
        obj_pose = ho3d_hor_util.pose_from_rotmat(rot_mat).dot(obj_pose)
        

        joints_uv = dataset_hor_util.transform_coords(joints_uv, affinetrans)  # hand landmark trans
        K = post_rot_trans.dot(K)
        
        K_vis = post_rot_trans.dot(K_vis)

        p2d = dataset_hor_util.transform_coords(p2d, affinetrans)  # obj landmark trans
        # get hand bbox and normalize landmarks to [0,1]
        bbox_hand = dataset_hor_util.get_bbox_joints(joints_uv, bbox_factor=1.1)
        joints_uv = dataset_hor_util.normalize_joints(joints_uv, bbox_hand)

        # get obj bbox and normalize landmarks to [0,1]
        bbox_obj = dataset_hor_util.get_bbox_joints(p2d, bbox_factor=1.0)
        p2d = dataset_hor_util.normalize_joints(p2d, bbox_obj)

        # Transform and crop
        img = dataset_hor_util.transform_img(img, affinetrans, self.inp_res)
        img = img.crop((0, 0, self.inp_res[0], self.inp_res[1]))

        if self.use_data_aug_h:
            # Img blurring and color jitter
            blur_radius = random.random() * self.blur_radius
            img = img.filter(ImageFilter.GaussianBlur(blur_radius))
            img = dataset_hor_util.color_jitter(img, brightness=self.brightness,
                                            saturation=self.saturation, hue=self.hue, contrast=self.contrast)

        # Generate object mask: gray segLabel transform and crop
        gray = dataset_hor_util.transform_img(gray, affinetrans, self.inp_res)
        gray = gray.crop((0, 0, self.inp_res[0], self.inp_res[1]))
        
        gray_copy = functional.to_tensor(copy.deepcopy(gray))
        
        gray = dataset_hor_util.get_mask_ROI(gray, bbox_obj)
        # Generate object mask
        gray = np.asarray(gray.resize((32, 32), Image.NEAREST))
        obj_mask = np.ma.getmaskarray(np.ma.masked_not_equal(gray, 0)).astype(int)
        obj_mask = torch.from_numpy(obj_mask)

        return functional.to_tensor(img), mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj, joints_3d, obj_pose, root_joint, root_joint_flip, gray_copy, K_vis, rot_mat
    
    # data augmentation of hand-object fusion input
    def data_aug_fuse(self, img, mano_param, joints_uv, K, gray, p2d, joints_3d, obj_pose, root_joint, root_joint_flip, K_vis):
        crop_hand = dataset_hor_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        crop_obj = dataset_hor_util.get_bbox_joints(p2d, bbox_factor=1.5)
        center, scale = dataset_hor_util.fuse_bbox(crop_hand, crop_obj, img.size)
        
        
        rot = 0
        if self.use_data_aug_ho:
            # Randomly jitter center
            center_offsets = (self.center_jittering * scale * np.random.uniform(low=-1, high=1, size=2))
            center = center + center_offsets

            # Scale jittering
            scale_jittering = self.scale_jittering * np.random.randn() + 1
            scale_jittering = np.clip(scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
            scale = scale * scale_jittering

            rot_factor = 30
            rot = np.clip(np.random.randn(), -2.0,
                    2.0) * rot_factor if random.random() <= 0.6 else 0
            rot = rot*self.max_rot/180
        
        
        affinetrans, post_rot_trans, rot_mat = dataset_hor_util.get_affine_transform(center, scale,
                                                                              self.inp_res, rot=rot,
                                                                              K=K)
        # 注意！
        mano_param[:3] = dataset_hor_util.rotation_angle(mano_param[:3], rot_mat, coord_change_mat=np.eye(3))
        
        joints_3d = joints_3d.dot(rot_mat.T)
        
        root_joint = root_joint.dot(rot_mat.T)
        
        root_joint_flip = root_joint_flip.dot(rot_mat.T)
        
        obj_pose = ho3d_hor_util.pose_from_rotmat(rot_mat).dot(obj_pose)
        

        joints_uv = dataset_hor_util.transform_coords(joints_uv, affinetrans)  # hand landmark trans
        K = post_rot_trans.dot(K)
        
        K_vis = post_rot_trans.dot(K_vis)

        p2d = dataset_hor_util.transform_coords(p2d, affinetrans)  # obj landmark trans
        # get hand bbox and normalize landmarks to [0,1]
        bbox_hand = dataset_hor_util.get_bbox_joints(joints_uv, bbox_factor=1.1)
        joints_uv = dataset_hor_util.normalize_joints(joints_uv, bbox_hand)

        # get obj bbox and normalize landmarks to [0,1]
        bbox_obj = dataset_hor_util.get_bbox_joints(p2d, bbox_factor=1.0)
        p2d = dataset_hor_util.normalize_joints(p2d, bbox_obj)

        # Transform and crop
        img = dataset_hor_util.transform_img(img, affinetrans, self.inp_res)
        img = img.crop((0, 0, self.inp_res[0], self.inp_res[1]))

        # Img blurring and color jitter
        if self.use_data_aug_ho:
            blur_radius = random.random() * self.blur_radius
            img = img.filter(ImageFilter.GaussianBlur(blur_radius))
            img = dataset_hor_util.color_jitter(img, brightness=self.brightness,
                                            saturation=self.saturation, hue=self.hue, contrast=self.contrast)

        # Generate object mask: gray segLabel transform and crop
        gray = dataset_hor_util.transform_img(gray, affinetrans, self.inp_res)
        gray = gray.crop((0, 0, self.inp_res[0], self.inp_res[1]))
        
        gray_copy = functional.to_tensor(copy.deepcopy(gray))
        
        gray = dataset_hor_util.get_mask_ROI(gray, bbox_obj)
        # Generate object mask
        gray = np.asarray(gray.resize((32, 32), Image.NEAREST))
        obj_mask = np.ma.getmaskarray(np.ma.masked_not_equal(gray, 0)).astype(int)
        obj_mask = torch.from_numpy(obj_mask)

        return functional.to_tensor(img), mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj, joints_3d, obj_pose, root_joint, root_joint_flip, gray_copy, K_vis, rot_mat
    
    # data augmentation of hand input
    def data_aug_ho_pair(self, img, mano_param, joints_uv, K, gray, p2d, joints_3d, obj_pose, root_joint, root_joint_flip,
                           img_aug, K_vis, gt_occ_cls):  # img_aug is at a different resolution
        # # for original hand
        # crop_hand = dataset_hor_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        # center_hand, scale_hand = dataset_hor_util.bbox_center_scale(crop_hand, img.size)
        
        # for original hand-object. This is the same for gen_img as we processed in HandObjectImageSynthesizer/dataset/DexYCB_GR.py
        crop_hand = dataset_hor_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        crop_obj = dataset_hor_util.get_bbox_joints(p2d, bbox_factor=1.5)
        center, scale = dataset_hor_util.fuse_bbox(crop_hand, crop_obj, img.size)
        
        # input
        in_imgs = [img, img_aug]
        
        # augmentable paramameters
        # NOTE: the inputs are different. The gen_img is a cropped version, so the center and scale are different.
        in_scales = [scale, scale]
        in_centers = [center, center]
        in_rots = [0, 0]
        in_filters = [None, None]
        in_transforms = [None, None]
        
        # if use data augmentation
        if self.use_data_aug_ho and self.use_data_aug_h_gen and self.pair_same_data_aug:
            # same set of augmentation parameters for all cases, the image region and gt are the same, except for their blur and color
            # raise NotImplementedError
            center_jittering_factor = np.random.uniform(low=-1, high=1, size=2)
            scale_jittering = self.random_scale_jittering()
            rot = self.random_rot()
            blur_filter = self.random_blur()
            color_transform = dataset_hor_util.color_jitter_transforms(brightness=self.brightness,
                                            saturation=self.saturation, hue=self.hue, contrast=self.contrast)
            
            # same center and scale for both images
            in_centers = [in_centers[i] + (self.center_jittering * in_scales[i] * center_jittering_factor) for i in range(2)]
            in_scales = [in_scales[i] *  scale_jittering for i in range(2)]
            # same rotation for both images
            in_rots = [rot for i in range(2)]
            # same blur and color jittering for both images
            in_filters = [blur_filter for i in range(2)]
            in_transforms = [copy.deepcopy(color_transform) for i in range(2)]
        elif self.use_data_aug_ho and self.use_data_aug_h_gen:
            # different augmentation parameters for each case, the image region and gt are different, which might be more difficult for network to learn
            # raise NotImplementedError
            # center, scale and rot augmentation only for the original image
            in_centers = [in_centers[i] + self.random_center_jittering(in_scales[i]) for i in range(2)]
            in_scales = [in_scales[i] * self.random_scale_jittering() for i in range(2)]
            in_rots = [self.random_rot() for i in range(2)]
            # different blur and color jittering for both images
            in_filters = [self.random_blur() for i in range(2)]
            in_transforms = [dataset_hor_util.color_jitter_transforms(brightness=self.brightness,
                                            saturation=self.saturation, hue=self.hue, contrast=self.contrast) for i in range(2)]
        elif self.use_data_aug_ho:
            # Data augmentation for the original image, the image region and gt are different, which might be more difficult for network to learn
            in_centers[0] = in_centers[0] + self.random_center_jittering(in_scales[0])
            in_scales[0] = in_scales[0] * self.random_scale_jittering()
            in_rots[0] = self.random_rot()
            in_filters[0] = self.random_blur()
            in_transforms[0] = dataset_hor_util.color_jitter_transforms(brightness=self.brightness,
                                            saturation=self.saturation, hue=self.hue, contrast=self.contrast)
        elif self.use_data_aug_h_gen:
            # only aug for generated data, no aug for original data
            # raise NotImplementedError
            in_centers[1] = in_centers[0] + self.random_center_jittering(in_scales[0])
            in_scales[1] = in_scales[0] * self.random_scale_jittering()
            in_rots[1] = self.random_rot()
            in_filters[1] = self.random_blur()
            in_transforms[1] = dataset_hor_util.color_jitter_transforms(brightness=self.brightness,
                                            saturation=self.saturation, hue=self.hue, contrast=self.contrast)
        
        # otherwise both wo/ data augmentation
        imgs = []
        mano_params = []
        Ks = []
        obj_masks = []
        p2ds = []
        joints_uvs = []
        bbox_hands = []
        bbox_objs = []
        joints_3ds = []
        obj_poses = []
        root_joints = []
        root_joint_flips = []
        gray_copys = []
        K_vis_s = []
        rot_mat_s = []

        for i in range(len(in_imgs)):
            img_i = in_imgs[i]
                
            affinetrans, post_rot_trans, rot_mat = dataset_hor_util.get_affine_transform(in_centers[i], in_scales[i],
                                                                                        self.inp_res, rot=in_rots[i],
                                                                                        K=K)
            # 注意！
            mano_param_i = copy.deepcopy(mano_param)
            mano_param_i[:3] = dataset_hor_util.rotation_angle(mano_param[:3], rot_mat, coord_change_mat=np.eye(3))
            
            joints_3d_i = joints_3d.dot(rot_mat.T)
            
            root_joint_i = root_joint.dot(rot_mat.T)
            
            root_joint_flip_i = root_joint_flip.dot(rot_mat.T)
            
            obj_pose_i = ho3d_hor_util.pose_from_rotmat(rot_mat).dot(obj_pose)

            joints_uv_i = dataset_hor_util.transform_coords(joints_uv, affinetrans)  # hand landmark trans
            K_i = post_rot_trans.dot(K)
            
            K_vis_i = post_rot_trans.dot(K_vis)

            p2d_i = dataset_hor_util.transform_coords(p2d, affinetrans)  # obj landmark trans
            # get hand bbox and normalize landmarks to [0,1]
            bbox_hand_i = dataset_hor_util.get_bbox_joints(joints_uv_i, bbox_factor=1.1)
            joints_uv_i = dataset_hor_util.normalize_joints(joints_uv_i, bbox_hand_i)

            # get obj bbox and normalize landmarks to [0,1]
            bbox_obj_i = dataset_hor_util.get_bbox_joints(p2d_i, bbox_factor=1.0)
            p2d_i = dataset_hor_util.normalize_joints(p2d_i, bbox_obj_i)

            # Transform and crop the original image
            # the version for the generated image is also impelemented
            img_i = dataset_hor_util.transform_img(img_i, affinetrans, self.inp_res)
            img_i = img_i.crop((0, 0, self.inp_res[0], self.inp_res[1]))

            # only use available filters and transforms
            if in_filters[i] is not None:
                img_i = img_i.filter(in_filters[i])
            if in_transforms[i] is not None:
                img_i = dataset_hor_util.color_jitter_apply(img_i, in_transforms[i])

            # Generate object mask: gray segLabel transform and crop
            gray_i = dataset_hor_util.transform_img(gray, affinetrans, self.inp_res)
            gray_i = gray_i.crop((0, 0, self.inp_res[0], self.inp_res[1]))
            
            gray_copy_i = functional.to_tensor(copy.deepcopy(gray_i))
            
            gray_i = dataset_hor_util.get_mask_ROI(gray_i, bbox_obj_i)
            # Generate object mask
            gray_i = np.asarray(gray_i.resize((32, 32), Image.NEAREST))
            obj_mask_i = np.ma.getmaskarray(np.ma.masked_not_equal(gray_i, 0)).astype(int)
            obj_mask_i = torch.from_numpy(obj_mask_i)
            
            imgs.append(functional.to_tensor(img_i))
            mano_params.append(mano_param_i)
            Ks.append(K_i)
            obj_masks.append(obj_mask_i)
            p2ds.append(p2d_i)
            joints_uvs.append(joints_uv_i)
            bbox_hands.append(bbox_hand_i)
            bbox_objs.append(bbox_obj_i)
            joints_3ds.append(joints_3d_i)
            obj_poses.append(obj_pose_i)
            root_joints.append(root_joint_i)
            root_joint_flips.append(root_joint_flip_i)
            gray_copys.append(gray_copy_i)
            
            K_vis_s.append(K_vis_i)
            rot_mat_s.append(rot_mat)

            
        return np.concatenate(imgs, 0), \
        np.concatenate(mano_params, 0),   \
        np.concatenate(Ks, 0), \
        np.concatenate(obj_masks, 0), \
        np.concatenate(p2ds, 0), \
        np.concatenate(joints_uvs, 0), \
        np.concatenate(bbox_hands, 0), \
        np.concatenate(bbox_objs, 0), \
        np.concatenate(joints_3ds, 0), \
        np.concatenate(obj_poses, 0), \
        np.concatenate(root_joints, 0), \
        np.concatenate(root_joint_flips, 0), \
        np.concatenate(gray_copys, 0), \
        np.concatenate(K_vis_s, 0), \
        np.concatenate(rot_mat_s, 0), \
        np.array([True, False]), \
        np.array([gt_occ_cls, False])
    
    def data_crop(self, img, K, hand_joints_2d, p2d, grasping, K_vis):
        if grasping:
            return self.data_crop_fuse(img, K, hand_joints_2d, p2d, K_vis)
        else:
            return self.data_crop_hand(img, K, hand_joints_2d, p2d, K_vis)
    
    def data_crop_fuse(self, img, K, hand_joints_2d, p2d, K_vis):
        crop_hand = dataset_hor_util.get_bbox_joints(hand_joints_2d, bbox_factor=1.5)
        crop_obj = dataset_hor_util.get_bbox_joints(p2d, bbox_factor=1.5)
        bbox_hand = dataset_hor_util.get_bbox_joints(hand_joints_2d, bbox_factor=1.1)
        bbox_obj = dataset_hor_util.get_bbox_joints(p2d, bbox_factor=1.0)
        center, scale = dataset_hor_util.fuse_bbox(crop_hand, crop_obj, img.size)
        affinetrans, _ = dataset_hor_util.get_affine_transform(center, scale, self.inp_res)
        bbox_hand = dataset_hor_util.transform_coords(bbox_hand.reshape(2, 2), affinetrans).flatten()
        bbox_obj = dataset_hor_util.transform_coords(bbox_obj.reshape(2, 2), affinetrans).flatten()
        # Transform and crop
        img = dataset_hor_util.transform_img(img, affinetrans, self.inp_res)
        img = img.crop((0, 0, self.inp_res[0], self.inp_res[1]))
        
        K = affinetrans.dot(K)
        K_vis = affinetrans.dot(K_vis)
        
        rot_mat = np.identity(3)
        
        p2d = dataset_hor_util.transform_coords(p2d, affinetrans)  # obj landmark trans
        p2d = dataset_hor_util.normalize_joints(p2d, bbox_obj)
        
        #normalize landmarks to [0,1]
        hand_joints_2d = dataset_hor_util.transform_coords(hand_joints_2d, affinetrans)
        joints_uv = dataset_hor_util.normalize_joints(hand_joints_2d, bbox_hand)
        
        return functional.to_tensor(img), K, bbox_hand, bbox_obj, affinetrans, K_vis, rot_mat, joints_uv, p2d
    
    def data_crop_ho_pair(self, img, K, hand_joints_2d, p2d, img_aug, K_vis, gt_occ_cls):
        crop_hand = dataset_hor_util.get_bbox_joints(hand_joints_2d, bbox_factor=1.5)
        crop_obj = dataset_hor_util.get_bbox_joints(p2d, bbox_factor=1.5)
        bbox_hand = dataset_hor_util.get_bbox_joints(hand_joints_2d, bbox_factor=1.1)
        bbox_obj = dataset_hor_util.get_bbox_joints(p2d, bbox_factor=1.0)
        center, scale = dataset_hor_util.fuse_bbox(crop_hand, crop_obj, img.size)
        affinetrans, _ = dataset_hor_util.get_affine_transform(center, scale, self.inp_res)
        bbox_hand = dataset_hor_util.transform_coords(bbox_hand.reshape(2, 2), affinetrans).flatten()
        bbox_obj = dataset_hor_util.transform_coords(bbox_obj.reshape(2, 2), affinetrans).flatten()
        # Transform and crop
        img = dataset_hor_util.transform_img(img, affinetrans, self.inp_res)
        img = img.crop((0, 0, self.inp_res[0], self.inp_res[1]))
        
        K = affinetrans.dot(K)
        K_vis = affinetrans.dot(K_vis)
        
        rot_mat = np.identity(3)
        
        #normalize landmarks to [0,1]
        hand_joints_2d = dataset_hor_util.transform_coords(hand_joints_2d, affinetrans)
        joints_uv = dataset_hor_util.normalize_joints(hand_joints_2d, bbox_hand)
        
        p2d = dataset_hor_util.transform_coords(p2d, affinetrans)  # obj landmark trans
        p2d = dataset_hor_util.normalize_joints(p2d, bbox_obj)
        
        return np.concatenate([functional.to_tensor(img), functional.to_tensor(img_aug)], 0), \
        np.tile(K, (2, 1)), \
        np.tile(bbox_hand, 2), \
        np.tile(bbox_obj, 2), \
        np.tile(affinetrans, (2, 1)), \
        np.tile(K_vis, (2, 1)), \
        np.tile(rot_mat, (2, 1)), \
        np.array([True, False]), \
        np.array([gt_occ_cls, False]), \
        np.tile(joints_uv, (2, 1)), \
        np.tile(p2d, (2, 1))
    
    def data_crop_hand(self, img, K, hand_joints_2d, p2d, K_vis):
        crop_hand = dataset_hor_util.get_bbox_joints(hand_joints_2d, bbox_factor=1.5)
        bbox_hand = dataset_hor_util.get_bbox_joints(hand_joints_2d, bbox_factor=1.1)
        
        bbox_obj = dataset_hor_util.get_bbox_joints(p2d, bbox_factor=1.0)
        
        center, scale = dataset_hor_util.bbox_center_scale(crop_hand, img.size)
        
        affinetrans, _ = dataset_hor_util.get_affine_transform(center, scale, self.inp_res)
        bbox_hand = dataset_hor_util.transform_coords(bbox_hand.reshape(2, 2), affinetrans).flatten()
        bbox_obj = dataset_hor_util.transform_coords(bbox_obj.reshape(2, 2), affinetrans).flatten()
        # Transform and crop
        img = dataset_hor_util.transform_img(img, affinetrans, self.inp_res)
        img = img.crop((0, 0, self.inp_res[0], self.inp_res[1]))
        
        K = affinetrans.dot(K)
        K_vis = affinetrans.dot(K_vis)
        
        rot_mat = np.identity(3)
        
        p2d = dataset_hor_util.transform_coords(p2d, affinetrans)  # obj landmark trans
        p2d = dataset_hor_util.normalize_joints(p2d, bbox_obj)
        
        #normalize landmarks to [0,1]
        hand_joints_2d = dataset_hor_util.transform_coords(hand_joints_2d, affinetrans)
        joints_uv = dataset_hor_util.normalize_joints(hand_joints_2d, bbox_hand)
        
        return functional.to_tensor(img), K, bbox_hand, bbox_obj, affinetrans, K_vis, rot_mat, joints_uv, p2d
    
    def revert_cropped_image(self, cropped_img, ori_img, crop_affinetrans):
        # resize to desired resolution first
        cropped_img = cv2.resize(cropped_img, (self.inp_res[0], self.inp_res[1]), interpolation=cv2.INTER_NEAREST)  # NOTE: 256x256
        
        # apply the inverse transform to the generated image
        cropped_img_revert = dataset_hor_util.transform_img(Image.fromarray(cropped_img), inv(crop_affinetrans), ori_img.size)
        
        gen_coords = np.array([[0, 0], [cropped_img.shape[1], cropped_img.shape[0]]])  # [[0, 0], [128, 128]] or [[0, 0], [256, 256]]
        
        # apply the inverse transform to the generated coordinates
        cropped_coords_revert = dataset_hor_util.transform_coords(gen_coords, inv(crop_affinetrans)).flatten()
        
        # pay attentino to avoid black edge
        min_x, min_y, max_x, max_y = math.ceil(cropped_coords_revert[0]), math.ceil(cropped_coords_revert[1]), math.floor(cropped_coords_revert[2]), math.floor(cropped_coords_revert[3])
        
        # fill in the generated image to the original image
        aug_img = copy.deepcopy(ori_img)
        aug_img.paste(cropped_img_revert.crop((min_x, min_y, max_x, max_y)), (min_x, min_y))
        return aug_img
    
    
    def __len__(self):
        return len(self.sample_list_processed)
    

    def __getitem__(self,idx):
        sample = {}
        
        if self.load_nori:  # load from nori
            nori_id = self.nori_dict[self.sample_list_processed[idx]]
            data = self.fetcher.get(nori_id)
            sample_info = pickle.loads(data)
            img = sample_info['color']
            sample_key = sample_info["sample_key"]
        else:  # load from local
            sample_info = self.sample_dict[self.sample_list_processed[idx]]
            img =Image.open(os.path.join(self.dataset_root,sample_info["color_file"])).convert("RGB")
            sample_key = self.sample_list_processed[idx]
        
        # import ipdb; ipdb.set_trace()
        do_flip = (sample_info["mano_side"] == 'left') 
        #camintr
        fx = sample_info['intrinsics']['fx']
        fy = sample_info['intrinsics']['fy']
        cx = sample_info['intrinsics']['ppx']
        cy = sample_info['intrinsics']['ppy']
        K = np.zeros((3,3))
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cx
        K[1,2] = cy
        K[2,2] = 1
      
        # for object evaluation
        cam_intr = copy.deepcopy(K)
        
        # flip cx for a copied one
        K_vis = copy.deepcopy(K)
        if do_flip:
            K_vis[0,2] = 640 - cx - 1
        
        ori_K = copy.deepcopy(K)
        
        if do_flip:
            img = np.array(img, np.uint8 , copy=True)
            img = img[:, ::-1, :]
            img = Image.fromarray(np.uint8(img))
        # for debug and visualization
        ori_img = functional.to_tensor(img)
        
        gt_grasping = self.grasp_gt_dict[sample_key]
        
        # condition for occlusion
        occ_condition = (self.occ_ratio_dict[sample_key] > self.occ_threshold)
        # occlusion label for regression
        gt_occ_cls = not occ_condition
        
        if self.mode == "train":
            #hand information
            mano_pose_pca_mean =  np.array(sample_info["pose_m"],dtype=np.float32).squeeze()
            mano_betas = np.array(sample_info["mano_betas"],dtype=np.float32)
            hand_joint_3d = np.array(sample_info["joint_3d"],dtype=np.float32).squeeze()
            joints_uv =  np.array(sample_info["joint_2d"],dtype=np.float32).squeeze()
            
            root_joint = copy.deepcopy(hand_joint_3d[self.joint_root_id])  # root joint before flipping
            
            # #前三维是全局旋转，后45维是pose ,后三维是平移
            mano_pose_aa_mean = np.concatenate((mano_pose_pca_mean[0:3],np.matmul(mano_pose_pca_mean[3:48],np.array(mano_layer.smpl_data["hands_components"])),mano_pose_pca_mean[48:]),axis=0)
            if do_flip:
                #注意左手的PCA转换系数与右手不同
                mano_pose_aa_mean = np.concatenate((mano_pose_pca_mean[0:3],np.matmul(mano_pose_pca_mean[3:48],np.array(mano_layerl.smpl_data["hands_components"])),mano_pose_pca_mean[48:]),axis=0)
                #包括全局旋转在内mano系数的后两维*-1
                mano_pose_aa_mean_wo_trans = mano_pose_aa_mean[:48].reshape(-1,3)
                mano_pose_aa_mean_wo_trans[:,1:]  *= -1
                mano_pose_aa_mean[0:48] = mano_pose_aa_mean_wo_trans.reshape(-1)               
                #3D点关于Y-O-X对称
                hand_joint_3d[:,0] *= -1 
                #2D点关于W/2对称
                joints_uv[:,0] = np.array(img.size[0],dtype=np.float32)  - joints_uv[:,0] - 1
                
            joints_uv_copy = copy.deepcopy(joints_uv)
            
            #前三维是全局旋转，后45维是pose
            mano_pose_aa_flat = np.concatenate((mano_pose_aa_mean[:3],mano_pose_aa_mean[3:48]+mano_layer.smpl_data["hands_mean"]),axis=0)
            # #前三维是全局旋转，后45维是pose ,后十维度是shape
            mano_param = np.concatenate((mano_pose_aa_flat,mano_betas)) 

            root_joint_flip = copy.deepcopy(hand_joint_3d[self.joint_root_id])  # root joint after flipping
                        
            if self.load_nori:
                gray = sample_info['gray'].convert("L")  # prevent some object seg becoming 3-channels out of unknown reasons
            else:
                gray = Image.open(os.path.join(self.dataset_root,sample_info["object_seg_file"]))
            if do_flip:
                gray = np.array(gray, np.uint8 , copy=True)
                gray = gray[:, ::-1]
                gray = Image.fromarray(np.uint8(gray))
            sample["ori_gray"] = functional.to_tensor(copy.deepcopy(gray))
            #object_information
            grasp_object_pose = np.array(sample_info["pose_y"][sample_info['ycb_grasp_ind']],dtype=np.float32)
            p3d, p2d= dex_ycb_hor_util.projectPoints(self.obj_bbox3d[sample_info["ycb_ids"][sample_info["ycb_grasp_ind"]]],K,rt=grasp_object_pose)  # NOTE: originally, it flip projected 2d keypoints. But with flipped K it is not correct anymore.
            
            grasp_object_pose = dex_ycb_hor_util.pose_from_initial_martrix(grasp_object_pose)  # 4x4
            if do_flip:
                p2d[:,0] = np.array(img.size[0],dtype=np.float32)  - p2d[:,0] - 1
                p3d[:,0] *= -1
                # grasp_object_pose = np.diag([-1, 1, 1, 1]).dot(grasp_object_pose)  # uncomment this to make visualization correct
            
            #data augumentation
            if self.contrastive: # return pair of hand and synthetic data. There're two samples for each original image
                if gt_grasping:
                    if self.load_nori:
                        gen_data_meta = self.gen_nori_dict[self.sample_list_processed[idx]]
                        nori_id = gen_data_meta['nori_id']
                        gen_data = self.fetcher.get(nori_id)
                        gen_sample_info = pickle.loads(gen_data)  # dict_keys(['sample_key', 'gen_img', 'ori_MPJPE', 'gen_MPJPE', 'ori_PA_MPJPE', 'gen_PA_MPJPE', 'ori_MPVPE', 'gen_MPVPE', 'ori_PA_MPVPE', 'gen_PA_MPVPE', 'ori_score', 'gen_score'])
                        gen_img = gen_sample_info['gen_img']  # 128x128
                        
                        # get the boundding box
                        crop_hand = dataset_hor_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
                        crop_obj = dataset_hor_util.get_bbox_joints(p2d, bbox_factor=1.5)
                        center, scale = dataset_hor_util.fuse_bbox(crop_hand, crop_obj, img.size)
                        gen_affinetrans, _ = dataset_hor_util.get_affine_transform(center, scale, self.inp_res)
                        
                        aug_img = self.revert_cropped_image(gen_img, img, gen_affinetrans)
                    else:
                        # TODO: load from local, which is not implemented yet
                        raise NotImplementedError
                    
                    # condition for gen score
                    ori_score = gen_data_meta['ori_score']
                    gen_score = gen_data_meta['gen_score']
                    score_condition = (gen_score + self.score_threshold - ori_score) >=  0
                    
                    if self.filter_rule == 'score':
                        filter_condition = score_condition
                    elif self.filter_rule == 'occ':
                        filter_condition = occ_condition
                    elif self.filter_rule == 'and':
                        filter_condition = (score_condition and occ_condition)
                    elif self.filter_rule == 'or':
                        filter_condition = (score_condition or occ_condition)
                    else:  # filter all the gen samples
                        filter_condition = True
                        
                    # store as the mask to the contrastive loss
                    sample["is_filtered"] = filter_condition  # if gen image filtered, then True. If gen image is used, then False.
                        
                    if not filter_condition:  # do not filter any case or do not need to be filtered
                        # return hand-object and generated hand pair
                        img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj, hand_joint_3d, grasp_object_pose, root_joint, root_joint_flip, gray, K_vis, aug_rot_mat, gt_grasping_ret, gt_occ_cls_ret = self.data_aug_ho_pair(img, mano_param, joints_uv, K, gray, p2d, hand_joint_3d, grasp_object_pose, root_joint, root_joint_flip, aug_img, K_vis, gt_occ_cls)
                    else: 
                        # return hand-object single
                        img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj, hand_joint_3d, grasp_object_pose, root_joint, root_joint_flip, gray, K_vis, aug_rot_mat = self.data_aug_fuse(img, mano_param, joints_uv, K, gray, p2d, hand_joint_3d, grasp_object_pose, root_joint, root_joint_flip, K_vis)

                else: 
                    # not grasping, return hand single, do not need distillation
                    img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj, hand_joint_3d, grasp_object_pose, root_joint, root_joint_flip, gray, K_vis, aug_rot_mat = self.data_aug_hand(img, mano_param, joints_uv, K, gray, p2d, hand_joint_3d, grasp_object_pose, root_joint, root_joint_flip, K_vis)
                    sample["is_filtered"] = True  # just to align with the gen pair version. If hand image, true
                    
                 # create pair for single output
                if img.shape[0] == 3:
                    img = np.tile(img, (2, 1, 1))
                    mano_param = np.tile(mano_param, (2,))
                    K = np.tile(K, (2, 1))
                    obj_mask = np.tile(obj_mask, (2, 1))
                    p2d = np.tile(p2d, (2, 1))
                    joints_uv = np.tile(joints_uv, (2, 1))
                    bbox_hand = np.tile(bbox_hand, 2)
                    bbox_obj = np.tile(bbox_obj, 2)
                    hand_joint_3d = np.tile(hand_joint_3d, (2, 1))
                    grasp_object_pose = np.tile(grasp_object_pose, (2, 1))
                    root_joint = np.tile(root_joint, 2)
                    root_joint_flip = np.tile(root_joint_flip, 2)
                    gray = np.tile(gray, (2, 1, 1))
                    K_vis = np.tile(K_vis, (2, 1))
                    aug_rot_mat = np.tile(aug_rot_mat, (2, 1))
                    gt_grasping_ret = np.tile(gt_grasping, 2)
                    gt_occ_cls_ret = np.tile(gt_occ_cls, 2)
                
            else:  # return original single image, either hand or hand-object
                # use gt grasping to determine the input
                img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj, hand_joint_3d, grasp_object_pose, root_joint, root_joint_flip, gray, K_vis, aug_rot_mat = self.data_aug(img, mano_param, joints_uv, K, gray, p2d, hand_joint_3d, grasp_object_pose, root_joint, root_joint_flip, gt_grasping, K_vis)
                gt_grasping_ret = np.tile(gt_grasping, 1) # create additional dimension
                gt_occ_cls_ret = np.tile(gt_occ_cls, 1)
                sample["is_filtered"] = True  # not meaningful, because no gen data
        
            sample["obj_p2d"] = p2d
            sample["obj_mask"] = obj_mask
            sample["gray"] = gray
        else:  # test mode
            # object
            grasp_object_pose = dex_ycb_hor_util.pose_from_initial_martrix(np.array(sample_info["pose_y"][sample_info['ycb_grasp_ind']],dtype=np.float32))
            _,p2d = dex_ycb_hor_util.projectPoints(self.obj_bbox3d[sample_info["ycb_ids"][sample_info["ycb_grasp_ind"]]],K,rt=grasp_object_pose)  # NOTE: originally, it flip projected 2d keypoints. But with flipped K it is not correct anymore.
            
            # added for evaluation
            mano_pose_pca_mean =  np.array(sample_info["pose_m"],dtype=np.float32).squeeze()
            mano_betas = np.array(sample_info["mano_betas"],dtype=np.float32)
            mano_pose_aa_mean = np.concatenate((mano_pose_pca_mean[0:3],np.matmul(mano_pose_pca_mean[3:48],np.array(mano_layer.smpl_data["hands_components"])),mano_pose_pca_mean[48:]),axis=0)
            
            if do_flip:
                p2d[:,0] = np.array(img.size[0],dtype=np.float32)  - p2d[:,0] - 1
                # grasp_object_pose = np.diag([-1, 1, 1, 1]).dot(grasp_object_pose)  # uncomment this to make visualization correct
                
                #注意左手的PCA转换系数与右手不同
                mano_pose_aa_mean = np.concatenate((mano_pose_pca_mean[0:3],np.matmul(mano_pose_pca_mean[3:48],np.array(mano_layerl.smpl_data["hands_components"])),mano_pose_pca_mean[48:]),axis=0)
                #包括全局旋转在内mano系数的后两维*-1
                mano_pose_aa_mean_wo_trans = mano_pose_aa_mean[:48].reshape(-1,3)
                mano_pose_aa_mean_wo_trans[:,1:]  *= -1
                mano_pose_aa_mean[0:48] = mano_pose_aa_mean_wo_trans.reshape(-1)
                
             #前三维是全局旋转，后45维是pose
            mano_pose_aa_flat = np.concatenate((mano_pose_aa_mean[:3],mano_pose_aa_mean[3:48]+mano_layer.smpl_data["hands_mean"]),axis=0)
            # #前三维是全局旋转，后45维是pose ,后十维度是shape
            mano_param = np.concatenate((mano_pose_aa_flat,mano_betas))
                
            #hand 
            joints_uv =  np.array(sample_info["joint_2d"],dtype=np.float32).squeeze()
            hand_joint_3d = np.array(sample_info["joint_3d"],dtype=np.float32).squeeze()
            root_joint = copy.deepcopy(hand_joint_3d[self.joint_root_id])  # root joint before flipping
            
            if do_flip:
                joints_uv[:,0] = np.array(img.size[0],dtype=np.float32)  - joints_uv[:,0] - 1
                hand_joint_3d[:,0] *= -1

            root_joint_flip = copy.deepcopy(hand_joint_3d[self.joint_root_id])  # root joint after flipping
            
            joints_uv_copy = copy.deepcopy(joints_uv)
            
            #crop
            # return original single sample
            img, K, bbox_hand, bbox_obj, affinetrans, K_vis, aug_rot_mat, joints_uv, p2d = self.data_crop(img, K, joints_uv, p2d, gt_grasping, K_vis)
            gt_grasping_ret = np.tile(gt_grasping, 1) # create additional dimension
            gt_occ_cls_ret = np.tile(gt_occ_cls, 1)

            # for object metric
            # objName = annotations['objName']
            obj_bbox3d = self.obj_bbox3d[sample_info["ycb_ids"][sample_info["ycb_grasp_ind"]]]  # (21, 3). metrics in m.
            # apply rotation and translation to object
            sample["obj_bbox3d"] = obj_bbox3d # the object keypoints at canonical space
            sample["affinetrans"]  = affinetrans
            sample["obj_pose"] = grasp_object_pose
            sample["obj_cls"] = sample_info["ycb_ids"][sample_info["ycb_grasp_ind"]]
            # sample["obj_diameter"] = self.obj_diameters[sample["obj_cls"]]
            sample["cam_intr"] = cam_intr  # original intrincis without cropping
            
        # for general hand-object input
        sample["img"] = img  # (6, 128, 128)
        
        if "input_img_shape_1" in self.cfg.data:
            input_img_shape_1 = self.cfg.data.input_img_shape_1
            # resize img to another size
            img_1 = functional.resize(img, input_img_shape_1)
            sample["img_1"] = img_1
        
        
        sample["mano_param"] = mano_param
        sample["bbox_hand"] = bbox_hand
        sample["bbox_obj"] = bbox_obj
        sample["aug_rot_mat"] = aug_rot_mat
        sample["gt_grasping"] = gt_grasping_ret
        sample["gt_occ_cls"] = gt_occ_cls_ret
        sample["obj_cls_label"] = np.array([sample_info["ycb_ids"][sample_info["ycb_grasp_ind"]] - 1])
        sample["gt_occ_ratio"] = self.occ_ratio_dict[sample_key]
        sample["do_flip"] = do_flip
        sample["joints_coord_cam"] = hand_joint_3d  # for hand evaluation. For DexYCB, it is the same as mano regression. But for HO3D and FreiHAND, it's different.
        
        B = img.shape[0]//3
        
        intrinsics = []
        mano_poses = []
        mano_shapes = []
        joints_imgs_hr = []
        
        for i in range(B):
            d = 3
            K_i = K[i*d:(i+1)*d]
            intrinsics_i = np.array([K_i[0, 0], K_i[1, 1], K_i[0, 2], K_i[1, 2]])  # the intrinsics after image crop
            intrinsics.append(intrinsics_i)
            
            d1 = 48
            d2 = 10
            mano_poses.append(mano_param[i*(d1+d2):i*(d1+d2)+d1])
            mano_shapes.append(mano_param[i*(d1+d2)+d1:i*(d1+d2)+d1+d2])
            
            d1 = 21
            d2 = 4
            joints_uv_hr = dataset_hor_util.denormalize_joints(joints_uv[i*d1:(i+1)*d1], bbox_hand[i*d2:(i+1)*d2])
            joints_uv_hr_norm = joints_uv_hr / self.inp_res
            joints_imgs_hr.append(joints_uv_hr_norm)
        
        sample['intrinsics'] = np.concatenate(intrinsics, 0)
        
        sample["root_joint"] = root_joint  # root joint before flipping
        sample["root_joint_flip"] = root_joint_flip # root joint after flipping
        
        
        if self.bbox_hand_normalize_2d_joint:
            # gt 2d joint for HFL-Net, normalized with bbox hand
            sample["joints_img"] = joints_uv
        else:
            # gt 2d joint for hand-mesh reconstruction methods, normalized with image
            sample["joints_img"] = np.concatenate(joints_imgs_hr, 0)  # (21*B, 2)  # I'm not too sure about the correctness now, please refer to original HFL-Net: sample["joints2d"] = joints_uv
        
        sample["mano_pose"] = np.concatenate(mano_poses, 0)
        sample["mano_shape"] = np.concatenate(mano_shapes, 0)
        sample["val_mano_pose"]= copy.deepcopy(sample["mano_pose"])
        
        # camera intrinsics, [fx, fy, cx, cy]
        ori_intrinsics = np.array([ori_K[0, 0], ori_K[1, 1], ori_K[0, 2], ori_K[1, 2]])  # the intrinsics before image crop
        
        
        # Fields for SimpleHand
        # normalize relative to original image space
        uv_norm = joints_uv_copy / np.array([640, 480], dtype=np.float32)
            
        coord_valid = (uv_norm > 0).astype("float32") * (uv_norm < 1).astype("float32") # Nx2x21x2
        coord_valid = coord_valid[:, 0] * coord_valid[:, 1]
        
        trans_coord_valid = (joints_uv_hr_norm > 0).astype("float32") * (joints_uv_hr_norm < 1).astype("float32") # Nx2x21x2
        trans_coord_valid = trans_coord_valid[:, 0] * trans_coord_valid[:, 1]
        trans_coord_valid *= coord_valid
        
        xyz_valid = 1
        if trans_coord_valid[9] == 0 and trans_coord_valid[0] == 0:
            xyz_valid = 0
        
        sample.update({
            # "img": img,
            "uv": joints_uv_hr_norm, # joints_uv_hr,
            "xyz": hand_joint_3d,
            # "vertices": results['vertices'],                
            "uv_valid": trans_coord_valid,
            "xyz_valid": xyz_valid,
        })
        
        
        return sample

    # only update object metric, since hand metric is calculated in loss.py
    def evaluate(self, batch_input, batch_output):
        if "preds_obj_0" in batch_output[0]:
            # object evaluation in HFL-Net
            self.REP_res_dict, self.ADD_res_dict = eval_batch_obj(batch_input, batch_output, self.mesh_dict, self.REP_res_dict, self.ADD_res_dict, self.obj_eval_mode)
    
    # ouput the metric for object
    def print_eval_result(self, epoch):
        # save object metric
        output_json_obj = osp.join(self.cfg.base.model_dir, "obj_result_epoch{}_{}.json".format(epoch, self.obj_eval_mode))
        self.dump_object(output_json_obj)
    
    # print eval object metric
    def dump_object(self, pred_out_path):
        # the HFL-Net version, which is hard for unseen objects
        ADD, REP, ADD_MEAN, REP_MEAN = eval_object_pose(self.REP_res_dict, self.ADD_res_dict, self.diameter_dict, unseen_objects=self.unseen_objects, rep_threshold=self.rep_threshold, add_threshold=self.add_threshold)
        
        REP = {dex_ycb_hor_util._YCB_CLASSES[k]:v for k, v in REP.items()}
        REP.update({"average": REP_MEAN})
        
        ADD = {dex_ycb_hor_util._YCB_CLASSES[k]:v for k, v in ADD.items()}
        ADD.update({"average": ADD_MEAN})
        
        out = {"REP": REP, "ADD": ADD}
        
        pred_out_base = osp.splitext(pred_out_path)[0]
        pred_out_path = pred_out_base + f"_{self.rep_threshold}_{self.add_threshold}.json"
        
        with open(pred_out_path, "w") as f:
            json.dump(out, f)
        print("Dumped object predictions to: ", pred_out_path)

    # directly return the hand predictions. For other purpose like individually eval on hand/hand-object images, or for whole-dataset visualization 
    def get_predictions(self, batch_input, batch_output):
        batch_size = len(batch_output)
        pred_result = [[], [], [], []]  # [joints_out, mesh_out, joints_gt, mesh_gt]
        for n in range(batch_size):
            input = batch_input[n]
            output = batch_output[n]
            
            verts_out = output["pred_verts3d_cam"]
            joints_out = output["pred_joints3d_cam"]
            verts_gt = output["gt_verts3d_cam"]
            joints_gt = output["gt_joints3d_cam"]

            verts_gt -= joints_gt[self.joint_root_id]
            joints_gt -= joints_gt[self.joint_root_id]

            # root centered
            verts_out -= joints_out[self.joint_root_id]
            joints_out -= joints_out[self.joint_root_id]

            # root align
            gt_root_joint_cam = input["root_joint_flip"]  # after flipping
            
            verts_out += gt_root_joint_cam
            joints_out += gt_root_joint_cam
            verts_gt += gt_root_joint_cam
            joints_gt += gt_root_joint_cam

            pred_result[0].append(joints_out)
            pred_result[1].append(verts_out)
            pred_result[2].append(joints_gt)
            pred_result[3].append(verts_gt)

        return pred_result
    