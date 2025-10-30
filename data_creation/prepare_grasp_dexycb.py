import os
import json
import argparse
import numpy as np
import os.path as osp

import torch
import yaml
import copy
import cv2
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from collections import defaultdict

import pyrender
from tqdm import tqdm

from common.utils.preprocessing import np_concatenate, np_inverse

def load_data():
    data_path = os.path.join(args.root_dir, "dex_ycb_{}_data.json".format(args.data_split))

    with open(data_path, 'r', encoding='utf-8') as f:
        sample_dict = json.load(f)
                
    return sample_dict


def load_ycb_objects():
    obj_file = {k: os.path.join(f'{args.root_dir}/models', v, 'textured_simple.obj') for k, v in YCB_CLASSES.items()}
    all_obj = {}
    for k, v in obj_file.items():
        mesh = trimesh.load(v)
        if args.version == 1:
            mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
            mesh.visual.face_colors = np.array(YCB_COLORS[k])
        elif args.version == 2:
            pass
        else:
            raise NotImplementedError(f'Not implement version {args.version}')
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        all_obj[k] = mesh
    return all_obj


def main(anns):
    # train: 406888, test: 78768, train with bbox: 401507
    out = {}

    num_data = len(anns)
    cnt = 0
    step = 0
    for key, ann in tqdm(anns.items(), total=len(anns)):
        img_path = os.path.join(args.root_dir, ann['color_file'])

        # load YCB meshes.
        meta_file = os.path.join('/'.join(img_path.split('/')[:-2]), 'meta.yml')
        with open(meta_file, 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        ycb_ids = meta['ycb_ids']

        # load label file
        label_path = img_path.replace('color', 'labels').replace('jpg', 'npz')
        label = np.load(label_path)
        pose_y = label['pose_y']
        current_pose_y = pose_y[meta['ycb_grasp_ind']]

        # get init object pose (first frame)
        current_color_name = img_path.split('/')[-1]
        init_label_path = img_path.replace(current_color_name, 'labels_000000.npz')
        init_label = np.load(init_label_path)
        init_current_pose_y = init_label['pose_y'][meta['ycb_grasp_ind']]
        
        # correct grasping condition
        concatenated = np_concatenate(np_inverse(current_pose_y), init_current_pose_y)
        rot_trace = concatenated[0, 0] + concatenated[1, 1] + concatenated[2, 2]
        residual_rotdeg = np.arccos(np.clip(0.5 * (rot_trace - 1), a_min=-1.0, a_max=1.0)) * 180.0 / np.pi
        # residual_transmag = np.linalg.norm(concatenated[:3, 3], axis=-1)
        residual_transmag = np.linalg.norm(concatenated[:, 3], axis=-1)
        err_r = np.mean(residual_rotdeg)
        err_t = np.mean(residual_transmag)
        grasping = (err_r > 5) or (err_t > 0.05)

        out[key] = bool(grasping)
        
        if args.debug:
            break
    
    # an updated version
    out_path = os.path.join(args.root_dir, "nori", f"grasp_{args.data_split}_v2.npy")
    print(out_path)
    np.save(out_path, out)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser('DexYCB grasping label preparation', add_help=True)
    parser.add_argument('--root_dir', '-r', type=str, default='data/DexYCB', help='root data dir')
    parser.add_argument('--data_split', '-p', type=str, default='s0_train', help='data split')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    anns = load_data()

    main(anns)