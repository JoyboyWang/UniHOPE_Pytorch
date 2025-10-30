# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Example of creating DexYCB datasets."""

import os
import os.path as osp
import json
import cv2

from dex_ycb_toolkit.factory import get_dataset
import numpy as np

from tqdm import tqdm

def main():
	anno_dir = args.root_dir
	step_size = 1 # default 1
	
	setup = ['s0', 's1', 's3']
	# for split in ('train', 'val', 'test'):
	for split in ('train', 'test'):
		name = '{}_{}'.format(setup, split)
			
		print('Dataset name: {}'.format(name))

		dataset = get_dataset(name)
		print('Dataset size: {}'.format(len(dataset)))

		annotations = {}
		for idx in tqdm(range(0, len(dataset), step_size), total=len(dataset)//step_size):
			sample = dataset[idx]

			mano_betas = sample['mano_betas']
			mano_side = sample['mano_side']

			label_path = osp.join(root_data_dir, sample['label_file'])
			label = dict(np.load(label_path)) # joint_3d, joint_2d for each view differs generally
			joint_3d = label['joint_3d'][0]
			joint_2d = label['joint_2d'][0]
			pose_m = label['pose_m'][0]

			if (joint_3d == -1).all() or (joint_2d == -1).all() or (pose_m == 0).all():
				continue

			sample_key = f"id_{idx}"
			
			annotations[sample_key] = sample
			
			label.pop('seg')
			annotations[sample_key].update(label)
			
			annotations[sample_key].update({
				'object_seg_file': osp.join('object_render', sample['color_file'].replace('color', 'grasp_object_seg').replace('jpg', 'png'))
			})
			
			for k, v in annotations[sample_key].items():
				if type(v) == np.ndarray:
					annotations[sample_key][k] = v.tolist()
			
			if args.debug:
				break
			
		# save annotations
		if step_size != 1:
			output_path = os.path.join(anno_dir, 'dex_ycb_' + name + f'_data_{step_size}.json')
		else:
			output_path = os.path.join(anno_dir, 'dex_ycb_' + name + '_data.json')
		
		print('output_path:', output_path)
		with open(output_path, 'w') as f:
			json.dump(annotations, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DexYCB annotation generation', add_help=True)
    parser.add_argument('--root_dir', '-r', type=str, default='data/DexYCB', help='root data dir')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

	main()
