# [CVPR 2025] UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation.

## Todo List:
- [x] Inference code and checkpoints for UniHOPE and previous methods
- [ ] Code for generative de-occluder (without which training is infeasible, but inference is OK)

NOTE: Our original code uses a special format on Object Storage Service (OSS) to store all the training and testing data. Due to the server migration, we need some time to clean the code such that it can support loading from local disk. Stay tuned!

## Usage:

### 0. Environment
* System: Ubuntu 20.04.6 LTS
* CUDA version: 11.8
* Python version: 3.9.18
* PyTorch version: 2.0.0

### 1. Data preparation
First, please ensure the [DexYCB dataset](https://dex-ycb.github.io/) and [HO3D_v2 dataset](https://1drv.ms/f/c/11742dd40d1cbdc1/ElPb2rhOCeRMg-dFSM3iwO8B5nS1SgnQJs9F6l28G0pKKg?e=TMuxgr) are downloaded into `./data`.
Besides the standard datasets, we have generated several annotation files, including labels of grasping status and hand occlusion for DexYCB/HO3D. You can download these files from [https://drive.google.com/drive/folders/1ZuafaCkx34atbtz_wAHkc8dikWU0Q5Qz?usp=drive_link](https://drive.google.com/drive/folders/1ZuafaCkx34atbtz_wAHkc8dikWU0Q5Qz?usp=drive_link), and put them under `./data/DexYCB` and `./data/HO3D_v2` seperately.

You may also use scripts under `./data_creation` to generate some files by your own. For example, 
To creating all the splits we used (i.e., s0, s1, and s3) for DexYCB, simply run: 

```python data_creation/create_dataset_dexycb.py --root_dir data/DexYCB```

To prepare grasping labels for DexYCB s0 split testset, run:

```python data_creation/prepare_grasp_dexycb.py --root_dir data/DexYCB --data_split s0_test```


Please note that for HO3D, we choose to annotate the grasping status manually rather than using an automatic pipeline to ensure accuracy.


### 2. Evaluation:
We provide three scripts that can be used for evaluation.

* `eval.py`: A thorough evaluation with full hand metrics, including MPJPE, MPVPE, F-score, AUC, etc.

* `eval_h_ho.py`: Evaluation with hand-only scene and hand-object scene seperated (e.g., Table 2 in the main paper).

* `eval_occ_intervals.py`:  Evaluation on different object-caused occlusion levels (e.g., Table 6 in the main paper).


Here is an example of evaluating UniHOPE on DexYCB s0 split.

```accelerate launch --config_file yaml/{$N}_gpu.yaml eval.py --model_dir experiment/unihope.dexycb_s0 --resume experiment/unihope.dexycb_s0/test_model_best.pth```

where `N` is the number of GPUs you want to use.

Another example is to evaluate classifier-combined HPE and HOPE (i.e., A+B in the main paper) on DexYCB s3 split.

```accelerate launch --config_file yaml/{$N}_gpu.yaml eval.py --model_dir experiment/classifier_h2onet_hflnet.dexycb_s3/ --resume_h experiment/h2onet.dexycb_s3.h_train/test_model_best.pth  --resume_ho experiment/hflnet.dexycb_s3.ho_train/test_model_best.pth --resume_cls experiment/classifier.dexycb_s3.h-ho_train/test_model_best.pth```

You can find all the available checkpoints in the below link: [https://drive.google.com/drive/folders/1SI61ynvsC6Z3DtBk-KwR-ODacA2plzqh?usp=sharing](https://drive.google.com/drive/folders/1SI61ynvsC6Z3DtBk-KwR-ODacA2plzqh?usp=sharing), and put them under corresponding folder of `./experiment`.

## Citation:
```
@InProceedings{Wang_2025_CVPR,
    author    = {Wang, Yinqiao and Xu, Hao and Heng, Pheng-Ann and Fu, Chi-Wing},
    title     = {UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {12231-12241}
}
```

## Acknolwegement:
* [MobRecon](https://github.com/SeanChenxy/HandMesh)
* [HancOccNet](https://github.com/namepllet/HandOccNet) 
* [H2ONet](https://github.com/hxwork/H2ONet_Pytorch)
* [HandRefiner](https://github.com/wenquanlu/HandRefiner)