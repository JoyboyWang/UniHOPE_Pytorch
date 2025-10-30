import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json
import torch
import copy
import open3d as o3d
from scipy.linalg import orthogonal_procrustes
from easydict import EasyDict as edict
from tqdm import tqdm
from pycocotools.coco import COCO

from model.fetch_model import fetch_model
from common import tool
from common.manager import Manager
from common.config import Config
from common.utils.mano import MANO
from common.utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db, rigid_transform_3D
from common.utils.manopth.mano.webuser.smpl_handpca_wrapper_HAND_only import load_model
from data_loader.data_loader import fetch_test_dataloader
from torch.nn.parallel import DataParallel as DP
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="", type=str, help="Directory containing params.json")
parser.add_argument("--resume", default=None, type=str, help="Path of model weights")
parser.add_argument("--resume_cls", default=None, type=str, help="Path of classification model weights")
parser.add_argument("--resume_h", default=None, type=str, help="Path of hand model weights")
parser.add_argument("--resume_ho", default=None, type=str, help="Path of hand-object model weights")


def load_two_in_one_cls_weights(model, logger, resume_h, resume_ho, resume_cls=None):
    # Remove 'module' prefix
    if resume_cls is not None:  # classification model is optional
        # cls model
        cls_model_weight = torch.load(resume_cls)
        
        new_cls_model_weight = OrderedDict()
        for k, v in cls_model_weight['state_dict'].items():
            name = k[7:] if k.startswith('module.') else k
            new_cls_model_weight[name] = v
        
        if hasattr(model, 'module'):
            missing_keys_0, unexpected_keys_0 = model.module.cls_model.load_state_dict(new_cls_model_weight, strict=False)
        else:
            missing_keys_0, unexpected_keys_0 = model.cls_model.load_state_dict(new_cls_model_weight, strict=False)
        
        logger.info("Load pre-trained weight {}".format(resume_cls))
        print(len(missing_keys_0), len(unexpected_keys_0))
    
    # h model
    h_model_weight = torch.load(resume_h)
    
    new_h_model_weight = OrderedDict()
    for k, v in h_model_weight['state_dict'].items():
        name = k[7:] if k.startswith('module.') else k
        new_h_model_weight[name] = v
    
    if hasattr(model, 'module'):
        missing_keys_1, unexpected_keys_1 = model.module.h_model.load_state_dict(new_h_model_weight, strict=False)
    else:
        missing_keys_1, unexpected_keys_1 = model.h_model.load_state_dict(new_h_model_weight, strict=False)
    
    logger.info("Load pre-trained weight {}".format(resume_h))
    print(len(missing_keys_1), len(unexpected_keys_1))
    
    # ho model
    ho_model_weight = torch.load(resume_ho)
    
    new_ho_model_weight = OrderedDict()
    for k, v in ho_model_weight['state_dict'].items():
        name = k[7:] if k.startswith('module.') else k
        new_ho_model_weight[name] = v
    
    if hasattr(model, 'module'):
        missing_keys_2, unexpected_keys_2 = model.module.ho_model.load_state_dict(new_ho_model_weight, strict=False)
    else:
        missing_keys_2, unexpected_keys_2 = model.ho_model.load_state_dict(new_ho_model_weight, strict=False)
    
    logger.info("Load pre-trained weight {}".format(resume_ho))
    print(len(missing_keys_2), len(unexpected_keys_2))

class EvalUtil:
    """ Util class for evaluation networks.
    """

    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred, skip_check=False):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        if not skip_check:
            keypoint_gt = np.squeeze(keypoint_gt)
            keypoint_pred = np.squeeze(keypoint_pred)
            keypoint_vis = np.squeeze(keypoint_vis).astype('bool')

            assert len(keypoint_gt.shape) == 2
            assert len(keypoint_pred.shape) == 2
            assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds

def verts2pcd(verts, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    if color is not None:
        if color == 'r':
            pcd.paint_uniform_color([1, 0.0, 0])
        if color == 'g':
            pcd.paint_uniform_color([0, 1.0, 0])
        if color == 'b':
            pcd.paint_uniform_color([0, 0, 1.0])
    return pcd


def calculate_fscore(gt, pr, th=0.01):
    gt = verts2pcd(gt)
    pr = verts2pcd(pr)
    # d1 = o3d.compute_point_cloud_to_point_cloud_distance(gt, pr)  # closest dist for each gt point
    # d2 = o3d.compute_point_cloud_to_point_cloud_distance(pr, gt)  # closest dist for each pred point
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))  # how many of our predicted points lie close to a gt point?
        precision = float(sum(d < th for d in d1)) / float(len(d1))  # how many of gt points are matched?

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0
    return fscore, precision, recall


def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def align_by_trafo(mtx, trafo):
    t2 = mtx.mean(0)
    mtx_t = mtx - t2
    R, s, s1, t1 = trafo
    return np.dot(mtx_t, R.T) * s * s1 + t1 + t2


class curve:

    def __init__(self, x_data, y_data, x_label, y_label, text):
        self.x_data = x_data
        self.y_data = y_data
        self.x_label = x_label
        self.y_label = y_label
        self.text = text


def createHTML(outputDir, curve_list):
    import base64
    curve_data_list = list()
    for idx, item in enumerate(curve_list):
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(item.x_data, item.y_data)
        ax.set_xlabel(item.x_label)
        ax.set_ylabel(item.y_label)
        img_path = os.path.join(outputDir, "img_path.{}.png".format(idx))
        plt.savefig(img_path, bbox_inches=0, dpi=300)

        # write image and create html embedding
        # with open(img_path, 'rb') as f:
        #     data_uri1 = f.read()
        # with open(img_path, "rb") as img_file:
        #     data_uri1 = base64.b64encode(img_file.read())
        # # data_uri1 = base64.b64encode(open(img_path, 'rb').read().tobytes()).replace('\n', '')
        # data_uri1 = data_uri1.replace('\n', '')
        # img_tag1 = 'src="data:image/png;base64,{0}"'.format(data_uri1)
        # curve_data_list.append((item.text, img_tag1))

        # os.remove(img_path)

    # htmlString = '''<!DOCTYPE html>
    # <html>
    # <body>
    # <h1>Detailed results:</h1>'''

    # for i, (text, img_embed) in enumerate(curve_data_list):
    #     htmlString += '''
    #     <h2>%s</h2>
    #     <p>
    #     <img border="0" %s alt="FROC" width="576pt" height="432pt">
    #     </p>
    #     <p>Raw curve data:</p>

    #     <p>x_axis: <small>%s</small></p>
    #     <p>y_axis: <small>%s</small></p>

    #     ''' % (text, img_embed, curve_list[i].x_data, curve_list[i].y_data)

    # htmlString += '''
    # </body>
    # </html>'''

    # htmlfile = open(os.path.join(outputDir, "scores.html"), "w")
    # htmlfile.write(htmlString)
    # htmlfile.close()


def main(gt_xyz_list, gt_verts_list, pred_xyz_list, pred_verts_list, output_dir):
    num_sample = gt_xyz_list.shape[0]

    # init eval utils
    eval_xyz, eval_xyz_aligned = EvalUtil(), EvalUtil()
    eval_mesh_err, eval_mesh_err_aligned = EvalUtil(num_kp=778), EvalUtil(num_kp=778)
    f_score, f_score_aligned = list(), list()
    f_threshs = [0.005, 0.015]

    shape_is_mano = None
    # iterate over the dataset once
    for idx in tqdm(range(num_sample)):

        xyz, verts = gt_xyz_list[idx], gt_verts_list[idx]
        xyz, verts = [np.array(x) for x in [xyz, verts]]

        xyz_pred, verts_pred = pred_xyz_list[idx], pred_verts_list[idx]
        xyz_pred, verts_pred = [np.array(x) for x in [xyz_pred, verts_pred]]

        # Not aligned errors
        eval_xyz.feed(xyz, np.ones_like(xyz[:, 0]), xyz_pred)

        if shape_is_mano is None:
            if verts_pred.shape[0] == verts.shape[0]:
                shape_is_mano = True
            else:
                shape_is_mano = False

        if shape_is_mano:
            eval_mesh_err.feed(verts, np.ones_like(verts[:, 0]), verts_pred)

        # align predictions
        xyz_pred_aligned = align_w_scale(xyz, xyz_pred)
        if shape_is_mano:
            verts_pred_aligned = align_w_scale(verts, verts_pred)
        else:
            # use trafo estimated from keypoints
            trafo = align_w_scale(xyz, xyz_pred, return_trafo=True)
            verts_pred_aligned = align_by_trafo(verts_pred, trafo)

        # Aligned errors
        eval_xyz_aligned.feed(xyz, np.ones_like(xyz[:, 0]), xyz_pred_aligned)

        if shape_is_mano:
            eval_mesh_err_aligned.feed(verts, np.ones_like(verts[:, 0]), verts_pred_aligned)

        # F-scores
        l, la = list(), list()
        for t in f_threshs:
            # for each threshold calculate the f score and the f score of the aligned vertices
            f, _, _ = calculate_fscore(verts, verts_pred, t)
            l.append(f)
            f, _, _ = calculate_fscore(verts, verts_pred_aligned, t)
            la.append(f)
        f_score.append(l)
        f_score_aligned.append(la)

    # Calculate results
    xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

    xyz_al_mean3d, _, xyz_al_auc3d, pck_xyz_al, thresh_xyz_al = eval_xyz_aligned.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP ALIGNED results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm\n' % (xyz_al_auc3d, xyz_al_mean3d * 100.0))

    if shape_is_mano:
        mesh_mean3d, _, mesh_auc3d, pck_mesh, thresh_mesh = eval_mesh_err.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D MESH results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (mesh_auc3d, mesh_mean3d * 100.0))

        mesh_al_mean3d, _, mesh_al_auc3d, pck_mesh_al, thresh_mesh_al = eval_mesh_err_aligned.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D MESH ALIGNED results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm\n' % (mesh_al_auc3d, mesh_al_mean3d * 100.0))
    else:
        mesh_mean3d, mesh_auc3d, mesh_al_mean3d, mesh_al_auc3d = -1.0, -1.0, -1.0, -1.0

        pck_mesh, thresh_mesh = np.array([-1.0, -1.0]), np.array([0.0, 1.0])
        pck_mesh_al, thresh_mesh_al = np.array([-1.0, -1.0]), np.array([0.0, 1.0])

    print('F-scores')
    f_out = list()
    f_score, f_score_aligned = np.array(f_score).T, np.array(f_score_aligned).T
    for f, fa, t in zip(f_score, f_score_aligned, f_threshs):
        print('F@%.1fmm = %.3f' % (t * 1000, f.mean()), '\tF_aligned@%.1fmm = %.3f' % (t * 1000, fa.mean()))
        f_out.append('f_score_%d: %f' % (round(t * 1000), f.mean()))
        f_out.append('f_al_score_%d: %f' % (round(t * 1000), fa.mean()))

    # Dump results
    score_path = os.path.join(output_dir, 'scores.txt')
    with open(score_path, 'w') as fo:
        xyz_mean3d *= 100
        xyz_al_mean3d *= 100
        fo.write('xyz_mean3d: %f\n' % xyz_mean3d)
        fo.write('xyz_auc3d: %f\n' % xyz_auc3d)
        fo.write('xyz_al_mean3d: %f\n' % xyz_al_mean3d)
        fo.write('xyz_al_auc3d: %f\n' % xyz_al_auc3d)

        mesh_mean3d *= 100
        mesh_al_mean3d *= 100
        fo.write('mesh_mean3d: %f\n' % mesh_mean3d)
        fo.write('mesh_auc3d: %f\n' % mesh_auc3d)
        fo.write('mesh_al_mean3d: %f\n' % mesh_al_mean3d)
        fo.write('mesh_al_auc3d: %f\n' % mesh_al_auc3d)
        for t in f_out:
            fo.write('%s\n' % t)
    print('Scores written to: %s' % score_path)

    # scale to cm
    thresh_xyz *= 100.0
    thresh_xyz_al *= 100.0
    thresh_mesh *= 100.0
    thresh_mesh_al *= 100.0

    createHTML(output_dir, [
        curve(thresh_xyz, pck_xyz, 'Distance in cm', 'Percentage of correct keypoints', 'PCK curve for keypoint error'),
        curve(thresh_xyz_al, pck_xyz_al, 'Distance in cm', 'Percentage of correct keypoints', 'PCK curve for aligned keypoint error'),
        curve(thresh_mesh, pck_mesh, 'Distance in cm', 'Percentage of correct vertices', 'PCV curve for mesh error'),
        curve(thresh_mesh_al, pck_mesh_al, 'Distance in cm', 'Percentage of correct vertices', 'PCV curve for aligned mesh error')
    ])

    pck_curve_data = {
        'xyz': [thresh_xyz.tolist(), pck_xyz.tolist()],
        'xyz_al': [thresh_xyz_al.tolist(), pck_xyz_al.tolist()],
        'mesh': [thresh_mesh.tolist(), pck_mesh.tolist()],
        'mesh_al': [thresh_mesh_al.tolist(), pck_mesh_al.tolist()],
    }
    with open(os.path.join(output_dir, 'pck_data.json'), 'w') as fo:
        json.dump(pck_curve_data, fo)

    print('Evaluation complete.')



def compute_hand_metric(model, mng: Manager):
    torch.cuda.empty_cache()
    model.eval()

    gt_xyz_list, gt_verts_list, pred_xyz_list, pred_verts_list = [], [], [], []
    
    with torch.no_grad():
        split = "test"
        # for split in ["val", "test"]:
        if split not in mng.dataloader:
            return
        # Initialize loss and metric statuses
        mng.reset_loss_status()
        mng.reset_metric_status(split)
        # Use tqdm for progress bar
        t = tqdm(total=len(mng.dataloader[split]))
        for batch_idx, batch_input in enumerate(mng.dataloader[split]):
            # Move data to GPU if available
            batch_input = tool.tensor_gpu(batch_input)
            
            # Compute model output
            batch_output = model(batch_input)
            
            # Get real batch size
            if "img" in batch_input:
                batch_size = batch_input["img"].size()[0]
            elif "img_0" in batch_input:
                batch_size = batch_input["img_0"].size()[0]
            else:
                batch_size = mng.cfg.test.batch_size

            batch_input = tool.tensor_gpu(batch_input, check_on=False)
            batch_input = [{k: v[bid] for k, v in batch_input.items()} for bid in range(batch_size)]

            batch_output = tool.tensor_gpu(batch_output, check_on=False)
            batch_output = [{k: v[bid] for k, v in batch_output.items()} for bid in range(batch_size)]
            
            pred_lst = mng.dataset[split].get_predictions(batch_input, batch_output)
            for b in range(batch_size):  # for different sample
                pred_xyz_list.append(pred_lst[0][b])
                pred_verts_list.append(pred_lst[1][b])
                gt_xyz_list.append(pred_lst[2][b])
                gt_verts_list.append(pred_lst[3][b])

            # Tqdm settings
            t.set_description(desc="")
            t.update()
        t.close()
    
        gt_xyz_list = np.stack(gt_xyz_list, axis=0)
        gt_verts_list = np.stack(gt_verts_list, axis=0)
        pred_xyz_list = np.stack(pred_xyz_list, axis=0)
        pred_verts_list = np.stack(pred_verts_list, axis=0)
        
        output_dir = os.path.join(cfg.base.model_dir, "all")
        os.makedirs(output_dir, exist_ok=True)
        main(gt_xyz_list, gt_verts_list, pred_xyz_list, pred_verts_list, output_dir)

def pred_1f(cfg):
    # Set rank and is_master flag
    cfg.base.only_weights = False
    # Set the logger
    logger = tool.set_logger(os.path.join(cfg.base.model_dir, "test.log"))
    # Print GPU ids
    # gpu_ids = ", ".join(str(i) for i in [j for j in range(cfg.base.num_gpu)])
    # logger.info("Using GPU ids: [{}]".format(gpu_ids))
    # Fetch dataloader
    cfg.data.eval_type = ["test"]
    dl, ds = fetch_test_dataloader(cfg)
    # Fetch model
    model = fetch_model(cfg.model.name, cfg)
    
    if cfg.base.cuda:
        num_gpu = torch.cuda.device_count()
        if num_gpu > 0:
            torch.cuda.set_device(0)

        model = model.cuda()
        model = DP(model)
    
    # Initialize manager
    mng = Manager(model=model, optimizer=None, scheduler=None, cfg=cfg, dataloader=dl, dataset=ds, logger=logger)
    # Test the model
    mng.logger.info("Starting test.")
    
    # Load weights from restore_file if specified
    if mng.cfg.base.resume is not None:
        mng.load_ckpt()
    elif mng.cfg.base.resume_cls is not None and mng.cfg.base.resume_h is not None and mng.cfg.base.resume_ho is not None:
        load_two_in_one_cls_weights(model, mng.logger, resume_h=mng.cfg.base.resume_h, resume_ho=mng.cfg.base.resume_ho, resume_cls=mng.cfg.base.resume_cls)
    else:
        raise NotImplementedError
        
    compute_hand_metric(model, mng)


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "cfg.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    cfg = Config(json_path).cfg
    
    # Update args into cfg.base
    cfg.base.update(vars(args))
    
    # Use GPU if available
    cfg.base.cuda = torch.cuda.is_available()
    if cfg.base.cuda:
        cfg.base.num_gpu = torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True
    # Main function
    pred_1f(cfg)
