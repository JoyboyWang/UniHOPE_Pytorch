import numpy as np
import os
import cv2
import torch
# object evaluation metric
# hand evaluation metric: https://github.com/shreyashampali/ho3d/blob/master/eval.py

from collections import defaultdict
from scipy import spatial
from pytorch3d.loss import chamfer

from model.kypt_transformer.common.utils.transforms import rot_param_rot_mat, rot_param_rot_mat_np

def vertices_reprojection(vertices, rt, k):
    p = np.matmul(k, np.matmul(rt[:3, 0:3], vertices.T) + rt[:3, 3].reshape(-1, 1))
    p[0] = p[0] / (p[2] + 1e-5)
    p[1] = p[1] / (p[2] + 1e-5)
    return p[:2].T

def compute_ADD_s_error(pred_pose, gt_pose, obj_mesh):
    #rot_dir_z = gt_pose[:3, 2] * np.pi  # N x 3
    #flipped_obj_rot_z = np.matmul(cv2.Rodrigues(rot_dir_z)[0].squeeze(),
    #                                        gt_pose[:3, 0:3])  # 3 x 3 # flipped rot
    N = obj_mesh.shape[0]
                                            
    add_gt = np.matmul(gt_pose[:3, 0:3], obj_mesh.T) + gt_pose[:3, 3].reshape(-1, 1)  # (3,N)
    add_gt = torch.from_numpy(add_gt.T).cuda()
    #add_gt = add_gt[None,:,:].repeat(N,1)
    add_gt = add_gt.unsqueeze(0).repeat(N,1,1)
    #add_gt_flip = np.matmul(flipped_obj_rot_z, obj_mesh.T) + gt_pose[:3, 3].reshape(-1, 1)  # (3,N)
    add_pred = np.matmul(pred_pose[:3, 0:3], obj_mesh.T) + pred_pose[:3, 3].reshape(-1, 1)
    add_pred = torch.from_numpy(add_pred.T).cuda()
    add_pred = add_pred.unsqueeze(1).repeat(1,N,1)
    #is_rot_sym_objs_z = cls in [6,21,10] #mustard, bleach, potted meat
    dis = torch.norm(add_gt - add_pred, dim=2)
    add_bias = torch.mean(torch.min(dis,dim=1)[0])
    add_bias = add_bias.detach().cpu().numpy()
    #add_bias = min(np.mean(np.linalg.norm(add_gt - add_pred, axis=0), axis=0),np.mean(np.linalg.norm(add_gt_flip - add_pred, axis=0), axis=0))
    #add_bias = np.mean(np.linalg.norm(add_gt - add_pred, axis=0), axis=0)
    return add_bias

def compute_REP_error(pred_pose, gt_pose, intrinsics, obj_mesh):
    reproj_pred = vertices_reprojection(obj_mesh, pred_pose, intrinsics)
    reproj_gt = vertices_reprojection(obj_mesh, gt_pose, intrinsics)
    reproj_diff = np.abs(reproj_gt - reproj_pred)
    reproj_bias = np.mean(np.linalg.norm(reproj_diff, axis=1), axis=0)
    return reproj_bias


def compute_ADD_error(pred_pose, gt_pose, obj_mesh):
    add_gt = np.matmul(gt_pose[:3, 0:3], obj_mesh.T) + gt_pose[:3, 3].reshape(-1, 1)  # (3,N) 
    add_pred = np.matmul(pred_pose[:3, 0:3], obj_mesh.T) + pred_pose[:3, 3].reshape(-1, 1)
    add_bias = np.mean(np.linalg.norm(add_gt - add_pred, axis=0), axis=0)
    return add_bias


def fuse_test(output, width, height, intrinsics, bestCnt, bbox_3d, cord_upleft, affinetrans=None,do_flip=None):
    predx = output[0]
    predy = output[1]
    det_confs = output[2]
    keypoints = bbox_3d
    nH, nW, nV = predx.shape

    xs = predx.reshape(nH * nW, -1) * width
    ys = predy.reshape(nH * nW, -1) * height
    det_confs = det_confs.reshape(nH * nW, -1)
    gridCnt = len(xs)

    p2d = None
    p3d = None
    candiBestCnt = min(gridCnt, bestCnt)
    for i in range(candiBestCnt):
        bestGrids = det_confs.argmax(axis=0) # choose best N count
        validmask = (det_confs[bestGrids, list(range(nV))] > 0.5)
        xsb = xs[bestGrids, list(range(nV))][validmask]
        ysb = ys[bestGrids, list(range(nV))][validmask]
        t2d = np.concatenate((xsb.reshape(-1, 1), ysb.reshape(-1, 1)), 1)
        t3d = keypoints[validmask]
        if p2d is None:
            p2d = t2d
            p3d = t3d
        else:
            p2d = np.concatenate((p2d, t2d), 0)
            p3d = np.concatenate((p3d, t3d), 0)
        det_confs[bestGrids, list(range(nV))] = 0

    if len(p3d) < 6:
        R = np.eye(3)
        T = np.array([0, 0, 1]).reshape(-1, 1)
        rt = np.concatenate((R, T), 1)
        return rt, p2d

    p2d[:, 0] += cord_upleft[0]
    p2d[:, 1] += cord_upleft[1]
    if affinetrans is not None:
        affinetrans = np.linalg.inv(affinetrans)
        homp2d = np.concatenate([p2d, np.ones([np.array(p2d).shape[0], 1])], 1)
        p2d = affinetrans.dot(homp2d.transpose()).transpose()[:, :2]
    if do_flip:  # why?
        p2d[:,0] = np.array(640,dtype=np.float32)  - p2d[:,0] - 1
    retval, rot, trans, inliers = cv2.solvePnPRansac(p3d, p2d, intrinsics, None, flags=cv2.SOLVEPNP_EPNP)
    if not retval:
        R = np.eye(3)
        T = np.array([0, 0, 1]).reshape(-1, 1)
    else:
        R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
        T = trans.reshape(-1, 1)
    rt = np.concatenate((R, T), 1)
    return rt, p2d


def eval_batch_obj(batch_input, batch_output, 
                mesh_dict,
                REP_res_dic, ADD_res_dic, obj_eval_mode="gt", bestCnt=10):
    # bestCnt: choose best N count for fusion
    bs = len(batch_output)
    for i in range(bs):
        input = batch_input[i]
        output = batch_output[i]
        
        if obj_eval_mode == "gt":
            eval_obj = bool(input["gt_grasping"])
        elif obj_eval_mode == "pred-gt":
            eval_obj = bool(input["gt_grasping"]) & bool(output["pred_grasping"])
        elif obj_eval_mode == "all":
            eval_obj = True
        else:
            raise NotImplementedError
        
        if eval_obj is False:  # NOTE: only when grasping the object
            continue
        
        obj_output = [output["preds_obj_0"], output["preds_obj_1"], output["preds_obj_2"]]
        bbox = input["bbox_obj"]
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cord_upleft = [bbox[0], bbox[1]]
        
        intrinsics = input["cam_intr"]
        
        bbox_3d = input["obj_bbox3d"]
        # if torch.is_tensor(input["obj_cls"]):
        #     cls = int(input["obj_cls"])
        # else:
        #     cls = input["obj_cls"].lstrip()
        # import ipdb; ipdb.set_trace()
        
        if np.isscalar(input["obj_cls"]):  # for DexYCB .dtype == np.float32
            cls = int(input["obj_cls"])
        else:  # for HO3D
            cls = bytes(input["obj_cls"].astype(np.uint8).tolist()).decode().lstrip()
            # print(cls)
            # cls = bytes(input["obj_cls"].tobytes()).decode().lstrip()
        
        mesh = mesh_dict[cls]
        do_flip=input["do_flip"]
        if "affinetrans" in input:
            affinetrans = input["affinetrans"]
        else:
            affinetrans = None
        pred_pose, p2d = fuse_test(obj_output, width, height, intrinsics, bestCnt, bbox_3d, cord_upleft,
                                   affinetrans=affinetrans,do_flip=do_flip)
        
        obj_pose = input["obj_pose"]
        # calculate REP and ADD error
        REP_error = compute_REP_error(pred_pose, obj_pose, intrinsics, mesh)
        if cls in [13,16,20,21]: # or cls in ['024_bowl', '036_wood_block', '052_extra_large_clamp', '061_foam_brick'] # this is the same as HFL-Net
            ADD_error = compute_ADD_s_error(pred_pose, obj_pose, mesh)
        else:
            ADD_error = compute_ADD_error(pred_pose, obj_pose, mesh)
        REP_res_dic[cls].append(REP_error)
        ADD_res_dic[cls].append(ADD_error)
    return REP_res_dic, ADD_res_dic


def eval_batch_obj_kypt(batch_input, batch_output, 
                mesh_dict,
                REP_res_dic, ADD_res_dic, obj_eval_mode="gt", bestCnt=10):
    # bestCnt: choose best N count for fusion
    bs = len(batch_output)
    for i in range(bs):
        input = batch_input[i]
        output = batch_output[i]
        
        if obj_eval_mode == "gt":
            eval_obj = bool(input["gt_grasping"])
        elif obj_eval_mode == "pred-gt":
            eval_obj = bool(input["gt_grasping"]) & bool(output["pred_grasping"])
        elif obj_eval_mode == "all":
            eval_obj = True
        else:
            raise NotImplementedError
        
        if eval_obj is False:  # NOTE: only when grasping the object
            continue
        
        intrinsics = input["cam_intr"]
        
        if np.isscalar(input["obj_cls"]):  # for DexYCB .dtype == np.float32
            cls = int(input["obj_cls"])
        else:  # for HO3D, string
            cls = bytes(input["obj_cls"].astype(np.uint8).tolist()).decode().lstrip()
        
        mesh = mesh_dict[cls]
        # do_flip=input["do_flip"]
        
        # import ipdb; ipdb.set_trace()
        joints_right = output['pred_joints3d_cam'] # already in m
        # joints_right = output['gt_joints3d_cam'] # # already in m
        
        pred_obj_trans = output['obj_trans_out'][:, None]
        # import ipdb; ipdb.set_trace()
        # for ho3d, use bug version. For DexYCB, use absolute version. Never use relative version to compare metric
        # NOTE: during training the object trans is relative to mano origin. during testing the mano origin is shifted to wrist joint.
        # pred_obj_trans = pred_obj_trans + joints_right[0][:, None] # absolute trans
        pred_obj_trans = pred_obj_trans - joints_right[0][:, None]  # origianl, seem to have bug, but best. 
        # pred_obj_trans = pred_obj_trans  # relative trans
        
        pred_obj_rot = rot_param_rot_mat_np(output['obj_rot_out'][None])[0] # 3x3 rotation matrix
        pred_pose = np.concatenate((pred_obj_rot, pred_obj_trans), axis=1)
        
        gt_obj_trans = input['rel_obj_trans'][:, None]  # relative trans
        # gt_obj_trans = input['obj_trans'][:, None]  # absollute translation
        # gt_obj_trans = input['rel_obj_trans'][:, None] + input['joint_cam'][0] / 1000 # absolute trans
        
        gt_obj_rot = cv2.Rodrigues(input['obj_rot'])[0]
        gt_pose = np.concatenate((gt_obj_rot, gt_obj_trans), axis=1)
        # gt_pose = input["obj_pose"]
        
        # calculate REP and ADD error
        REP_error = compute_REP_error(pred_pose, gt_pose, intrinsics, mesh)
        if cls in [13,16,20,21]: # or cls in ['024_bowl', '036_wood_block', '052_extra_large_clamp', '061_foam_brick'] # this is the same as HFL-Net
            ADD_error = compute_ADD_s_error(pred_pose, gt_pose, mesh)
        else:
            ADD_error = compute_ADD_error(pred_pose, gt_pose, mesh)
        REP_res_dic[cls].append(REP_error)
        ADD_res_dic[cls].append(ADD_error)
    return REP_res_dic, ADD_res_dic


def eval_object_pose(REP_res_dic, ADD_res_dic, diameter_dic, unseen_objects=[], epoch=None, rep_threshold=5, add_threshold=0.1): # outpath, 
    # REP_res_dic: key: object class, value: REP error distance
    # ADD_res_dic: key: object class, value: ADD error distance

    # # object result file
    # if not os.path.exists(outpath):
    #     os.makedirs(outpath)
    # log_path = os.path.join(outpath, "object_result.txt") if epoch is None else os.path.join(outpath, "object_result_epoch{}.txt".format(epoch))
    # log_file = open(log_path, "w+")
    # import ipdb; ipdb.set_trace()
    # REP_5 = {}
    # for k in REP_res_dic.keys():
    #     if len(REP_res_dic[k]) > 0:
    #         value = np.mean(np.array(REP_res_dic[k]) <= 5)
    #     else:
    #         value = 0
    #     # REP_5["REP_5-{}".format(k)] = value
    #     REP_5[k] = value

    # ADD_10 = {}
    # for k in ADD_res_dic.keys():
    #     if len(ADD_res_dic[k]) > 0:
    #         value = np.mean(np.array(ADD_res_dic[k]) <= 0.1 * diameter_dic[k])
    #     else:
    #         value = 0
    #     # ADD_10["ADD_10-{}".format(k)] = value
    #     ADD_10[k] = value
    # import ipdb; ipdb.set_trace()
    
    num_grasping = 0
    
    rep_sum = 0
    add_sum = 0
    
    REP_5 = {}
    for k in REP_res_dic.keys():
        num_grasping += len(REP_res_dic[k])
        
        rep_5_lst = (np.array(REP_res_dic[k]) <= rep_threshold)
        # if len(rep_5_lst) > 0:
        REP_5[k] = np.mean(rep_5_lst)  # if no element, get NaN for this object
        # else:  # avoid nan
        #     REP_5[k] = 0
        
        rep_sum += np.sum(rep_5_lst)  # if no element, return 0
    rep_sum /= num_grasping
    
        
    ADD_10 = {}
    for k in ADD_res_dic.keys():
        
        add_10_lst = (np.array(ADD_res_dic[k]) <= add_threshold * diameter_dic[k])
        # if len(add_10_lst) > 0:
        ADD_10[k] = np.mean(add_10_lst)  # if no element, get NaN for this object
        # else:  # avoid nan
        #     ADD_10[k] = 0
        
        add_sum += np.sum(add_10_lst)  # if no element, return 0
    add_sum /= num_grasping

    print("{} of grasping used as object evaluation.".format(num_grasping))
    # for k in ADD_res_dic.keys():
    #     if k in unseen_objects:
    #         REP_5.pop(k, None)
    #         ADD_10.pop(k, None)

    # write down result
    # print('REP-5', file=log_file)
    # print(REP_5, file=log_file)
    # print('ADD-10', file=log_file)
    # print(ADD_10, file=log_file)
    # log_file.close()
    return ADD_10, REP_5, add_sum, rep_sum


def eval_object_pose_no_count(REP_res_dic, ADD_res_dic, diameter_dic, unseen_objects=[], epoch=None): # outpath, 
    num_grasping = 0
    
    rep_sum = 0
    add_sum = 0
    
    REP_5 = {}
    for k in REP_res_dic.keys():
        num_grasping += len(REP_res_dic[k])
        REP_5[k] = np.nanmean(np.array(REP_res_dic[k]))  # np.mean will result in NaN on s3 split
        
        rep_sum += np.sum(np.array(REP_res_dic[k]))
    rep_sum /= num_grasping
        
    ADD_10 = {}
    for k in ADD_res_dic.keys():
        ADD_10[k] = np.nanmean(np.array(ADD_res_dic[k])) # np.mean will result in NaN on s3 split
        
        add_sum += np.sum(np.array(ADD_res_dic[k]))
    add_sum /= num_grasping

    print("{} of grasping used as object evaluation.".format(num_grasping))

    return ADD_10, REP_5, add_sum, rep_sum


def eval_hand_pose_result(hand_eval_result,outpath, epoch):
    #hand result file
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    log_path = os.path.join(outpath, "hand_result.txt") if epoch is None else os.path.join(outpath, "hand_result_epoch{}.txt".format(epoch))
    log_file = open(log_path, "w+")
    print('mpjpe', file=log_file)
    print(np.mean(np.array(hand_eval_result[0])), file=log_file)
    print('pa-mpjpe', file=log_file)
    print(np.mean(np.array(hand_eval_result[1])), file=log_file)

    log_file.close()


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s) 

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A,B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2


def eval_hand(preds_joint,gts_root_joint,gts_hand_type,gts_joints_coord_cam,hand_eval_result):
    sample_num = len(preds_joint)
    for n in range (sample_num):
        pred_joint = preds_joint[n]
        gt_hand_type = gts_hand_type[n]
        gt_root_joint = gts_root_joint[n].detach().cpu().numpy()
        gt_joints_coord_cam = gts_joints_coord_cam[n].detach().cpu().numpy()

        # root centered
        #\u5df2\u5728mano_ho3d\u5c42\u505a\u8fc7

        # flip back to left hand
        if gt_hand_type == 'left':
            pred_joint[:,0] *= -1

        # root align
        pred_joint += gt_root_joint

        # GT and rigid align
        joints_out_aligned = rigid_align(pred_joint, gt_joints_coord_cam)

        #m to mm
        pred_joint *= 1000
        joints_out_aligned *= 1000
        gt_joints_coord_cam *= 1000

            
        #[mpjpe_list, pa-mpjpe_list]
        hand_eval_result[0].append(np.sqrt(np.sum((pred_joint - gt_joints_coord_cam)**2,1)).mean())
        hand_eval_result[1].append(np.sqrt(np.sum((joints_out_aligned - gt_joints_coord_cam)**2,1)).mean())

    return hand_eval_result


def get_point_metrics(gt_points, pred_points):
    results = {}
    # results = defaultdict(list)
    # dist_mat = distutils.batch_pairwise_dist(gt_points, pred_points)
    # Checked against pytorch3d chamfer loss
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        chamfer_dists = chamfer.chamfer_distance(torch.from_numpy(gt_points).unsqueeze(0).cuda(), # 
                                                 torch.from_numpy(pred_points).unsqueeze(0).cuda(),  # 
                                                 batch_reduction=None)[0]
    results["chamfer_dists"] = chamfer_dists.item()
    
    # ADD-S from
    # https://github.com/thodan/bop_toolkit/blob/53150b649467976b4f619fbffb9efe525c7e11ca/
    # bop_toolkit_lib/pose_error.py#L164
    nn_index = spatial.cKDTree(pred_points)
    nn_dists, _ = nn_index.query(gt_points, k=1)
    # adis.append(nn_dists.mean())
    adis = nn_dists.mean()
    
    # results["add-s"].extend(adis)
    results["add-s"] = adis
    if gt_points.shape[0] == pred_points.shape[0]:  # TODO: very strange since it is always satisfied
        # Use vertex assignments
        vert_mean_dists = np.linalg.norm(gt_points - pred_points, 2, -1).mean(-1)
        results["verts_dists"] = vert_mean_dists
    else:
        # Else, repeat symmetric term
        # results["verts_dists"].extend(adis)
        results["verts_dists"]= adis
    return dict(results)


def rt_transform(vertices, rt):
    return (np.matmul(rt[:3, 0:3], vertices.T) + rt[:3, 3].reshape(-1, 1)).T
