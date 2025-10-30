import numpy as np
from PIL import Image
import cv2
import random


def transform_coords(pts, affine_trans):
    # if (pts == -1).all():
    #     return pts
    
    hom2d = np.concatenate([pts, np.ones([np.array(pts).shape[0], 1])], 1)
    transformed_rows = affine_trans.dot(hom2d.transpose()).transpose()[:, :2]
    return transformed_rows


def transform_img(img, affine_trans, res):
    trans = np.linalg.inv(affine_trans)
    img = img.transform(tuple(res), Image.AFFINE, (trans[0, 0], trans[0, 1], trans[0, 2],
                                                   trans[1, 0], trans[1, 1], trans[1, 2]))  # default is Resampling.NEAREST
    return img


# def transform_np_img_flip_first(img_np, affinetrans, do_flip, out_shape):
#     # first flip the image content
#     if do_flip:
#         img_np = img_np[:, ::-1]
        
#     # convert to PIL.Image
#     # img_pil = Image.fromarray(img_np)
#     img_pil = Image.fromarray(np.uint8(img_np))
#     img_pil = transform_img(img_pil, affinetrans, out_shape)
#     # img_pil = img_pil.crop((0, 0, out_shape[0], out_shape[1]))
#     # convert back to numpy array
#     # img_np = np.array(img_pil)
#     img_np = np.array(img_pil, np.uint8, copy=True)
    
#     return img_np


def get_affine_transform(center, scale, res, rot=0, K=None):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    origin_rot_center = rot_mat.dot(center.tolist() + [1, ])[:2]
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, res)
    total_trans = post_rot_trans.dot(rot_mat)
    if K is not None:
        t_mat = np.eye(3)
        t_mat[0, 2] = -K[0, 2]
        t_mat[1, 2] = -K[1, 2]
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        transformed_center = t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [1, ])
        affinetrans_post_rot = get_affine_trans_no_rot(transformed_center[:2], scale, res)
        return total_trans.astype(np.float32), affinetrans_post_rot.astype(np.float32), rot_mat.astype(np.float32)
    else:
        return total_trans.astype(np.float32), rot_mat.astype(np.float32)
    
    
# def get_affine_transform_arbitary_ratio(center, scale, res, rot=0, K=None):
#     rot_mat = np.zeros((3, 3))
#     sn, cs = np.sin(rot), np.cos(rot)
#     rot_mat[0, :2] = [cs, -sn]
#     rot_mat[1, :2] = [sn, cs]
#     rot_mat[2, 2] = 1
#     origin_rot_center = rot_mat.dot(center.tolist() + [1, ])[:2]
#     post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, res)
#     total_trans = post_rot_trans.dot(rot_mat)
#     if K is not None:
#         t_mat = np.eye(3)
#         t_mat[0, 2] = -K[0, 2]
#         t_mat[1, 2] = -K[1, 2]
#         t_inv = t_mat.copy()
#         t_inv[:2, 2] *= -1
#         transformed_center = t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [1, ])
#         affinetrans_post_rot = get_affine_trans_no_rot(transformed_center[:2], scale, res)
#         return total_trans.astype(np.float32), affinetrans_post_rot.astype(np.float32), rot_mat.astype(np.float32)
#     else:
#         return total_trans.astype(np.float32), rot_mat.astype(np.float32)


def get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    if type(scale) is list:
        affinet[0, 0] = float(res[0]) / scale[0]
        affinet[1, 1] = float(res[1]) / scale[1]
        affinet[0, 2] = res[1] * (-float(center[0]) / scale[0] + .5)
        affinet[1, 2] = res[0] * (-float(center[1]) / scale[1] + .5)
    else:
        affinet[0, 0] = float(res[0]) / scale
        affinet[1, 1] = float(res[1]) / scale
        affinet[0, 2] = res[1] * (-float(center[0]) / scale + .5)
        affinet[1, 2] = res[0] * (-float(center[1]) / scale + .5)
    affinet[2, 2] = 1
    return affinet


def rotation_angle(angle, rot_mat, coord_change_mat=None):
    per_rdg, _ = cv2.Rodrigues(angle)
    if coord_change_mat is not None:
        rot_mat = np.dot(rot_mat, coord_change_mat)
    resrot, _ = cv2.Rodrigues(np.dot(rot_mat, per_rdg))
    return resrot[:, 0].astype(np.float32)


def get_bbox_joints(joints2d, bbox_factor=1.1):
    min_x, min_y = joints2d.min(0)
    max_x, max_y = joints2d.max(0)
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    center = np.asarray([c_x, c_y])
    bbox_delta_x = (max_x - min_x) * bbox_factor / 2
    bbox_delta_y = (max_y - min_y) * bbox_factor / 2
    bbox_delta = np.asarray([bbox_delta_x, bbox_delta_y])
    bbox = np.array([*(center - bbox_delta), *(center + bbox_delta)], dtype=np.float32)
    return bbox


def dilate_bbox(input_bbox, bbox_factor=1.1):
    min_x, min_y = input_bbox[0], input_bbox[1]
    max_x, max_y = input_bbox[2], input_bbox[3]
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    center = np.asarray([c_x, c_y])
    bbox_delta_x = (max_x - min_x) * bbox_factor / 2
    bbox_delta_y = (max_y - min_y) * bbox_factor / 2
    bbox_delta = np.asarray([bbox_delta_x, bbox_delta_y])
    bbox = np.array([*(center - bbox_delta), *(center + bbox_delta)], dtype=np.float32)
    return bbox


def normalize_joints(joints2d, bbox):
    # if (joints2d == -1).all():
    #     return joints2d
    
    bbox = bbox.reshape(2, 2)
    # if (bbox[1, :] - bbox[0, :]).any() == 0:
    #     import ipdb; ipdb.set_trace()
    joints2d = (joints2d - bbox[0, :]) / (bbox[1, :] - bbox[0, :])
    return joints2d


def denormalize_joints(normed_joints2d, bbox):
    bbox = bbox.reshape(2, 2)
    joints2d = normed_joints2d * (bbox[1, :] - bbox[0, :]) + bbox[0, :]
    return joints2d


def recover_joints(joints2d, bbox):
    bbox = bbox.reshape(2, 2)
    joints2d = joints2d * (bbox[1, :] - bbox[0, :]) + bbox[0, :]
    return joints2d


def get_mask_ROI(mask, bbox):
    mask = mask.crop(bbox)
    return mask


def get_color_params(brightness=0, contrast=0, saturation=0, hue=0):
    if brightness > 0:
        brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
    else:
        brightness_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
    else:
        saturation_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return brightness_factor, contrast_factor, saturation_factor, hue_factor


def color_jitter(img, brightness=0, contrast=0, saturation=0, hue=0):
    import torchvision
    brightness, contrast, saturation, hue = get_color_params(brightness=brightness,
                                                             contrast=contrast,
                                                             saturation=saturation,
                                                             hue=hue)

    # Create img transform function sequence
    img_transforms = []
    if brightness is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
    if saturation is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
    if hue is not None:
        img_transforms.append(
            lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
    if contrast is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
    random.shuffle(img_transforms)

    jittered_img = img
    for func in img_transforms:
        jittered_img = func(jittered_img)
    return jittered_img

# get random color_jitter transforms
def color_jitter_transforms(brightness=0, contrast=0, saturation=0, hue=0):
    import torchvision
    brightness, contrast, saturation, hue = get_color_params(brightness=brightness,
                                                             contrast=contrast,
                                                             saturation=saturation,
                                                             hue=hue)

    # Create img transform function sequence
    img_transforms = []
    if brightness is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
    if saturation is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
    if hue is not None:
        img_transforms.append(
            lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
    if contrast is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
    random.shuffle(img_transforms)
    
    return img_transforms

def color_jitter_apply(img, img_transforms):
    jittered_img = img
    for func in img_transforms:
        jittered_img = func(jittered_img)
    return jittered_img

def get_bbox21_3d_from_dict(vertex):
    bbox21_3d = {}
    for key in vertex:
        vp = vertex[key][:]
        x = vp[:, 0].reshape((1, -1))
        y = vp[:, 1].reshape((1, -1))
        z = vp[:, 2].reshape((1, -1))
        x_max, x_min, y_max, y_min, z_max, z_min = np.max(x), np.min(x), np.max(y), np.min(y), np.max(z), np.min(z)
        p_blb = np.array([x_min, y_min, z_min])  # bottem, left, behind
        p_brb = np.array([x_max, y_min, z_min])  # bottem, right, behind
        p_blf = np.array([x_min, y_max, z_min])  # bottem, left, front
        p_brf = np.array([x_max, y_max, z_min])  # bottem, right, front
        #
        p_tlb = np.array([x_min, y_min, z_max])  # top, left, behind
        p_trb = np.array([x_max, y_min, z_max])  # top, right, behind
        p_tlf = np.array([x_min, y_max, z_max])  # top, left, front
        p_trf = np.array([x_max, y_max, z_max])  # top, right, front
        #
        p_center = (p_tlb + p_brf) / 2
        #
        p_ble = (p_blb + p_blf) / 2  # bottem, left, edge center
        p_bre = (p_brb + p_brf) / 2  # bottem, right, edge center
        p_bfe = (p_blf + p_brf) / 2  # bottem, front, edge center
        p_bbe = (p_blb + p_brb) / 2  # bottem, behind, edge center
        #
        p_tle = (p_tlb + p_tlf) / 2  # top, left, edge center
        p_tre = (p_trb + p_trf) / 2  # top, right, edge center
        p_tfe = (p_tlf + p_trf) / 2  # top, front, edge center
        p_tbe = (p_tlb + p_trb) / 2  # top, behind, edge center
        #
        p_lfe = (p_tlf + p_blf) / 2  # left, front, edge center
        p_lbe = (p_tlb + p_blb) / 2  # left, behind, edge center
        p_rfe = (p_trf + p_brf) / 2  # left, front, edge center
        p_rbe = (p_trb + p_brb) / 2  # left, behind, edge center

        pts = np.stack((p_blb, p_brb, p_blf, p_brf,
                        p_tlb, p_trb, p_tlf, p_trf,
                        p_ble, p_bre, p_bfe, p_bbe,
                        p_tle, p_tre, p_tfe, p_tbe,
                        p_lfe, p_lbe, p_rfe, p_rbe,
                        p_center))
        bbox21_3d[key] = pts
    return bbox21_3d

def get_bbox8_3d_from_dict(vertex):
    bbox8_3d = {}
    for key in vertex:
        vp = vertex[key][:]
        x = vp[:, 0].reshape((1, -1))
        y = vp[:, 1].reshape((1, -1))
        z = vp[:, 2].reshape((1, -1))
        x_max, x_min, y_max, y_min, z_max, z_min = np.max(x), np.min(x), np.max(y), np.min(y), np.max(z), np.min(z)
        p_blb = np.array([x_min, y_min, z_min])  # bottem, left, behind
        p_brb = np.array([x_max, y_min, z_min])  # bottem, right, behind
        p_blf = np.array([x_min, y_max, z_min])  # bottem, left, front
        p_brf = np.array([x_max, y_max, z_min])  # bottem, right, front
        #
        p_tlb = np.array([x_min, y_min, z_max])  # top, left, behind
        p_trb = np.array([x_max, y_min, z_max])  # top, right, behind
        p_tlf = np.array([x_min, y_max, z_max])  # top, left, front
        p_trf = np.array([x_max, y_max, z_max])  # top, right, front
        #

        pts = np.stack((p_blb, p_brb, p_blf, p_brf,
                        p_tlb, p_trb, p_tlf, p_trf))
        bbox8_3d[key] = pts
    return bbox8_3d

def get_diameter(vertex):
    diameters = {}
    for key in vertex:
        vp = vertex[key][:]
        x = vp[:, 0].reshape((1, -1))
        y = vp[:, 1].reshape((1, -1))
        z = vp[:, 2].reshape((1, -1))
        x_max, x_min, y_max, y_min, z_max, z_min = np.max(x), np.min(x), np.max(y), np.min(y), np.max(z), np.min(z)
        diameter_x = abs(x_max - x_min)
        diameter_y = abs(y_max - y_min)
        diameter_z = abs(z_max - z_min)
        diameters[key] = np.sqrt(diameter_x ** 2 + diameter_y ** 2 + diameter_z ** 2)
    return diameters


def fuse_bbox(bbox_1, bbox_2, img_shape, scale_factor=1.):
    bbox = np.concatenate((bbox_1.reshape(2, 2), bbox_2.reshape(2, 2)), axis=0)
    min_x, min_y = bbox.min(0)
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = bbox.max(0)
    max_x, max_y = min(max_x, img_shape[0]), min(max_y, img_shape[1])
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    center = np.asarray([c_x, c_y])
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    max_delta = max(delta_x, delta_y)
    scale = max_delta * scale_factor
    return center, scale
    
    # return center, scale, np.asarray([min_x, min_y, delta_x, delta_y])

def fuse_bbox_range(bbox_1, bbox_2, img_shape):
    bbox = np.concatenate((bbox_1.reshape(2, 2), bbox_2.reshape(2, 2)), axis=0)
    min_x, min_y = bbox.min(0)
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = bbox.max(0)
    max_x, max_y = min(max_x, img_shape[0]), min(max_y, img_shape[1])
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    # max_delta = max(delta_x, delta_y)
    max_delta = int(max(delta_x, delta_y))
    min_x = int(min_x)
    min_y = int(min_y)
    return [min_x, min_y, min_x + max_delta, min_y + max_delta, max_delta]

def fuse_bbox_min_delta(bbox_1, bbox_2, img_shape, scale_factor=1.):
    bbox = np.concatenate((bbox_1.reshape(2, 2), bbox_2.reshape(2, 2)), axis=0)
    min_x, min_y = bbox.min(0)
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = bbox.max(0)
    max_x, max_y = min(max_x, img_shape[0]), min(max_y, img_shape[1])
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    center = np.asarray([c_x, c_y])
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    min_delta = min(delta_x, delta_y)
    scale = min_delta * scale_factor
    return center, scale

    # return center, scale, np.asarray([min_x, min_y, delta_x, delta_y])


def fuse_bbox_arbitary_ratio(bbox_1, bbox_2, img_shape, scale_factor=1.):
    bbox = np.concatenate((bbox_1.reshape(2, 2), bbox_2.reshape(2, 2)), axis=0)
    min_x, min_y = bbox.min(0)
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = bbox.max(0)
    max_x, max_y = min(max_x, img_shape[0]), min(max_y, img_shape[1])
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    center = np.asarray([c_x, c_y])
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    return center, [delta_x*scale_factor, delta_y*scale_factor]
    
    # return center, scale, np.asarray([min_x, min_y, delta_x, delta_y])

def bbox_center_scale(bbox, img_shape, scale_factor=1.):
    min_x, min_y = max(0, bbox[0]), max(0, bbox[1])
    max_x, max_y = min(bbox[2], img_shape[0]), min(bbox[3], img_shape[1])
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    center = np.asarray([c_x, c_y])
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    max_delta = max(delta_x, delta_y)
    scale = max_delta * scale_factor
    return center, scale

def get_unseen_test_object():
    # unseen objects appear in evaluation set but not in training set
    return ['019_pitcher_base']


def no_black_edge(bbox, img_height, img_width):
    # offset bbox if it has black edge, bbox: x, y, w, h
    if bbox[0] < 0:
        bbox[0] = 0.
    elif (bbox[0] + bbox[2]) > img_width:
        offset_x = bbox[0] + bbox[2] - (img_width - 1)
        bbox[0] -= offset_x

    if bbox[1] < 0:
        bbox[1] = 0.
    elif (bbox[1] + bbox[3]) > img_height:
        offset_y = bbox[1] + bbox[3] - (img_height - 1)
        bbox[1] -= offset_y
    return bbox