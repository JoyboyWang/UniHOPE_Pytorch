import os
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
from common.utils.seg import *

# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh
from pyrender.shader_program import ShaderProgramCache

def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    # kp_mask = np.ascontiguousarray(img, dtype=np.uint8)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


from common.utils.transforms import cam2pixel
def render_keypoints(img, kps_3d, cam_param):
    gt_joints_out_2d = cam2pixel(kps_3d, cam_param["focal"], cam_param["princpt"])
    return vis_keypoints(img, gt_joints_out_2d)


def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)


def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

        if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1, 0] > 0:
            ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=colors[l], marker="o")
        if kpt_3d_vis[i2, 0] > 0:
            ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=colors[l], marker="o")

    if filename is None:
        ax.set_title("3D vis")
    else:
        ax.set_title(filename)

    ax.set_xlabel("X Label")
    ax.set_ylabel("Z Label")
    ax.set_zlabel("Y Label")
    ax.legend()

    plt.show()
    cv2.waitKey(0)


def save_obj(v, f, visual=None, file_name="output.obj"):
    mesh = trimesh.Trimesh(v, f, process=False)
    if visual is not None:
        mesh.visual = visual
    mesh.export(file_name)


def render_mesh(img, mesh, face, cam_param, return_mesh=False):
    # print(img.shape)  # (480, 640, 3)

    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    ori_mesh = copy.deepcopy(mesh)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)  # False
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3), bg_color=(0., 0., 0.))
    scene.add(mesh, "mesh")

    focal, princpt = cam_param["focal"], cam_param["princpt"]
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    # img = rgb
    if return_mesh:
        return img, ori_mesh
    else:
        return img, rgb
    
# work with camera intrinsics that have not been flipped
def render_mesh_w_offset(img, mesh, face, cam_param, return_mesh=False, add_offset=False):
    # print(img.shape)  # (480, 640, 3)
    height, width, _ = img.shape

    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    ori_mesh = copy.deepcopy(mesh)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)  # False
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3), bg_color=(0., 0., 0.))
    scene.add(mesh, "mesh")

    focal, princpt = cam_param["focal"], cam_param["princpt"]
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # import ipdb; ipdb.set_trace()
    if add_offset:
        # xoffset = int(2 * (320 - princpt[0]))  # 308.5481
        xoffset = 2 * (width/2 - princpt[0])  # 308.5481
        # print(xoffset)
        T = np.array([[1, 0, xoffset], [0, 1, 0]], dtype=np.float32)
        valid_mask = cv2.warpAffine(valid_mask.astype(np.uint8), T, (width, height))[:, :, None]
        rgb = cv2.warpAffine(rgb, T, (width, height))

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    # img = rgb
    if return_mesh:
        return img, ori_mesh
    else:
        return img, rgb


def render_ycb_w_offset(img, ycb_path, pose, cam_param, add_offset=False):
    mesh = trimesh.load(ycb_path)
    mesh = pyrender.Mesh.from_trimesh(mesh)

    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3), bg_color=(0., 0., 0.))
    scene.add(mesh, pose=pose)

    focal, princpt = cam_param["focal"], cam_param["princpt"]
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # import ipdb; ipdb.set_trace()
    if add_offset:
        # xoffset = int(2 * (320 - princpt[0]))  # 308.5481
        xoffset = 2 * (320 - princpt[0])  # 308.5481
        T = np.array([[1, 0, xoffset], [0, 1, 0]], dtype=np.float32)
        valid_mask = cv2.warpAffine(valid_mask.astype(np.uint8), T, (640, 480))[:, :, None]
        rgb = cv2.warpAffine(rgb, T, (640, 480))

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    # img = rgb
    return img, rgb


def render_ycb(img, mesh, pose, cam_param):
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3), bg_color=(0., 0., 0.))
    scene.add(mesh, pose=pose)

    focal, princpt = cam_param["focal"], cam_param["princpt"]
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # # light
    # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    # light_pose = np.eye(4)
    # light_pose[:3, 3] = np.array([0, -1, 1])
    # scene.add(light, pose=light_pose)
    # light_pose[:3, 3] = np.array([0, 1, 1])
    # scene.add(light, pose=light_pose)
    # light_pose[:3, 3] = np.array([1, 1, 2])
    # scene.add(light, pose=light_pose)

    # render
    depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    # rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    
    # rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # # save to image
    # img = rgb * valid_mask + img * (1 - valid_mask)
    # img = rgb
    # return img, rgb, valid_mask
    return valid_mask


def render_mesh_seg(img, mesh, face, cam_param):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    face_colors = np.zeros_like(face)
    face_colors[xmz_faces_idx] = xmz_color
    face_colors[wmz_faces_idx] = wmz_color
    face_colors[zz_faces_idx] = zz_color
    face_colors[sz_faces_idx] = sz_color
    face_colors[dmz_faces_idx] = dmz_color
    face_colors[palm_faces_idx] = palm_color

    mesh.visual.face_colors = face_colors
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene()
    scene.add(mesh, 'mesh')

    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    renderer.delete()
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    return img, valid_mask


def render_mesh_seg_wo_color(img_shape_PIL, mesh, face, cam_param):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene()
    scene.add(mesh, 'mesh')

    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img_shape_PIL[0], viewport_height=img_shape_PIL[1], point_size=1.0)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    renderer.delete()
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    return valid_mask


def vis_2d_bbox(img, bbox):
    top_left = bbox[:2].astype(np.int)
    bottom_right = bbox[2:].astype(np.int)
    
    # Define the color of the box (B, G, R) and the thickness of the box lines
    color = (0, 255, 0)  # Green color
    thickness = 2  # Thickness of the lines
    
    # bbox: [x0, y0, x1, y1]
    return cv2.rectangle(img, top_left, bottom_right, color, thickness)
    # return cv2.rectangle(img, bbox[0], bbox[1], bbox[2], bbox[3])
    
    
def render_hand_obj(img, hand_verts, hand_faces, obj_mesh, obj_pose, cam_param, add_offset=False):
    shape = img.shape
    
    # hand mesh
    hand_mesh = trimesh.Trimesh(hand_verts, hand_faces, process=False)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    hand_mesh.apply_transform(rot)
    
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    
    hand_mesh = pyrender.Mesh.from_trimesh(hand_mesh, material=material, smooth=True)

    # new scene
    scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]))

    # add hand mesh
    scene.add(hand_mesh, 'hand_mesh')

    # add obj mesh
    for o in range(len(obj_pose)):
        if np.all(obj_pose[o] == 0.0):
            continue
        if obj_pose[o].shape[0] == 3:
            pose = np.vstack((obj_pose[o], np.array([[0, 0, 0, 1]], dtype=np.float32)))
        else:
            pose = obj_pose[o]
            
        pose[1] *= -1
        pose[2] *= -1
        scene.add(obj_mesh[o], pose=pose)

    # add camera
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=shape[1], viewport_height=shape[0], point_size=1.0)
    
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]
    
    if add_offset:
        height, width = shape[0], shape[1]
        # xoffset = int(2 * (320 - princpt[0]))  # 308.5481
        xoffset = 2 * (width/2 - princpt[0])  # 308.5481
        # print(xoffset)
        T = np.array([[1, 0, xoffset], [0, 1, 0]], dtype=np.float32)
        valid_mask = cv2.warpAffine(valid_mask.astype(np.uint8), T, (width, height))[:, :, None]
        rgb = cv2.warpAffine(rgb, T, (width, height))

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    # img = rgb
    return img, rgb


def render_obj(img, obj_mesh, obj_pose, cam_param, add_offset=False):
    shape = img.shape

    # new scene
    scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]))

    # add obj mesh
    for o in range(len(obj_pose)):
        if np.all(obj_pose[o] == 0.0):
            continue
        if obj_pose[o].shape[0] == 3:
            pose = np.vstack((obj_pose[o], np.array([[0, 0, 0, 1]], dtype=np.float32)))
        else:
            pose = obj_pose[o]
            
        pose[1] *= -1
        pose[2] *= -1
        scene.add(obj_mesh[o], pose=pose)

    # add camera
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=shape[1], viewport_height=shape[0], point_size=1.0)
    
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]
    
    if add_offset:
        height, width = shape[0], shape[1]
        # xoffset = int(2 * (320 - princpt[0]))  # 308.5481
        xoffset = 2 * (width/2 - princpt[0])  # 308.5481
        # print(xoffset)
        T = np.array([[1, 0, xoffset], [0, 1, 0]], dtype=np.float32)
        valid_mask = cv2.warpAffine(valid_mask.astype(np.uint8), T, (width, height))[:, :, None]
        rgb = cv2.warpAffine(rgb, T, (width, height))

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    # img = rgb
    return img, rgb


def render_hand_property(shape, hand_verts, hand_faces, cam_param):
    # hand mesh
    hand_mesh = trimesh.Trimesh(hand_verts, hand_faces, process=False)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    hand_mesh.apply_transform(rot)
    hand_mesh = pyrender.Mesh.from_trimesh(hand_mesh, smooth=True)

    # new scene
    scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]))

    # add hand mesh
    scene.add(hand_mesh, 'hand_mesh')

    # add camera
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=shape[1], viewport_height=shape[0], point_size=1.0)

    # render
    renderer._renderer._program_cache = ShaderProgramCache(shader_dir='./shaders')
    normal, depth = renderer.render(scene)
    renderer.delete()

    # post processing
    depth = depth.astype(np.float32)

    return normal, depth


# visualize depth, change colormap here
def vis_depth(depth):
    depth = copy.deepcopy(depth)
    mask = depth > 0
    if np.sum(mask) > 0:
        depth_for_vis = depth - np.min(depth[mask])
    else:
        depth_for_vis = depth
    depth_for_vis[np.logical_not(mask)] = 0
    depth_for_vis = plt.cm.plasma(depth_for_vis)
    depth_for_vis = np.clip(depth_for_vis * 255, 0, 255).astype(np.uint8)
    depth_for_vis = depth_for_vis[:, :, :3]
    depth_for_vis[np.logical_not(mask)] = 0
    return depth_for_vis