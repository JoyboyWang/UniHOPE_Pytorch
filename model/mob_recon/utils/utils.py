import torch
import os
import numpy as np
import openmesh as om


def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_edge_index(mat):
    return torch.LongTensor(np.vstack(mat.nonzero()))


def to_sparse(spmat):
    return torch.sparse.FloatTensor(torch.LongTensor([spmat.tocoo().row, spmat.tocoo().col]), torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    from .generate_spiral_seq import extract_spirals
    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        n_vertices = face.max() + 1
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    spirals = torch.tensor(extract_spirals(mesh, seq_length=seq_length, dilation=dilation))
    return spirals


def camera2world(xcam, R, T):
    # torch.Size([B, N, 3])
    # R: torch.Shape: [B, 3, 3]
    # T: torch.Shape: [B, 3]
    xcam = torch.bmm(R, xcam.permute(0, 2, 1))  # matmul
    x = xcam + T.unsqueeze(2)  # rotate and translate
    return x.permute(0, 2, 1)


def world2camera(x, R, T):
    xcam = torch.bmm(R.permute(0, 2, 1), x.permute(0, 2, 1) - T.unsqueeze(2))  # matmul
    return xcam.permute(0, 2, 1)  # to camera coordinates


def camera2world_RT(xcam, RT):
    xcam = torch.cat((xcam, torch.ones_like(xcam[:, :, :1])), dim=2)  # [B, 778, 4]
    return torch.bmm(RT, xcam.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]  # [B, 778, 3]

def world2camera_RT(x, RT):
    x = torch.cat((x, torch.ones_like(x[:, :, :1])), dim=2)  # [B, 778, 4]
    return torch.bmm(torch.inverse(RT), x.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]  # [B, 778, 3]

def weird_division(n, d):
    # import ipdb; ipdb.set_trace()
    # if divided by zero, set it to zero
    mask = (d == 0).squeeze(2)
    out = n / d
    out[mask] = 0
    return out


# FIXME: remove relying on batch dimension
def camera2image(xcam, f, c):
    # import ipdb; ipdb.set_trace()
    # ximg = weird_division(xcam[:, :, :2], xcam[:, :, 2:])  # divide by depth, be careful
    ximg = xcam[:, :, :2] / xcam[:, :, 2:]
    return (f.unsqueeze(1) * ximg) + c.unsqueeze(1)  # 2D projection in image


def world2image(x, R, T, f, c):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: 2x1 Camera focal length
        c: 2x1 Camera center
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    # import ipdb; ipdb.set_trace()
    xcam = world2camera(x, R, T)
    ximg = camera2image(xcam, f, c)
    return ximg

def cropview(ximg, w, h):
    return ximg[ximg >= 0 and ximg[:, :, 0] <= w and ximg[:, :, 1] <= h]


def masked_mean(tensor, mask, dim=None):
    # Apply the mask using an element-wise multiply
    masked = torch.mul(tensor, mask)  

    # Find the average!
    if dim is not None:
        return masked.sum(dim=dim) / (mask.sum(dim=dim) + 1e-8)  
    else:
        return masked.sum() / (mask.sum() + 1e-8)

def masked_max(tensor, mask, dim):
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -torch.inf  # Place the smallest values possible in masked positions

    res = (masked + neg_inf).max(dim=dim)[0]
    # if all inf, return zero
    res[res == (-torch.inf)] = 0
    return res

def masked_std(tensor, mask, dim):
    mean = masked_mean(tensor, mask, dim).unsqueeze(dim=dim)  # torch.Size([B, C, 21]) -> torch.Size([B,(1) C, (1) 21])
    res = torch.mul(tensor - mean, mask).pow(2).sum(dim=dim)
    res = res / (mask.sum(dim=dim) + 1e-8)  # only average by the masked elements

    return torch.sqrt(res)

def masked_norm(tensor, mask):
    bool_mask = mask.astype(bool)
    if not np.any(bool_mask):  # if no mask, return tensor itself
        return tensor

    tensor = tensor.astype(np.float32)
    masked_tensor = tensor[bool_mask]
    tensor[bool_mask] = (masked_tensor - masked_tensor.min()) / (masked_tensor.max() - masked_tensor.min() + 1e-6)
    return tensor
