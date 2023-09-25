import copy
import os
import time
from curses import use_default_colors
from re import T

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.pyplot import axis
from run_nerf_helpers import *
from utils.flow_utils import flow_to_image
import torch.distributed as dist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batchify_rays(t, chain_5frames,
                rays_flat, chunk=1024*16, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[1], chunk):
        ret = render_rays(t, chain_5frames, rays_flat[:, i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k: torch.cat(all_ret[k], 1) for k in all_ret}
    return all_ret


def render(t, chain_5frames,
           H, W, focal, focal_render=None,
           chunk=1024*16, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        if focal_render is not None: # False
            # Render full image using different focal length for dolly zoom. Inference only.
            rays_o, rays_d = get_rays(H, W, focal_render, c2w)
        elif len(H) >= 1:
            rays_o, rays_d = [], []
            for i in range(len(H)):
                curr_rays_o, curr_rays_d = get_rays(H[i].item(), W[i].item(), focal[i].item(), c2w[i])
                rays_o.append(curr_rays_o)
                rays_d.append(curr_rays_d)
            rays_o = torch.stack(rays_o, dim=0)
            rays_d = torch.stack(rays_d, dim=0)
        else:
            rays_o, rays_d = get_rays(H.item(), W.item(), focal.item(), c2w)
            rays_o = rays_o.unsqueeze(0)
            rays_d = rays_d.unsqueeze(0)
    else:
        # use provided ray batch
        rays_o, rays_d = rays[:, 0], rays[:, 1]
    if use_viewdirs: #True
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            raise NotImplementedError
        # Make all directions unit magnitude.
        # shape: [scense_size, batch_size, 3]
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [viewdirs.shape[0], -1, 3]).float() # (270, 480, 3) -> (270*480, 3)

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [rays_o.shape[0], -1, 3]).float()
    rays_d = torch.reshape(rays_d, [rays_d.shape[0], -1, 3]).float()
    near, far = near * \
        torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(t, chain_5frames,
                        rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[2:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    return all_ret


def render_path_batch(render_poses, time2render,
                    hwf, chunk, render_kwargs, savedir=None, focal2render=None):
    """Render frames using batch.

    Args:
      render_poses: array of shape [num_frame, 3, 4]. Camera-to-world transformation matrix of each frame.
      time2render: array of shape [num_frame]. Time of each frame.
      hwf: list. [Height of image in pixels, Width of image in pixels, Focal length of pinhole camera]
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      render_kwargs: dictionary. args for the render function.
      savedir: string. Directory to save results.
      focal2render: list. Only used to perform dolly-zoom.
    Returns:
      ret_dict: dictionary. Final and intermediate results.
    """
    H, W, focal = hwf

    ret_dict = {}
    rgbs = []
    rgbs_d = []
    rgbs_s = []
    dynamicnesses = []

    time_curr = time.time()
    for i, c2w in enumerate(render_poses):

        # print(i, time.time() - time_curr)
        print(f"rank {dist.get_rank()} img{i} {time.time() - time_curr}")
        time_curr = time.time()

        t = time2render[i]

        if focal2render is not None:
            # Render full image using different focal length
            rays_o, rays_d = get_rays(H, W, focal2render[i], c2w)
        else:
            rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o = torch.reshape(rays_o, (-1, 3))
        rays_d = torch.reshape(rays_d, (-1, 3))
        batch_rays = torch.stack([rays_o, rays_d], 0)
        rgb = []
        rgb_d = []
        rgb_s = []
        dynamicness = []
        for j in range(0, batch_rays.shape[1], chunk):
            # print(j, '/', batch_rays.shape[1])
            ret = render(t, False,
                         H, W, focal,
                         chunk=chunk, rays=batch_rays[:, j:j+chunk, :],
                         **render_kwargs)
            rgb.append(ret['rgb_map_full'].cpu())
            rgb_d.append(ret['rgb_map_d'].cpu())
            rgb_s.append(ret['rgb_map_s'].cpu())
            dynamicness.append(ret['dynamicness_map'].cpu())
        rgb = torch.reshape(torch.cat(rgb, 0), (H, W, 3)).numpy()
        rgb_d = torch.reshape(torch.cat(rgb_d, 0), (H, W, 3)).numpy()
        rgb_s = torch.reshape(torch.cat(rgb_s, 0), (H, W, 3)).numpy()
        dynamicness = torch.reshape(torch.cat(dynamicness, 0), (H, W)).numpy()

        # Not a good solution. Should take care of this when preparing the data.
        if W%2 == 1:
            # rgb = cv2.resize(rgb, (W - 1, H))
            rgb = rgb[:, :-1, :]
            rgb_d = rgb_d[:, :-1, :]
            rgb_s = rgb_s[:, :-1, :]
            dynamicness = dynamicness[:, :-1]
        rgbs.append(rgb)
        rgbs_d.append(rgb_d)
        rgbs_s.append(rgb_s)
        dynamicnesses.append(dynamicness)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    ret_dict['rgbs'] = np.stack(rgbs, 0)
    ret_dict['rgbs_d'] = np.stack(rgbs_d, 0)
    ret_dict['rgbs_s'] = np.stack(rgbs_s, 0)
    ret_dict['dynamicnesses'] = np.stack(dynamicnesses, 0)

    return ret_dict


def render_path(render_poses,
                time2render,
                hwf,
                chunk,
                render_kwargs,
                savedir=None,
                flows_gt_f=None,
                flows_gt_b=None,
                focal2render=None,
                feature_dict=None,
                near=0.0,
                eval_only=False,
                ):
    """Render frames.

    Args:
      render_poses: array of shape [num_frame, 3, 4]. Camera-to-world transformation matrix of each frame.
      time2render: array of shape [num_frame]. Time of each frame.
      hwf: list. [Height of image in pixels, Width of image in pixels, Focal length of pinhole camera]
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      render_kwargs: dictionary. args for the render function.
      savedir: string. Directory to save results.
      focal2render: list. Only used to perform dolly-zoom.
    Returns:
      ret_dict: dictionary. Final and intermediate results.
    """
    H, W, focal = torch.split(hwf, [1, 1, 1], -1)
    H = H.int()
    W = W.int()
    focal = focal.cuda()

    ret_dict = {}
    rgbs = []
    rgbs_d = []
    rgbs_s = []
    depths = []
    depths_d = []
    depths_s = []
    flows_f = []
    flows_b = []
    dynamicness = []
    blending = []
    rgbs_dyreg_d = []

    grid = np.stack(np.meshgrid(np.arange(W[0].item(), dtype=np.float32),
                       np.arange(H[0].item(), dtype=np.float32), indexing='xy'), -1)
    grid = np.tile(grid, (len(hwf), 1, 1, 1))
    grid = torch.Tensor(grid).cuda() # 270, 480, 2
    time_curr = time.time()
    for i in range(len(render_poses[0])):
        t = time2render[:, i]
        t = torch.Tensor(t).cuda()
        c2w = render_poses[:, i]
        pose = c2w[:, :3, :4]
        print(f"rank {dist.get_rank()} img {i} {time.time() - time_curr}")
        # print(i, time.time() - time_curr)
        time_curr = time.time()

        if focal2render is None:
            # Normal rendering.
            ret = render(t, False,
                         H, W, focal,
                         chunk=1024*32, c2w=pose,
                         feature_dict=feature_dict,
                         **render_kwargs)
        else:
            # Render image using different focal length.
            ret = render(t, False,
                         H, W, focal, focal_render=focal2render[i],
                         chunk=1024*32, c2w=pose,
                         feature_dict=feature_dict,
                         **render_kwargs)

        rgbs.append(ret['rgb_map_full'].cpu().numpy())
        rgbs_d.append(ret['rgb_map_d'].cpu().numpy())
        rgbs_s.append(ret['rgb_map_s'].cpu().numpy())

        if render_kwargs['ndc']:
            depths.append(ret['depth_map_full'].cpu().numpy())
            depths_d.append(ret['depth_map_d'].cpu().numpy())
            depths_s.append(ret['depth_map_s'].cpu().numpy())
        else:
            depths.append(normalize_depth(ret['depth_map_full'], near).cpu().numpy())
            depths_d.append(normalize_depth(ret['depth_map_d'], near).cpu().numpy())
            depths_s.append(normalize_depth(ret['depth_map_s'], near).cpu().numpy())

        dynamicness.append(ret['dynamicness_map'].cpu().numpy())
        rgbs_dyreg_d.append(ret['rgb_map_dyreg_d'].cpu().numpy())

        if flows_gt_f is not None:
            # Reconstruction. Flow is caused by both changing camera and changing time.
            pose_f = render_poses[:, min(i + 1, int(len(render_poses)) - 1), :3, :4]
            pose_b = render_poses[:, max(i - 1, 0), :3, :4]
        else:
            # Non training view-time. Flow is caused by changing time (just for visualization).
            pose_f = render_poses[:, i, :3, :4]
            pose_b = render_poses[:, i, :3, :4]

        # Sceneflow induced optical flow
        curr_flows_f = []
        curr_flows_b = []
        for b in range(len(H)):
            if render_kwargs['ndc']:
                induced_flow_f_ = induce_flow(H[b], W[b], focal[b].cpu(), pose_f[b].cpu(), ret['weights_d'][b], ret['raw_pts_f'][b], grid[b, ..., :2].cpu())
                induced_flow_b_ = induce_flow(H[b], W[b], focal[b].cpu(), pose_b[b].cpu(), ret['weights_d'][b], ret['raw_pts_b'][b], grid[b, ..., :2].cpu())
            else:
                induced_flow_f_ = induce_flow_wo_ndc(H[b].item(), W[b].item(), focal[b], pose_f[b], ret['weights_d'][b], ret['raw_pts_f'][b], grid[b, ..., :2])
                induced_flow_b_ = induce_flow_wo_ndc(H[b].item(), W[b].item(), focal[b], pose_b[b], ret['weights_d'][b], ret['raw_pts_b'][b], grid[b, ..., :2])

            if (i + 1) >= len(render_poses[0]):
                induced_flow_f = np.zeros((H[b].item(), W[b].item(), 2))
            else:
                induced_flow_f = induced_flow_f_.cpu().numpy()
            if flows_gt_f is not None:
                flow_gt_f = flows_gt_f[b, i].cpu().numpy()
                induced_flow_f = np.concatenate((induced_flow_f, flow_gt_f), 0)
            induced_flow_f_img = flow_to_image(induced_flow_f)
            curr_flows_f.append(induced_flow_f_img)

            if (i - 1) < 0:
                induced_flow_b = np.zeros((H[b].item(), W[b].item(), 2))
            else:
                induced_flow_b = induced_flow_b_.cpu().numpy()
            if flows_gt_b is not None:
                flow_gt_b = flows_gt_b[b, i].cpu().numpy()
                induced_flow_b = np.concatenate((induced_flow_b, flow_gt_b), 0)
            induced_flow_b_img = flow_to_image(induced_flow_b)
            curr_flows_b.append(induced_flow_b_img)

        curr_flows_f = np.stack(curr_flows_f, axis=0)
        curr_flows_b = np.stack(curr_flows_b, axis=0)
        flows_b.append(curr_flows_b)
        flows_f.append(curr_flows_f)

        if i == 0:
            ret_dict['sceneflow_f_NDC'] = ret['sceneflow_f'].cpu().numpy()
            ret_dict['sceneflow_b_NDC'] = ret['sceneflow_b'].cpu().numpy()
            ret_dict['blending'] = ret['blending'].cpu().numpy()

            weights = np.concatenate((ret['weights_d'][..., None].cpu().numpy(),
                                      ret['weights_s'][..., None].cpu().numpy(),
                                      ret['blending'][..., None].cpu().numpy(),
                                      ret['weights_full'][..., None].cpu().numpy()), axis=1)
            ret_dict['weights'] = np.moveaxis(weights, [0, 1, 2, 3, 4], [0, 2, 3, 1, 4])


        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            os.makedirs(os.path.join(savedir, 'imgs'), exist_ok=True)
            if eval_only:
                os.makedirs(os.path.join(savedir, 'flows_f'), exist_ok=True)
                os.makedirs(os.path.join(savedir, 'flows_b'), exist_ok=True)
            for b in range(len(rgb8)):
                filename = os.path.join(savedir, 'imgs', '{:03d}_{:02d}.png'.format(i, b))
                imageio.imwrite(filename, rgb8[b])
                if eval_only:
                    if i > 0:
                        filename = os.path.join(savedir, 'flows_b', '{:03d}_{:02d}.png'.format(i, b))
                        imageio.imwrite(filename, flows_b[-1][b])
                    if i < len(render_poses[0]) - 1:
                        filename = os.path.join(savedir, 'flows_f', '{:03d}_{:02d}.png'.format(i, b))
                        imageio.imwrite(filename, flows_f[-1][b])


        if "features_d" in ret.keys():
            del ret['features_d'], ret['features_d_f'], ret['features_d_b']

    ret_dict['rgbs'] = np.stack(rgbs, 1)
    ret_dict['rgbs_d'] = np.stack(rgbs_d, 1)
    ret_dict['rgbs_s'] = np.stack(rgbs_s, 1)
    ret_dict['depths'] = np.stack(depths, 1)
    ret_dict['depths_d'] = np.stack(depths_d, 1)
    ret_dict['depths_s'] = np.stack(depths_s, 1)
    ret_dict['dynamicness'] = np.stack(dynamicness, 1)
    ret_dict['flows_f'] = np.stack(flows_f, 1)
    ret_dict['flows_b'] = np.stack(flows_b, 1)
    ret_dict['rgbs_dyreg_d'] = np.stack(rgbs_dyreg_d, 1)

    return ret_dict


def raw2outputs(raw_s,
                raw_d,
                blending,
                z_vals,
                rays_d,
                raw_noise_std):
    """Transforms model's predictions to semantically meaningful values.

    Args:
      raw_d: [num_rays, num_samples along ray, 4]. Prediction from Dynamic model.
      raw_s: [num_rays, num_samples along ray, 4]. Prediction from Static model.
      z_vals: [num_rays, num_samples along ray]. Integration time.
      rays_d: [num_rays, 3]. Direction of each ray.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      disp_map: [num_rays]. Disparity map. Inverse of depth map.
      acc_map: [num_rays]. Sum of weights along each ray.
      weights: [num_rays, num_samples]. Weights assigned to each sampled color.
      depth_map: [num_rays]. Estimated distance to object.
    """
    # Function for computing density from model prediction. This value is
    # strictly between [0, 1].

    def raw2alpha(raw, dists, act_fn=F.relu): return 1.0 - \
        torch.exp(-act_fn(raw) * dists)

    # Compute 'distance' (in time) between each integration time along a ray.
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # The 'distance' from the last integration time is infinity.
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).cuda()],
         -1) # [N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Extract RGB of each sample position along each ray.
    rgb_d = torch.sigmoid(raw_d[..., :3])  # [N_rays, N_samples, 3]
    rgb_s = torch.sigmoid(raw_s[..., :3])  # [N_rays, N_samples, 3]

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_d[..., 3].shape, device=device) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha_d = raw2alpha(raw_d[..., 3] + noise, dists) # [N_rays, N_samples]
    alpha_s = raw2alpha(raw_s[..., 3] + noise, dists) # [N_rays, N_samples]
    alphas  = 1. - (1. - alpha_s) * (1. - alpha_d) # [N_rays, N_samples]

    T_d    = torch.cumprod(torch.cat([torch.ones((*alpha_d.shape[:2], 1), device=device), 1. - alpha_d + 1e-10], -1), -1)[:, :, :-1]
    T_s    = torch.cumprod(torch.cat([torch.ones((*alpha_s.shape[:2], 1), device=device), 1. - alpha_s + 1e-10], -1), -1)[:, :, :-1]
    T_full = torch.cumprod(torch.cat([torch.ones((*alpha_d.shape[:2], 1), device=device), (1. - alpha_d * blending) * (1. - alpha_s * (1. - blending)) + 1e-10], -1), -1)[:, :, :-1]
    # T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)), torch.pow(1. - alpha_d + 1e-10, blending) * torch.pow(1. - alpha_s + 1e-10, 1. - blending)], -1), -1)[:, :-1]
    # T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)), (1. - alpha_d) * (1. - alpha_s) + 1e-10], -1), -1)[:, :-1]

    # Compute weight for RGB of each sample along each ray.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    # alpha_s = alpha_s.flip(0)
    # rgb_s = rgb_s.flip(0)
    # print("WARNING: FLIPING BACKGROUND")
    weights_d = alpha_d * T_d
    weights_dyreg_d = alpha_d * blending * T_d
    weights_s = alpha_s * T_s
    weights_full = (alpha_d * blending + alpha_s * (1. - blending)) * T_full
    # weights_full = alphas * T_full

    # Computed weighted color of each sample along each ray.
    rgb_map_d = torch.sum(weights_d[..., None] * rgb_d, -2)
    rgb_map_dyreg_d = torch.sum(weights_dyreg_d[..., None] * rgb_d, -2)
    rgb_map_s = torch.sum(weights_s[..., None] * rgb_s, -2)
    rgb_map_full = torch.sum(
        (T_full * alpha_d * blending)[..., None] * rgb_d + \
        (T_full * alpha_s * (1. - blending))[..., None] * rgb_s, -2)

    # Estimated depth map is expected distance.
    depth_map_d = torch.sum(weights_d * z_vals, -1)
    depth_map_s = torch.sum(weights_s * z_vals, -1)
    depth_map_full = torch.sum(weights_full * z_vals, -1)

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map_d = torch.sum(weights_d, -1)
    acc_map_s = torch.sum(weights_s, -1)
    acc_map_full = torch.sum(weights_full, -1)

    # Computed dynamicness
    dynamicness_map = torch.sum(weights_full * blending, -1)
    # dynamicness_map = 1 - T_d[..., -1]

    # Computed blending map
    blending_map = z_vals * blending

    return rgb_map_full, depth_map_full, acc_map_full, weights_full, \
           rgb_map_s, depth_map_s, acc_map_s, weights_s, \
           rgb_map_d, depth_map_d, acc_map_d, weights_d, dynamicness_map, \
           blending_map, rgb_map_dyreg_d


def raw2outputs_d(raw_d,
                  z_vals,
                  rays_d,
                  raw_noise_std):

    # Function for computing density from model prediction. This value is
    # strictly between [0, 1].
    def raw2alpha(raw, dists, act_fn=F.relu): return 1.0 - \
        torch.exp(-act_fn(raw) * dists)

    # Compute 'distance' (in time) between each integration time along a ray.
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # The 'distance' from the last integration time is infinity.
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).cuda()],
        -1)  # [N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Extract RGB of each sample position along each ray.
    rgb_d = torch.sigmoid(raw_d[..., :3])  # [N_rays, N_samples, 3]

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_d[..., 3].shape, device=device) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha_d = raw2alpha(raw_d[..., 3] + noise, dists)  # [N_rays, N_samples]

    T_d = torch.cumprod(torch.cat([torch.ones((*alpha_d.shape[:2], 1), device=device), 1. - alpha_d + 1e-10], -1), -1)[:, :, :-1]
    # Compute weight for RGB of each sample along each ray.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    weights_d = alpha_d * T_d

    # Computed weighted color of each sample along each ray.
    rgb_map_d = torch.sum(weights_d[..., None] * rgb_d, -2)
    depth_map_d = torch.sum(weights_d.detach() * z_vals.detach(), -1)
    acc_map_d = torch.sum(weights_d.detach(), -1)

    return rgb_map_d, weights_d, depth_map_d, acc_map_d, alpha_d, rgb_d


def render_rays(t,
                chain_5frames,
                ray_batch,
                network_fn_d,
                network_query_fn_d,
                N_samples,
                num_img,
                DyNeRF_blending,
                pretrain=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                raw_noise_std=0.,
                inference=False,
                use_feature=False,
                feature_dict=None):

    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn_d: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn_d: function used for passing queries to network_fn_d.
      N_samples: int. Number of different times to sample along each ray.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      raw_noise_std: ...
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    # batch size
    N_scenes, N_rays = ray_batch.shape[:2]

    # ray_batch: [N_scenes, N_rays, 11]
    # rays_o:    [N_scenes, N_rays, 0:3]
    # rays_d:    [N_scenes, N_rays, 3:6]
    # near:      [N_scenes, N_rays, 6:7]
    # far:       [N_scenes, N_rays, 7:8]
    # viewdirs:  [N_scenes, N_rays, 8:11]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, :, 0:3], ray_batch[:, :, 3:6] # [N_scenes, N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, :, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = torch.reshape(ray_batch[..., 6:8], [N_scenes, -1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = z_vals.expand([N_scenes, N_rays, N_samples])

    # Perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand


    # Points in space to evaluate model at.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None] # [N_scenes, N_rays, N_samples, 3]

    # Add the time dimension to xyz.
    pts_ref = torch.cat([pts, torch.ones_like(pts[..., 0:1], device=device) * t[:, None, None, None]], -1)

    # First pass: we have the staticNeRF results
    # raw_s = network_query_fn_s(pts_ref[..., :3], viewdirs, network_fn_s)
    # raw_s:          [N_rays, N_samples, 5]
    # raw_s_rgb:      [N_rays, N_samples, 0:3]
    # raw_s_a:        [N_rays, N_samples, 3:4]
    # raw_s_blending: [N_rays, N_samples, 4:5]

    # Second pass: we have the DyanmicNeRF results and the blending weight
    # import ipdb; ipdb.set_trace()
    # H = 270; W = 480; f=418.9622; near=1.0
    # world_xyz = NDC2world(pts_ref, H, W, f)

    if pretrain:
        raw_s, raw_d = network_query_fn_d(pts_ref, viewdirs, network_fn_d, use_feature=use_feature, use_network='static', feature_dict=feature_dict)
    else:
        raw_s, raw_d = network_query_fn_d(pts_ref, viewdirs, network_fn_d, use_feature=use_feature, use_network='static dynamic', feature_dict=feature_dict, chain_5frames=chain_5frames)
    # raw_d:          [N_rays, N_samples, 11+256]
    # raw_d_rgb:      [N_rays, N_samples, 0:3]
    # raw_d_a:        [N_rays, N_samples, 3:4]
    # sceneflow_b:    [N_rays, N_samples, 4:7]
    # sceneflow_f:    [N_rays, N_samples, 7:10]
    # raw_d_blending: [N_rays, N_samples, 10:11]
    # raw_d_features: [N_rays, N_samples, 11:267]
    if pretrain:
        rgb_map_s, weights_s, depth_map_s, acc_map_s, alpha_s, rgb_s = raw2outputs_d(raw_s[..., :4],
                                     z_vals,
                                     rays_d,
                                     raw_noise_std)
        ret = {
            'rgb_map_s': rgb_map_s,
            'weights_s': weights_s,
            'depth_map_s': depth_map_s,
            'acc_map_s': acc_map_s,
            'alpha_s': alpha_s,
            'rgb_s': rgb_s,
            }
        return ret

    raw_s_rgba = raw_s[..., :4]
    raw_d_rgba = raw_d[..., :4]

    # We need the sceneflow from the dynamicNeRF.
    sceneflow_b = raw_d[..., 4:7] - pts
    sceneflow_f = raw_d[..., 7:10] - pts

    if DyNeRF_blending:
        blending = raw_d[..., 10]
    else:
        blending = raw_s[..., 4]

    # if sfmask:
    #     sceneflow_f = sceneflow_f * blending.detach()[..., None]
    #     sceneflow_b = sceneflow_b * blending.detach()[..., None]

    # Rerndering.
    rgb_map_full, depth_map_full, acc_map_full, weights_full, \
    rgb_map_s, depth_map_s, acc_map_s, weights_s, \
    rgb_map_d, depth_map_d, acc_map_d, weights_d, \
    dynamicness_map, blending_map, rgb_map_dyreg_d = raw2outputs(raw_s_rgba,
                                  raw_d_rgba,
                                  blending,
                                  z_vals,
                                  rays_d,
                                  raw_noise_std)

    ret = {'rgb_map_full': rgb_map_full, 'depth_map_full': depth_map_full, 'acc_map_full': acc_map_full, 'weights_full': weights_full,
           'rgb_map_s': rgb_map_s, 'depth_map_s': depth_map_s, 'acc_map_s': acc_map_s, 'weights_s': weights_s,
           'rgb_map_d': rgb_map_d, 'depth_map_d': depth_map_d, 'acc_map_d': acc_map_d, 'weights_d': weights_d,
           'dynamicness_map': dynamicness_map, 'blending_map': blending_map, 'z_vals': z_vals, 'rgb_map_dyreg_d': rgb_map_dyreg_d}
    if raw_d.shape[-1] > 11:
        ret['features_d'] = raw_d[..., 11:]

    t_interval = 1. / num_img * 2.
    pts_f = torch.cat([pts + sceneflow_f, torch.ones_like(pts[..., 0:1]) * (t + t_interval)[:, None, None, None]], -1)
    pts_b = torch.cat([pts + sceneflow_b, torch.ones_like(pts[..., 0:1]) * (t - t_interval)[:, None, None, None]], -1)

    ret['sceneflow_b'] = sceneflow_b
    ret['sceneflow_f'] = sceneflow_f
    ret['raw_pts'] = pts_ref[..., :3]
    ret['raw_pts_f'] = pts_f[..., :3]
    ret['raw_pts_b'] = pts_b[..., :3]
    ret['blending'] = blending

    # Third pass: we have the DyanmicNeRF results at time t - 1
    raw_s_b, raw_d_b = network_query_fn_d(pts_b, viewdirs, network_fn_d, use_feature=use_feature, use_network='dynamic',  feature_dict=feature_dict)
    raw_d_b_rgba = raw_d_b[..., :4]
    sceneflow_b_b = raw_d_b[..., 4:7] - pts_b[..., :3]
    sceneflow_b_f = raw_d_b[..., 7:10] - pts_b[..., :3]
    blending_b = raw_d_b[..., 10]

    # Rerndering t - 1
    rgb_map_d_b, weights_d_b, *___ = raw2outputs_d(raw_d_b_rgba,
                                             z_vals,
                                             rays_d,
                                             raw_noise_std)

    ret['sceneflow_b_f'] = sceneflow_b_f
    ret['rgb_map_d_b'] = rgb_map_d_b
    ret['acc_map_d_b'] = torch.abs(torch.sum(weights_d_b - weights_d, -1))
    ret['blending_b'] = blending_b
    if raw_d_b.shape[-1] > 11:
        ret['features_d_b'] = raw_d_b[..., 11:]

    # Fourth pass: we have the DyanmicNeRF results at time t + 1
    raw_s_f, raw_d_f = network_query_fn_d(pts_f, viewdirs, network_fn_d, use_feature=use_feature, use_network='dynamic',  feature_dict=feature_dict)
    raw_d_f_rgba = raw_d_f[..., :4]
    sceneflow_f_b = raw_d_f[..., 4:7]  - pts_f[..., :3]
    sceneflow_f_f = raw_d_f[..., 7:10] - pts_f[..., :3]
    blending_f = raw_d_f[..., 10]

    # Rerndering t + 1
    rgb_map_d_f, weights_d_f, *___ = raw2outputs_d(raw_d_f_rgba,
                                             z_vals,
                                             rays_d,
                                             raw_noise_std)

    ret['sceneflow_f_b'] = sceneflow_f_b
    ret['rgb_map_d_f'] = rgb_map_d_f
    ret['acc_map_d_f'] = torch.abs(torch.sum(weights_d_f - weights_d, -1))
    ret['blending_f'] = blending_f
    if raw_d_f.shape[-1] > 11:
        ret['features_d_f'] = raw_d_f[..., 11:]

    if inference:
        for key in ret.keys():
            ret[key] = ret[key].cpu()
        return ret

    # Also consider time t - 2 and t + 2 (Learn from NSFF)

    # Fifth pass: we have the DyanmicNeRF results at time t - 2
    pts_b_b = torch.cat([pts_b[..., :3] + sceneflow_b_b, torch.ones_like(pts[..., 0:1]) * (t - t_interval * 2)[:, None, None, None]], -1)
    ret['raw_pts_b_b'] = pts_b_b[..., :3]

    if chain_5frames:
        sceneflow_b_0 = raw_d[..., 11:14] - pts
        sceneflow_f_1 = raw_d[..., 14:17] - pts
        pts_f_1 = torch.cat([pts + sceneflow_f_1, (11/12*2-1) * torch.ones_like(pts[..., 0:1])], -1)
        pts_b_0 = torch.cat([pts + sceneflow_b_0, -1 * torch.ones_like(pts[..., 0:1])], -1)
        raw_s_b_b, raw_d_b_b = network_query_fn_d(pts_b_0, viewdirs, network_fn_d, use_feature=use_feature, use_network='dynamic',  feature_dict=feature_dict)
        raw_d_b_b_rgba = raw_d_b_b[..., :4]
        rgb_map_d_b_b, *___ = raw2outputs_d(raw_d_b_b_rgba,
                                      z_vals,
                                      rays_d,
                                      raw_noise_std)

        ret['rgb_map_d_b_b'] = rgb_map_d_b_b

    # Sixth pass: we have the DyanmicNeRF results at time t + 2
    pts_f_f = torch.cat([pts_f[..., :3] + sceneflow_f_f, torch.ones_like(pts[..., 0:1]) * (t + t_interval * 2)[:, None, None, None]], -1)
    ret['raw_pts_f_f'] = pts_f_f[..., :3]

    if chain_5frames:
        raw_s_f_f, raw_d_f_f = network_query_fn_d(pts_f_1, viewdirs, network_fn_d, use_feature=use_feature, use_network='dynamic',  feature_dict=feature_dict)
        raw_d_f_f_rgba = raw_d_f_f[..., :4]
        rgb_map_d_f_f, *___= raw2outputs_d(raw_d_f_f_rgba,
                                      z_vals,
                                      rays_d,
                                      raw_noise_std)

        ret['rgb_map_d_f_f'] = rgb_map_d_f_f

    for k in ret:
        if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
            print(f'! [Numerical Error] {k} contains nan or inf.')
            import ipdb; ipdb.set_trace()

    return ret
