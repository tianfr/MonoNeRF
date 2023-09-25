import os
import time
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import trainlib
from render_utils import *
from run_nerf_helpers import *
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src import util

import torch.distributed as dist


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


args, conf = util.args.parse_args()
device = util.get_cuda(args.gpu_id[0])


#!--------------------------------------
args.add_features = conf['info.add_features']
args.basedir = conf['info.basedir']
args.expname = conf['info.expname']
args.dataset_file_lists = conf['info.dataset_file_lists']
args.ft_path_S = conf['info.ft_path_S']
args.ft_S = conf['info.ft_S']
args.no_ndc = conf['info.no_ndc']
args.random_seed = conf['info.random_seed']
args.freeze_enc = conf['info.freeze_enc']
args.N_rand = conf['info.N_rand']
args.blending_thickness = conf['info.blending_thickness']
args.chunk = conf['info.chunk']
args.slow_loss_lambda = conf['info.slow_loss_lambda']
args.flow_loss_lambda = conf['info.flow_loss_lambda']
#!--------------------------------------


class MonoNeRFTrainer(trainlib.Trainer):
    def __init__(self, args, conf):
        super().__init__(args, conf, device=None)

    def train_step(self, data, global_step):

        encoder_imgs = data['encoder_imgs'].permute(0, 4, 1, 2, 3)
        images = data['images']
        invdepths = data['invdepths']
        masks = data['masks']
        poses = data['poses']
        bds = data['bds']
        render_poses = data['render_poses']
        render_focals = data['render_focals']
        grids = data['grids']

        hwf = poses[:, 0, :3, -1]
        poses = poses[:, :, :3, :4]
        num_img = float(poses.shape[1])
        assert len(poses) == len(images)
        self.logger.info('Loaded llff'+str(images.shape)+
            str(render_poses.shape)+str(hwf)+str(self.args.dataset_file_lists))

        # Use all views to train
        i_train = np.array([i for i in np.arange(int(images.shape[1]))])

        self.logger.info('DEFINING BOUNDS')
        #! Change the pos
        if self.args.no_ndc:
            near = bds.min() * .9
            far = bds.max() * 1.
        else:
            near = 0.
            far = 1.
            self.logger.info(f'NEAR FAR {near} {far}')

        H, W, focal = torch.split(hwf, [1,1,1], -1)
        H, W = H.int(), W.int()
        hwf = torch.cat([H, W, focal], dim=-1)


        global_step = self.start

        bds_dict = {
            'near': near,
            'far': far,
            'num_img': num_img,
        }
        self.render_kwargs_train.update(bds_dict)
        self.render_kwargs_test.update(bds_dict)

        self.render_kwargs_train['use_feature'] = True
        self.render_kwargs_test['use_feature'] = True

        N_rand = self.args.N_rand #1024

        # Move training data to GPU
        images = torch.Tensor(images).cuda()
        invdepths = torch.Tensor(invdepths).cuda()
        masks = 1.0 - torch.Tensor(masks).cuda()
        poses = torch.Tensor(poses).cuda()
        grids = torch.Tensor(grids).cuda()

        if self.rank == 0:
            self.logger.info('Begin')
            self.logger.info('TRAIN views are'+ str(i_train))


        decay_iteration = self.conf.get("info.decay_iteration", max(25, num_img))

        # Pre-train StaticNeRF
        if self.args.pretrain:
            self.net.pretrain = True
            self.render_kwargs_train.update({'pretrain': True})
            self.render_kwargs_test.update({'pretrain': True})

            # Pre-train StaticNeRF first and use DynamicNeRF to blend
            assert self.args.DyNeRF_blending == True

            if self.conf['model']['mlp_static']['origin_pipeline'] == True or self.args.freeze_enc:
                with torch.no_grad():
                    feature_dict = self.net.train_step(encoder_imgs, poses, focal, images2d=images, mode="encode")
            # Train StaticNeRF from scratch
            with logging_redirect_tqdm():
                for i in tqdm(range(self.args.N_static_iters)):
                    time0 = time.time()
                    batch_masks = []
                    batch_rays = []
                    batch_poses = []
                    batch_invdepths = []
                    target_rgbs = []
                    rays_o = []
                    rays_d = []
                    t = []

                    if self.conf['model']['mlp_static']['origin_pipeline'] == False and not self.args.freeze_enc:
                        feature_dict = self.net.train_step(encoder_imgs, poses, focal, images2d=images, mode="encode")

                    for b in range(images.shape[0]):
                        # No raybatching as we need to take random rays from one image at a time
                        img_i = np.random.choice(i_train)
                        curr_t = img_i / num_img * 2. - 1.0 # time of the current frame ## range from -1 to 1
                        target = images[b, img_i]
                        pose = poses[b, img_i, :3, :4]
                        mask = masks[b, img_i] # Static region mask
                        invdepth = invdepths[b, img_i]
                        curr_rays_o, curr_rays_d = get_rays(H[b].item(), W[b].item(), focal[b].item(), pose) # (H, W, 3), (H, W, 3)
                        coords_s = torch.stack((torch.where(mask >= 0.5)), -1)
                        select_inds_s = np.random.choice(coords_s.shape[0], size=[N_rand], replace=False)
                        select_coords = coords_s[select_inds_s]

                        def select_batch(value, select_coords=select_coords):
                            return value[select_coords[:, 0], select_coords[:, 1]]

                        curr_rays_o = select_batch(curr_rays_o) # (N_rand, 3)
                        curr_rays_d = select_batch(curr_rays_d) # (N_rand, 3)
                        curr_target_rgb = select_batch(target)
                        curr_batch_mask = select_batch(mask[..., None])
                        curr_batch_invdepth = select_batch(invdepth)
                        curr_batch_rays = torch.stack([curr_rays_o, curr_rays_d], 0)

                        rays_o.append(curr_rays_o)
                        rays_d.append(curr_rays_d)
                        target_rgbs.append(curr_target_rgb)
                        batch_masks.append(curr_batch_mask)
                        batch_rays.append(curr_batch_rays)
                        batch_poses.append(pose)
                        batch_invdepths.append(curr_batch_invdepth)
                        t.append(curr_t)
                    rays_o = torch.stack(rays_o, dim=0)
                    rays_d = torch.stack(rays_d, dim=0)
                    target_rgbs = torch.stack(target_rgbs, dim=0)
                    batch_masks = torch.stack(batch_masks, dim=0)
                    batch_rays = torch.stack(batch_rays, dim=0)
                    batch_poses = torch.stack(batch_poses, dim=0)
                    batch_invdepths = torch.stack(batch_invdepths)
                    t = torch.Tensor(t).cuda()

                    #####  Core optimization loop  #####
                    ret = render(t,
                                False,
                                H, W, focal.cuda(),
                                chunk=self.args.chunk,
                                rays=batch_rays,
                                feature_dict=feature_dict,
                                **self.render_kwargs_train)

                    self.optimizer.zero_grad()
                    # if i % 10000 == 0:
                    #     import ipdb; ipdb.set_trace()

                    loss_dict = {}
                    # Compute MSE loss between rgb_s and true RGB.
                    img_s_loss = img2mse(ret['rgb_map_s'], target_rgbs)
                    psnr_s = mse2psnr(img_s_loss)
                    loss = self.args.static_loss_lambda * img_s_loss
                    loss_dict['psnr_s'] = psnr_s
                    loss_dict['img_s_loss'] = img_s_loss

                    loss.backward()
                    self.optimizer.step()

                    # Learning rate decay.
                    decay_rate = 0.1
                    decay_steps = self.args.lrate_decay
                    new_lrate = self.args.lrate * (decay_rate ** (global_step / decay_steps))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lrate

                    dt = time.time() - time0
                    if i % self.args.i_log == 0 and self.rank == 0:
                        self.logger.info(f'Pretraining step: {global_step}, Loss: {loss}, Time: {dt}, expname: {self.expname}')
                        loss_details = ''
                        for key in loss_dict.keys():
                            loss_details += f'{key}: ' + '%.3f' %(loss_dict[key].mean().item()) + ' '
                        self.logger.info(loss_details)

                    if i % self.args.i_print == 0 and self.rank == 0:
                        self.writer.add_scalar('loss', loss.item(), i)
                        self.writer.add_scalar('lr', new_lrate, i)
                        for loss_key in loss_dict:
                            self.writer.add_scalar(loss_key, loss_dict[loss_key].item(), i)

                    if i % self.args.i_img == 0 and self.rank == 0:
                        target = images[:, img_i]
                        pose = poses[:, img_i, :3, :4]
                        mask = masks[:, img_i]
                        with torch.no_grad():
                            ret = render(t,
                                        False,
                                        H, W, focal.cuda(),
                                        chunk=self.args.chunk,
                                        c2w=pose,
                                        feature_dict=feature_dict,
                                        **self.render_kwargs_test)

                            # Save out the validation image for Tensorboard-free monitoring
                            if len(target.shape) == 4:
                                for each in range(target.shape[0]):
                                    self.writer.add_image(f'rgb_holdout{each}', target[each], global_step=i, dataformats='HWC')
                                    self.writer.add_image(f'mask{each}', mask[each], global_step=i, dataformats='HW')
                                    self.writer.add_image(f'rgb_s{each}', torch.clamp(ret['rgb_map_s'][each], 0., 1.), global_step=i, dataformats='HWC')
                                    self.writer.add_image(f'depth_s{each}', normalize_depth(ret['depth_map_s'][each], near), global_step=i, dataformats='HW')
                                    self.writer.add_image(f'acc_s{each}', ret['acc_map_s'][each], global_step=i, dataformats='HW')
                            else:
                                self.writer.add_image('rgb_holdout0', target, global_step=i, dataformats='HWC')
                                self.writer.add_image('mask0', masks, global_step=i, dataformats='HW')
                                self.writer.add_image('rgb_s0', torch.clamp(ret['rgb_map_s'], 0., 1.), global_step=i, dataformats='HWC')
                                self.writer.add_image('depth_s0', normalize_depth(ret['depth_map_s'], near), global_step=i, dataformats='HW')
                                self.writer.add_image('acc_s0', ret['acc_map_s'], global_step=i, dataformats='HW')

                            del ret

                    global_step += 1
            if self.rank == 0:
                static_nerf_dict = {}

                for key in self.render_kwargs_train['network_fn_d'].module.state_dict().keys():
                    if 'mlp_static' in key or "encoder2d" in key:
                        static_nerf_dict[key] = self.render_kwargs_train['network_fn_d'].module.state_dict()[key]

                torch.save({
                    'global_step': global_step,
                    'network_fn_d_state_dict': static_nerf_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.basedir, self.expname, 'Pretrained_S.tar'))
            # Reset
        self.render_kwargs_train.update({'pretrain': False})
        self.render_kwargs_test.update({'pretrain': False})
        global_step = self.start
        self.net.pretrain = False

        grad_vars = list(self.net.encoder.parameters()) + list(self.net.mlp_dynamic.parameters())
        self.optimizer = torch.optim.Adam(params=grad_vars, lr=self.args.lrate, betas=(0.9, 0.999))
        set_requires_grad(self.net.encoder2d, False)
        set_requires_grad(self.net.mlp_static, False)

        if self.conf['model']['mlp_dynamic']['origin_pipeline'] == True or self.args.freeze_enc:
            with torch.no_grad():
                feature_dict = self.net.train_step(encoder_imgs, poses, focal, images2d=images, mode="encode")
        
        with logging_redirect_tqdm():
            for i in tqdm(range(self.start, self.args.N_iters)):
                time0 = time.time()
                batch_masks = []
                batch_rays = []
                batch_poses = []
                batch_invdepths = []
                batch_grids = []
                target_rgbs = []
                rays_o = []
                rays_d = []
                t = []
                img_ies = []

                # self.net.encode(encoder_imgs, poses, focal, images2d=images)
                if self.conf['model']['mlp_dynamic']['origin_pipeline'] == False and not self.args.freeze_enc:
                    feature_dict = self.net.train_step(encoder_imgs, poses, focal, images2d=images, mode="encode")
                if self.conf.get_bool("info.change_background", False) and i == 0:
                    print("WARNING: CHANGING BACKGROUND.")
                    feature_dict['latent2d_dict']['latent'] = feature_dict['latent2d_dict']['latent'].flip(0)

                # Use frames at t-2, t-1, t, t+1, t+2 (adapted from NSFF)
                if i < self.conf.get_int('info.chain_5frames_epoch', 300000):
                # if i < decay_iteration * 2000:
                    chain_5frames = False
                else:
                    chain_5frames = True

                # Lambda decay.
                if self.conf.get_bool('info.use_lambda_decay', False):
                    Temp = 1. / (10 ** (i // (decay_iteration * 1000)))
                else:
                    Temp = 1

                # if i % (decay_iteration * 1000) == 0:
                #     torch.cuda.empty_cache()

                for b in range(images.shape[0]):
                    # No raybatching as we need to take random rays from one image at a time
                    img_i = np.random.choice(i_train)
                    curr_t = img_i / num_img * 2. - 1.0 # time of the current frame
                    target = images[b, img_i]
                    pose = poses[b, img_i, :3, :4]
                    mask = masks[b, img_i] # Static region mask
                    invdepth = invdepths[b, img_i]
                    grid = grids[b, img_i]

                    curr_rays_o, curr_rays_d = get_rays(H[b].item(), W[b].item(), focal[b].item(), pose) # (H, W, 3), (H, W, 3)
                    coords_d = torch.stack((torch.where(mask < 0.5)), -1)
                    coords_s = torch.stack((torch.where(mask >= 0.5)), -1)
                    coords = torch.stack((torch.where(mask > -1)), -1)
                    # test coord transfer.
                    my_coords = torch.stack((torch.where(mask[:32, :32])), -1)

                    # Evenly sample dynamic region and static region
                    select_inds_d = np.random.choice(coords_d.shape[0], size=[min(len(coords_d), N_rand//2)], replace=False)
                    select_inds_s = np.random.choice(coords_s.shape[0], size=[N_rand//2], replace=False)
                    select_coords = torch.cat([coords_s[select_inds_s],
                                            coords_d[select_inds_d]], 0)

                    def select_batch(value, select_coords=select_coords):
                        return value[select_coords[:, 0], select_coords[:, 1]]
                    # def select_batch(value, select_coords=my_coords):
                    #     return value[select_coords[:, 0], select_coords[:, 1]]
                    curr_rays_o = select_batch(curr_rays_o) # (N_rand, 3)
                    curr_rays_d = select_batch(curr_rays_d) # (N_rand, 3)
                    curr_target_rgb = select_batch(target)
                    curr_batch_grid = select_batch(grid) # (N_rand, 8)
                    curr_batch_mask = select_batch(mask[..., None])
                    curr_batch_invdepth = select_batch(invdepth)
                    curr_batch_rays = torch.stack([curr_rays_o, curr_rays_d], 0)

                    rays_o.append(curr_rays_o)
                    rays_d.append(curr_rays_d)
                    target_rgbs.append(curr_target_rgb)
                    batch_masks.append(curr_batch_mask)
                    batch_rays.append(curr_batch_rays)
                    batch_poses.append(pose)
                    batch_grids.append(curr_batch_grid)
                    batch_invdepths.append(curr_batch_invdepth)
                    t.append(curr_t)
                    img_ies.append(img_i)

                rays_o = torch.stack(rays_o, dim=0)
                rays_d = torch.stack(rays_d, dim=0)
                target_rgbs = torch.stack(target_rgbs, dim=0)
                batch_masks = torch.stack(batch_masks, dim=0)
                batch_rays = torch.stack(batch_rays, dim=0)
                batch_poses = torch.stack(batch_poses, dim=0)
                batch_grids = torch.stack(batch_grids, dim=0)
                batch_invdepths = torch.stack(batch_invdepths)
                t = torch.Tensor(t).cuda()
                #####  Core optimization loop  #####
                ret = render(t,
                            chain_5frames,
                            H, W, focal.cuda(),
                            chunk=self.args.chunk,
                            rays=batch_rays,
                            feature_dict=feature_dict,
                            **self.render_kwargs_train)

                self.optimizer.zero_grad()
                loss = 0
                loss_dict = {}

                # First train the static NeRF by using rgb_map_s.
                # Compute MSE loss between rgb_s and true RGB.
                img_s_loss = img2mse(ret['rgb_map_s'][:N_rand//2], target_rgbs[:N_rand//2], batch_masks[:N_rand//2])
                psnr_s = mse2psnr(img_s_loss)
                loss_dict['psnr_s'] = psnr_s
                loss_dict['img_s_loss'] = img_s_loss


                # Next freeze the static NeRF and train the dynamic NeRF
                # Compute MSE loss between rgb_full and true RGB.
                img_loss = img2mse(ret['rgb_map_full'], target_rgbs)
                psnr = mse2psnr(img_loss)
                loss_dict['psnr'] = psnr
                loss_dict['img_loss'] = img_loss
                loss += self.args.full_loss_lambda * loss_dict['img_loss']

                # Compute MSE loss between rgb_d and true RGB.
                img_d_loss = img2mse(ret['rgb_map_d'], target_rgbs)
                psnr_d = mse2psnr(img_d_loss)
                loss_dict['psnr_d'] = psnr_d
                loss_dict['img_d_loss'] = img_d_loss
                loss += self.args.dynamic_loss_lambda * loss_dict['img_d_loss']

                # Compute MSE loss between rgb_d_f and true RGB.
                if self.conf.get_bool("info.use_feat_consistency", True):
                    if img_i < num_img - 1:
                        img_d_f_loss = img2mse(ret['rgb_map_d_f'], target_rgbs)
                        psnr_d_f = mse2psnr(img_d_f_loss)
                        loss_dict['psnr_d_f'] = psnr_d_f
                        loss_dict['img_d_f_loss'] = img_d_f_loss
                        loss += self.args.dynamic_loss_lambda * loss_dict['img_d_f_loss']

                    # Compute MSE loss between rgb_d_b and true RGB.
                    if img_i > 0:
                        img_d_b_loss = img2mse(ret['rgb_map_d_b'], target_rgbs)
                        psnr_d_b = mse2psnr(img_d_b_loss)
                        loss_dict['psnr_d_b'] = psnr_d_b
                        loss_dict['img_d_b_loss'] = img_d_b_loss
                        loss += self.args.dynamic_loss_lambda * loss_dict['img_d_b_loss']

                if self.conf.get_bool("info.use_flow", True):

                    for b in range(len(img_ies)):
                        img_i = img_ies[b]
                        # Motion loss.
                        # Compuate EPE between induced flow and true flow (forward flow).
                        # The last frame does not have forward flow.
                        if img_i < num_img - 1:
                            pts_f = ret['raw_pts_f'][b]
                            weight = ret['weights_d'][b]
                            pose_f = poses[b, img_i + 1, :3, :4]
                            if self.args.no_ndc:
                                induced_flow_f = induce_flow_wo_ndc(H[b].item(), W[b].item(), focal[b].item(), pose_f, weight, pts_f, batch_grids[b, ..., :2])
                            else:
                                induced_flow_f = induce_flow(H[b].cuda(), W[b].cuda(), focal[b].cuda(), pose_f, weight, pts_f, batch_grids[b, ..., :2])
                            flow_f_loss = img2mae(induced_flow_f, batch_grids[b, :, 2:4], batch_grids[b, :, 4:5])
                            if 'flow_f_loss' in loss_dict:
                                loss_dict['flow_f_loss'] += flow_f_loss
                            else:
                                loss_dict['flow_f_loss'] = flow_f_loss

                        # Compuate EPE between induced flow and true flow (backward flow).
                        # The first frame does not have backward flow.
                        if img_i > 0:
                            pts_b = ret['raw_pts_b'][b]
                            weight = ret['weights_d'][b]
                            pose_b = poses[b, img_i - 1, :3, :4]
                            if self.args.no_ndc:
                                induced_flow_b = induce_flow_wo_ndc(H[b].item(), W[b].item(), focal[b].item(), pose_b, weight, pts_b, batch_grids[b, ..., :2])
                            else:
                                induced_flow_b = induce_flow(H[b].cuda(), W[b].cuda(), focal[b].cuda(), pose_b, weight, pts_b, batch_grids[b, ..., :2])
                            flow_b_loss = img2mae(induced_flow_b, batch_grids[b, :, 5:7], batch_grids[b, :, 7:8])
                            if 'flow_b_loss' in loss_dict:
                                loss_dict['flow_b_loss'] += flow_b_loss
                            else:
                                loss_dict['flow_b_loss'] = flow_b_loss

                    if 'flow_f_loss' in loss_dict:
                        loss += self.args.flow_loss_lambda * Temp * loss_dict['flow_f_loss']
                    if 'flow_b_loss' in loss_dict:
                        loss += self.args.flow_loss_lambda * Temp * loss_dict['flow_b_loss']
                    # Slow scene flow. The forward and backward sceneflow should be small.
                    slow_loss = L1(ret['sceneflow_b']) + L1(ret['sceneflow_f'])
                    loss_dict['slow_loss'] = slow_loss
                    loss += self.args.slow_loss_lambda * loss_dict['slow_loss']

                    # Smooth scene flow. The summation of the forward and backward sceneflow should be small.
                    if self.args.no_ndc:
                        smooth_loss = compute_sf_smooth_loss_wo_ndc(ret['raw_pts'],
                                                            ret['raw_pts_f'],
                                                            ret['raw_pts_b'],
                                                            H, W, focal)
                    else:
                        smooth_loss = compute_sf_smooth_loss(ret['raw_pts'],
                                                            ret['raw_pts_f'],
                                                            ret['raw_pts_b'],
                                                            H.cuda(), W.cuda(), focal.cuda())
                    loss_dict['smooth_loss'] = smooth_loss
                    loss += self.args.smooth_loss_lambda * loss_dict['smooth_loss']

                    # Spatial smooth scene flow. (loss adapted from NSFF)
                    if self.args.no_ndc:
                        sp_smooth_loss = compute_sf_smooth_s_loss_wo_ndc(ret['raw_pts'], ret['raw_pts_f'], H, W, focal) \
                                    + compute_sf_smooth_s_loss_wo_ndc(ret['raw_pts'], ret['raw_pts_b'], H, W, focal)
                    else:
                        sp_smooth_loss = compute_sf_smooth_s_loss(ret['raw_pts'], ret['raw_pts_f'], H.cuda(), W.cuda(), focal.cuda()) \
                                        + compute_sf_smooth_s_loss(ret['raw_pts'], ret['raw_pts_b'], H.cuda(), W.cuda(), focal.cuda())

                    loss_dict['sp_smooth_loss'] = sp_smooth_loss
                    loss += self.args.smooth_loss_lambda * loss_dict['sp_smooth_loss']

                    # Consistency loss.
                    consistency_loss = L1(ret['sceneflow_f'] + ret['sceneflow_f_b']) + \
                                    L1(ret['sceneflow_b'] + ret['sceneflow_b_f'])
                    loss_dict['consistency_loss'] = consistency_loss
                    loss += self.args.consistency_loss_lambda * loss_dict['consistency_loss']

                # Mask loss.
                mask_loss = L1(ret['blending'][batch_masks[:, :, 0].type(torch.bool)]) + \
                            img2mae(ret['dynamicness_map'][..., None], 1 - batch_masks)
                loss_dict['mask_loss'] = mask_loss
                if self.conf.get_bool("info.use_lambda_decay", False):
                    if i < decay_iteration * 1000:
                        loss += self.args.mask_loss_lambda * loss_dict['mask_loss'] 
                else:
                    loss += self.args.mask_loss_lambda * loss_dict['mask_loss']

                # Sparsity loss.
                sparse_weights_d_loss = entropy(ret['weights_d'])
                sparse_blending_loss = entropy(ret['blending'])
                sparse_loss = sparse_weights_d_loss + sparse_blending_loss
                # sparse_loss = entropy(ret['weights_d']) + entropy(ret['blending'])
                loss_dict['sparse_loss'] = sparse_loss
                loss_dict['sparse_weights_d_loss'] = sparse_weights_d_loss
                loss_dict['sparse_blending_loss'] = sparse_blending_loss
                loss += self.args.sparse_loss_lambda * loss_dict['sparse_loss']

                # Dynamic blending loss.
                if self.conf.get_bool('info.use_depth_blending_loss', False):
                    dynamic_reg = 1 - batch_masks
                    depth_blending_loss = compute_depth_blending_loss(
                        batch_invdepths, ret['z_vals'], ret['blending_map'],
                        near=near, far=far,
                        min_depth=batch_invdepths.max(),
                        max_depth=batch_invdepths.min(),
                        M=dynamic_reg, thickness=self.args.blending_thickness
                    )
                    loss_dict['depth_blending_loss'] = depth_blending_loss
                    loss += depth_blending_loss

                # Static flow loss.
                if self.conf.get_bool('info.use_static_flow_loss', False):
                    static_flow_loss = L2(ret['sceneflow_b'][:, :N_rand//2]) + L2(ret['sceneflow_f'][:, :N_rand//2])
                    loss_dict['static_flow_loss'] = static_flow_loss
                    loss += self.conf.get_float('info.static_flow_loss_lambda', 0.5) * loss_dict['static_flow_loss']
                
                # Mask flow loss.
                if self.conf.get_bool('info.use_mask_flow_loss', False):
                    mask_f_loss = L1(ret['blending_f'][batch_masks[:, :, 0].type(torch.bool)]) 
                                    # img2mae(ret['dynamicness_map_f'][..., None], 1 - batch_masks)
                    mask_b_loss = L1(ret['blending_b'][batch_masks[:, :, 0].type(torch.bool)]) 
                                    # img2mae(ret['dynamicness_map_b'][..., None], 1 - batch_masks)
                    loss_dict['mask_flow_loss'] = mask_f_loss + mask_b_loss
                    if self.conf.get_bool("info.use_lambda_decay", False):
                        if i < decay_iteration * 1000:
                            loss += self.conf.get_float('info.mask_flow_loss_lambda', self.args.mask_loss_lambda) * \
                                loss_dict['mask_flow_loss']
                    else:
                        loss += self.conf.get_float('info.mask_flow_loss_lambda', self.args.mask_loss_lambda) * \
                            loss_dict['mask_flow_loss']
                # Depth constraint
                # Depth in NDC space equals to negative disparity in Euclidean space.
                depth_loss = compute_depth_loss(ret['depth_map_d'], -batch_invdepths)
                loss_dict['depth_loss'] = depth_loss
                # loss += self.args.depth_loss_lambda * Temp * loss_dict['depth_loss']
                loss += self.args.depth_loss_lambda * loss_dict['depth_loss']

                # Order loss
                order_loss = torch.mean(torch.square(ret['depth_map_d'][batch_masks[:, :, 0].type(torch.bool)] - \
                                                    ret['depth_map_s'].detach()[batch_masks[:, :, 0].type(torch.bool)]))
                loss_dict['order_loss'] = order_loss
                loss += self.args.order_loss_lambda * loss_dict['order_loss']

                if self.args.no_ndc:
                    sf_smooth_loss = compute_sf_smooth_loss_wo_ndc(ret['raw_pts_b'],
                                                            ret['raw_pts'],
                                                            ret['raw_pts_b_b'],
                                                            H, W, focal) + \
                    compute_sf_smooth_loss_wo_ndc(ret['raw_pts_f'],
                                                            ret['raw_pts_f_f'],
                                                            ret['raw_pts'],
                                                            H, W, focal)
                else:
                    sf_smooth_loss = compute_sf_smooth_loss(ret['raw_pts_b'],
                                                            ret['raw_pts'],
                                                            ret['raw_pts_b_b'],
                                                            H.cuda(), W.cuda(), focal.cuda()) + \
                                    compute_sf_smooth_loss(ret['raw_pts_f'],
                                                            ret['raw_pts_f_f'],
                                                            ret['raw_pts'],
                                                            H.cuda(), W.cuda(), focal.cuda())
                loss_dict['sf_smooth_loss'] = sf_smooth_loss
                loss += self.args.smooth_loss_lambda * loss_dict['sf_smooth_loss']

                if chain_5frames:
                    img_d_b_b_loss = img2mse(ret['rgb_map_d_b_b'], target_rgbs)
                    loss_dict['img_d_b_b_loss'] = img_d_b_b_loss
                    loss += self.args.dynamic_loss_lambda * loss_dict['img_d_b_b_loss']

                    img_d_f_f_loss = img2mse(ret['rgb_map_d_f_f'], target_rgbs)
                    loss_dict['img_d_f_f_loss'] = img_d_f_f_loss
                    loss += self.args.dynamic_loss_lambda * loss_dict['img_d_f_f_loss']

                if torch.isnan(loss).sum() != 0:
                    import ipdb; ipdb.set_trace()
                if self.conf.get_bool("info.change_background", False):
                    print("WARNING: CHANGING BACKGROUND.")
                else:
                    loss.backward()
                if self.conf.get_bool('info.use_clip_grad_norm', False):
                    clip_grad_norm_(self.net.parameters(), max_norm=10, norm_type=2)
                # set_requires_grad(self.net.mlp_static, True)
                self.optimizer.step()

                # Learning rate decay.
                decay_rate = 0.1
                decay_steps = self.args.lrate_decay
                new_lrate = self.args.lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lrate

                dt = time.time() - time0
                # import ipdb; ipdb.set_trace()

                if i % self.args.i_log == 0 and self.rank == 0:
                    self.logger.info(f'Step: {global_step}, Loss: {loss}, Time: {dt}, chain_5frames: {chain_5frames}, self.expname: {self.expname}')
                    loss_details = ''
                    for key in loss_dict.keys():
                        loss_details += f'{key}: ' + '%.3f' %(loss_dict[key].mean().item()) + ' '
                    self.logger.info(loss_details)

                # Rest is logging
                if i % self.args.i_weights==0 and i > 0 and self.rank == 0:
                    path = os.path.join(self.basedir, self.expname, '{:06d}.tar'.format(i))

                    if self.args.N_importance > 0:
                        raise NotImplementedError
                    else:
                        torch.save({
                            'global_step': global_step,
                            'network_fn_d_state_dict': self.render_kwargs_train['network_fn_d'].module.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                        }, path)

                    self.logger.info('Saved weights at'+ str(path))

                if i % self.args.i_video == 0 and self.rank == 0 and i > 0:
                    print("rank ", self.rank, " is in the i_video")

                    time2renders = []
                    pose2renders = []
                    for b in range(images.shape[0]):

                        # Change time and change view at the same time.
                        time2render = np.concatenate((np.repeat((i_train / float(num_img) * 2. - 1.0), 4),
                                                    np.repeat((i_train / float(num_img) * 2. - 1.0)[::-1][1:-1], 4)))
                        if len(time2render) > len(render_poses[b]):
                            pose2render = np.tile(render_poses[b], (int(np.ceil(len(time2render) / len(render_poses[b]))), 1, 1))
                            pose2render = pose2render[:len(time2render)]
                            pose2render = torch.Tensor(pose2render).cuda()
                        else:
                            time2render = np.tile(time2render, int(np.ceil(len(render_poses[b]) / len(time2render))))
                            time2render = time2render[:len(render_poses[b])]
                            pose2render = torch.Tensor(render_poses[b]).cuda()

                        time2renders.append(time2render)
                        pose2renders.append(pose2render)

                    time2renders = np.stack(time2renders, axis=0)
                    pose2renders = torch.stack(pose2renders, dim=0)

                    result_type = 'novelviewtime'

                    testsavedir = os.path.join(
                        self.basedir, self.expname, result_type + '_{:06d}'.format(i))
                    os.makedirs(testsavedir, exist_ok=True)
                    with torch.no_grad():
                        ret = render_path(pose2renders, time2renders,
                                        hwf, self.args.chunk, self.render_kwargs_test, savedir=testsavedir,
                                        feature_dict=feature_dict, near=near, eval_only=True)
                    moviebase = os.path.join(
                        testsavedir, '{}_{}_{:06d}_'.format(self.expname, result_type, i))
                    save_res(moviebase, ret, images.shape[0])

                if i % self.args.i_testset == 0 and self.rank == 0 and i > 0:
                    print("rank ", self.rank, " is in the i_teset_set")
                    # Change view and time.
                    pose2render = poses[...]
                    time2render = i_train / float(num_img) * 2. - 1.0
                    time2render = np.tile(time2render, [poses.shape[0], 1])
                    result_type = 'testset'

                    testsavedir = os.path.join(
                        self.basedir, self.expname, result_type + '_{:06d}'.format(i))
                    os.makedirs(testsavedir, exist_ok=True)
                    with torch.no_grad():
                        ret = render_path(pose2render, time2render,
                                        hwf, self.args.chunk, self.render_kwargs_test, savedir=testsavedir,
                                        flows_gt_f=grids[:, :, :, :, 2:4], flows_gt_b=grids[:, :, :, :, 5:7],
                                        feature_dict=feature_dict, near=near, eval_only=True)
                    moviebase = os.path.join(
                        testsavedir, '{}_{}_{:06d}_'.format(self.expname, result_type, i))
                    save_res(moviebase, ret, images.shape[0])

                    # Fix view (first view) and change time.
                    pose2render = poses[:, 0:1, ...].expand([poses.shape[0], int(num_img), 3, 4])
                    time2render = i_train / float(num_img) * 2. - 1.0
                    time2render = np.tile(time2render, [poses.shape[0], 1])
                    result_type = 'testset_view000'

                    testsavedir = os.path.join(
                        self.basedir, self.expname, result_type + '_{:06d}'.format(i))
                    os.makedirs(testsavedir, exist_ok=True)
                    with torch.no_grad():
                        ret = render_path(pose2render, time2render,
                                        hwf, self.args.chunk, self.render_kwargs_test, savedir=testsavedir,
                                        feature_dict=feature_dict, near=near, eval_only=True)
                    moviebase = os.path.join(
                        testsavedir, '{}_{}_{:06d}_'.format(self.expname, result_type, i))
                    save_res(moviebase, ret, images.shape[0])

                    # Fix time (the first timestamp) and change view.
                    pose2render = poses[...]
                    time2render = np.tile(i_train[0], [int(num_img)]) / float(num_img) * 2. - 1.0
                    time2render = np.tile(time2render, [poses.shape[0], 1])
                    result_type = 'testset_time000'

                    testsavedir = os.path.join(
                        self.basedir, self.expname, result_type + '_{:06d}'.format(i))
                    os.makedirs(testsavedir, exist_ok=True)
                    with torch.no_grad():
                        ret = render_path(pose2render, time2render,
                                        hwf, self.args.chunk, self.render_kwargs_test, savedir=testsavedir,
                                        feature_dict=feature_dict, near=near, eval_only=True)
                    moviebase = os.path.join(
                        testsavedir, '{}_{}_{:06d}_'.format(self.expname, result_type, i))
                    save_res(moviebase, ret, images.shape[0])

                if i % self.args.i_testset_all == 0 and self.rank == 0 and i > 0 and self.conf.get_bool("info.testset_all", False):
                    evaldir = conf['info.evaldir']
                    print("rank ", self.rank, " is in the i_testset_all")
                    for each in range(len(i_train)):
                        pose2render = poses[...]
                        time2render = np.tile(i_train[each], [int(num_img)]) / float(num_img) * 2. - 1.0
                        time2render = np.tile(time2render, [poses.shape[0], 1])
                        result_type = '{:08d}'.format(each + 1)

                        testsavedir = os.path.join(
                            self.basedir, self.expname, evaldir, 'step_{:06d}'.format(global_step), result_type)
                        os.makedirs(testsavedir, exist_ok=True)
                        with torch.no_grad():
                            ret = render_path(pose2render, time2render,
                                            hwf, self.args.chunk, self.render_kwargs_test, savedir=testsavedir,
                                            feature_dict=feature_dict, near=near, eval_only=True)
                        moviebase = os.path.join(
                            testsavedir, '{}_{}_{:06d}_'.format(self.expname, result_type, global_step))
                        save_res(moviebase, ret, images.shape[0])

                if i % self.args.i_print == 0 and self.rank == 0:
                    self.writer.add_scalar('loss', loss.item(), i)
                    self.writer.add_scalar('lr', new_lrate, i)
                    self.writer.add_scalar('Temp', Temp, i)
                    for loss_key in loss_dict:
                        self.writer.add_scalar(loss_key, loss_dict[loss_key].item(), i)

                if i % self.args.i_img == 0 and self.rank == 0:
                    # Log a rendered training view to Tensorboard.
                    # img_i = np.random.choice(i_train[1:-1])
                    select_bs = list(range(len(img_ies)))
                    target = images[select_bs, img_ies]
                    pose = poses[select_bs, img_ies, :3, :4]
                    mask = masks[select_bs, img_ies]
                    grid = grids[select_bs, img_ies]
                    invdepth = invdepths[select_bs, img_ies]
                    flow_f_img, flow_b_img = [], []
                    for b in range(len(img_ies)):
                        curr_flow_f_img = flow_to_image(grid[b, ..., 2:4].cpu().numpy())
                        curr_flow_b_img = flow_to_image(grid[b, ..., 5:7].cpu().numpy())
                        flow_f_img.append(curr_flow_f_img)
                        flow_b_img.append(curr_flow_b_img)
                    flow_f_img = np.stack(flow_f_img, axis=0)
                    flow_b_img = np.stack(flow_b_img, axis=0)

                    with torch.no_grad():
                        ret = render(t,
                                    False,
                                    H, W, focal.cuda(),
                                    chunk=1024*64,
                                    c2w=pose,
                                    feature_dict=feature_dict,
                                    **self.render_kwargs_test)

                        for b in range(len(img_ies)):
                            img_i = img_ies[b]

                            # The last frame does not have forward flow.
                            pose_f = poses[b, min(img_i + 1, int(num_img) - 1), :3, :4]
                            if self.args.no_ndc:
                                induced_flow_f = induce_flow_wo_ndc(H[b].item(), W[b].item(), focal[b].item(), pose_f, ret['weights_d'][b], ret['raw_pts_f'][b], grid[b, ..., :2])
                            else:
                                induced_flow_f = induce_flow(H[b], W[b], focal[b], pose_f.cpu(), ret['weights_d'][b], ret['raw_pts_f'][b], grid[b, ..., :2].cpu())

                            # The first frame does not have backward flow.
                            pose_b = poses[b, max(img_i - 1, 0), :3, :4]
                            if self.args.no_ndc:
                                induced_flow_b = induce_flow_wo_ndc(H[b].item(), W[b].item(), focal[b].item(), pose_b, ret['weights_d'][b], ret['raw_pts_b'][b], grid[b, ..., :2])
                            else:
                                induced_flow_b = induce_flow(H[b], W[b], focal[b], pose_b.cpu(), ret['weights_d'][b], ret['raw_pts_b'][b], grid[b, ..., :2].cpu())

                            induced_flow_f_img = flow_to_image(induced_flow_f.cpu().numpy())
                            induced_flow_b_img = flow_to_image(induced_flow_b.cpu().numpy())

                            psnr = mse2psnr(img2mse(ret['rgb_map_full'][b], target[b].cpu()))

                            # Save out the validation image for Tensorboard-free monitoring
                            testimgdir = os.path.join(self.basedir, self.expname, 'tboard_val_imgs')
                            if i == 0:
                                os.makedirs(testimgdir, exist_ok=True)
                            imageio.imwrite(os.path.join(testimgdir, '{:06d}_{:02d}.png'.format(i, b)), to8b(ret['rgb_map_full'][b].cpu().numpy()))

                            self.writer.add_scalar('psnr_holdout_{}'.format(b), psnr.item(), i)
                            self.writer.add_image('rgb_holdout_{}'.format(b), target[b], global_step=i, dataformats='HWC')
                            self.writer.add_image('mask_{}'.format(b), mask[b], global_step=i, dataformats='HW')
                            self.writer.add_image('disp_{}'.format(b), torch.clamp(invdepth[b] / percentile(invdepth[b], 97), 0., 1.), global_step=i, dataformats='HW')

                            self.writer.add_image('rgb_{}'.format(b), torch.clamp(ret['rgb_map_full'][b], 0., 1.), global_step=i, dataformats='HWC')
                            self.writer.add_image('depth_{}'.format(b), normalize_depth(ret['depth_map_full'][b], near), global_step=i, dataformats='HW')
                            self.writer.add_image('acc_{}'.format(b), ret['acc_map_full'][b], global_step=i, dataformats='HW')

                            self.writer.add_image('rgb_s_{}'.format(b), torch.clamp(ret['rgb_map_s'][b], 0., 1.), global_step=i, dataformats='HWC')
                            self.writer.add_image('depth_s_{}'.format(b), normalize_depth(ret['depth_map_s'][b], near), global_step=i, dataformats='HW')
                            self.writer.add_image('acc_s_{}'.format(b), ret['acc_map_s'][b], global_step=i, dataformats='HW')

                            self.writer.add_image('rgb_d_{}'.format(b), torch.clamp(ret['rgb_map_d'][b], 0., 1.), global_step=i, dataformats='HWC')
                            self.writer.add_image('depth_d_{}'.format(b), normalize_depth(ret['depth_map_d'][b], near), global_step=i, dataformats='HW')
                            self.writer.add_image('acc_d_{}'.format(b), ret['acc_map_d'][b], global_step=i, dataformats='HW')

                            self.writer.add_image('induced_flow_f_{}'.format(b), induced_flow_f_img, global_step=i, dataformats='HWC')
                            self.writer.add_image('induced_flow_b_{}'.format(b), induced_flow_b_img, global_step=i, dataformats='HWC')
                            self.writer.add_image('flow_f_gt_{}'.format(b), flow_f_img[b], global_step=i, dataformats='HWC')
                            self.writer.add_image('flow_b_gt_{}'.format(b), flow_b_img[b], global_step=i, dataformats='HWC')

                            self.writer.add_image('dynamicness_{}'.format(b), ret['dynamicness_map'][b], global_step=i, dataformats='HW')

                    self.render_kwargs_test['network_fn_d'].module.batch = None
                    self.render_kwargs_test['network_fn_d'].module.select_batch = False
                # dist.barrier()
                del ret
                global_step += 1

        return



trainer = MonoNeRFTrainer(args, conf)
trainer.start_train()
