"""
Main model implementation
"""
import os
import os.path as osp
from socket import TIPC_DEST_DROPPABLE
import warnings

import torch
import torch.autograd.profiler as profiler
import torch.nn.functional as F

from src.util import repeat_interleave

from .encoder import ImageEncoder
from .model_util import make_encoder, make_mlp

from .resnetfc import ResnetBlockFC
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def NDC2world(pts, H, W, f):

    # NDC coordinate to world coordinate
    pts_z = 2 / (torch.clamp(pts[..., 2:], min=-1., max=1-1e-3) - 1)
    pts_x = - pts[..., 0:1] * pts_z * W / 2 / f[:, None, None]
    pts_y = - pts[..., 1:2] * pts_z * H / 2 / f[:, None, None]
    pts_world = torch.cat([pts_x, pts_y, pts_z], -1)

    return pts_world

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.
    Space such that the canvas is a cube with sides [-1, 1] in each axis.
    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.
    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
    (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


class PixelNeRFNet(torch.nn.Module):
    def __init__(self, conf, stop_encoder_grad=True, use_static_resnet=False):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        self.encoder = make_encoder(conf['encoder'])
        self.use_encoder2d = conf['encoder'].get_bool('use_encoder_2d', False)
        if self.use_encoder2d:
            print('Using sparate Encoders for static and dynamic NeRF.')
            self.encoder2d = make_encoder(conf['encoder_2d'])
        self.use_encoder = conf.get_bool('use_encoder', False)  # Image features?

        self.use_xyz = conf.get_bool('use_xyz', True)
        self.use_ndc = conf.get_bool('use_ndc', True)

        assert self.use_encoder or self.use_xyz  # Must use some feature..

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will
        # be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = conf.get_bool('normalize_z', True)

        self.stop_encoder_grad = (
            stop_encoder_grad  # Stop ConvNet gradient (freeze weights)
        )
        self.use_code = conf.get_bool('use_code', False)  # Positional encoding
        self.use_code_viewdirs = conf.get_bool(
            'use_code_viewdirs', True
        )  # Positional encoding applies to viewdirs

        # Enable view directions
        self.use_viewdirs = conf.get_bool('use_viewdirs', False)

        # Global image features?
        self.use_global_encoder = conf.get_bool('use_global_encoder', False)

        # Enable resnet static nerf
        self.use_static_resnet = use_static_resnet


        d_latent = self.encoder.latent_size if self.use_encoder else 0
        print('used latent size: ', d_latent)
        d_in = 3 if self.use_xyz else 1

        if self.use_viewdirs and self.use_code_viewdirs:
            # Apply positional encoding to viewdirs
            d_in += 3

        if self.use_viewdirs and not self.use_code_viewdirs:
            # Don't apply positional encoding to viewdirs (concat after encoded)
            d_in += 3

        if self.use_global_encoder:
            # Global image feature
            self.global_encoder = ImageEncoder.from_conf(conf['global_encoder'])
            self.global_latent_size = self.global_encoder.latent_size
            d_latent += self.global_latent_size

        d_out = 4
        ##### Warning!!!!!!!!!!!!!!!!!!!! ########
        ##### d_in does not work in building model.
        d_in = 84 + 27 # rewrite in resnetfc.py
        self.latent_size = self.encoder.latent_size
        self.flow_mode = conf.get_string('mlp_dynamic.flow_mode', "discrete")
        if self.flow_mode == "discrete":
            self.discrete_steps = conf.get_int('mlp_dynamic.discrete_steps', 1)
        self.mlp_dynamic = make_mlp(conf['mlp_dynamic'], d_in, d_latent, d_out=d_out, add_features=self.use_encoder)
        if self.use_static_resnet:
            if self.use_encoder2d:
                d_latent2d = self.encoder2d.latent_size
                print('used 2d latent size: ', d_latent2d)
            else:
                d_latent2d = d_latent
            self.mlp_static = make_mlp(
            conf['mlp_static'], d_in=63+27, d_latent=d_latent2d, d_out=d_out, add_features=self.use_encoder
        )
        else:
            self.mlp_static = None
        self.random_static_frame = conf.get_bool('random_static_frame', True)
        
        self.flow_net = torch.nn.Sequential(
            self.mlp_dynamic.lin_in,
            torch.nn.ReLU(),   
        )
        self.blocks = self.mlp_dynamic.global_blocks
        self.global_blocks = self.mlp_dynamic.global_blocks
        self.lin_global_z = self.mlp_dynamic.lin_global_z

        for i in range(len(self.blocks)):
            self.flow_net.add_module('block{}'.format(i), self.blocks[i])     
        

        self.flow_fc = self.mlp_dynamic.sf_linear
        self.use_feat_guided_flow = conf.get_bool("feat_guided_flow", True)

        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_latent
        self.d_latent2d = d_latent2d

        self.num_objs = 0
        self.num_views_per_obj = 1
        self.masks = None

        self.pretrain = False
        self.unseen_frame_rendering = conf.get_bool('unseen_frame_rendering', False)
        if self.unseen_frame_rendering:
            self.training_frames = conf.get_int('training_frames')

    def encode(self, images, poses, focal, z_bounds=None, c=None, images2d=None, mode="encode"):
        """
        :param images (NS, 3, H, W)
        NS is number of input (aka source or reference) views
        :param poses (NS, 4, 4)
        :param focal focal length () or (2) or (NS) or (NS, 2) [fx, fy]
        :param z_bounds ignored argument (used in the past)
        :param c principal point None or () or (2) or (NS) or (NS, 2) [cx, cy],
        default is center of image
        """
        # self.num_objs = images.size(0)
        num_imgs = images.shape[-3]
        image_shape = torch.empty(2)
        num_views_per_obj = 1
        if len(images.shape) == 6:
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(
                1
            )  # Be consistent with NS = num input views
            num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            num_views_per_obj = 1

        latent_dict = self.encoder(images)
        if self.use_encoder2d:
            latent2d_dict = self.encoder2d(images2d)
        else:
            latent2d_dict = latent_dict
        rot = poses[:, :, :3, :3].transpose(2, 3).contiguous()  # (B, 3, 3)
        trans = -torch.matmul(rot, poses[:, :, :3, 3:])  # (B, 3, 1)
        poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        #! ray coords is W * H not H * W)
        image_shape[0] = images.shape[-2]
        image_shape[1] = images.shape[-1]

        # Handle various focal length/principal point formats
        if len(focal.shape) == 2:
            # Scalar: fx = fy = value for all views
            focal = focal[:, :, None].repeat((1, 1, 2))
        else:
            raise ValueError('focal dimension error!')
        # elif len(focal.shape) == 2:
        #     # Vector f: fx = fy = f_i *for view i*
        #     # Length should match NS (or 1 for broadcast)
        #     focal = focal.unsqueeze(-1).repeat((1, 2))
        # else:
        #     focal = focal.clone()
        # focal = torch.Tensor(focal.float())
        focal[..., 1] *= -1.0
        if c is None:
            # Default principal point is center of image
            c = (image_shape * 0.5).unsqueeze(0).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))

        if self.use_global_encoder:
            self.global_encoder(images)

        self.select_batch = False
        self.batch = None

        feature_dict = dict(
            poses = poses,
            c = c,
            focal = focal,
            latent_dict = latent_dict,
            latent2d_dict = latent2d_dict,
            image_shape = image_shape,
            num_imgs = num_imgs,
            num_views_per_obj = num_views_per_obj,
        )

        return feature_dict

    def train_step(self, *inputs, **kwargs):

        # raise NotImplementedError("Need define each forward.")

        func = getattr(self, kwargs["mode"])
        return func(*inputs, **kwargs)


    def load_masks(self, masks):
        """
        Load static and dynamic mask.
        """
        self.masks = masks
        self.coords_d = torch.stack((torch.where(masks < 0.5)), -1)
        self.coords_s = torch.stack((torch.where(masks >= 0.5)), -1)
        self.coords = torch.stack((torch.where(masks > -1)), -1)

    def forward(self, xyzt_ndc, ndc_wo_embed, use_network='dynamic', feature_dict=None, coarse=True, viewdirs=None, far=False, mode="forward", chain_5frames=False):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
        with profiler.record_function('model_inference'):
            xyz, t_ = ndc_wo_embed[..., :3], ndc_wo_embed[..., 3]
            t_ = (t_ + 1.0) / 2 * feature_dict["num_imgs"]
            t = torch.ceil(t_-0.5).int()
            SB, num_rays, _ = xyz.shape
            xyz = xyz.reshape(*xyzt_ndc.shape[0: 2], xyz.shape[-1])
            NS = feature_dict["num_views_per_obj"]

            W, H = feature_dict["image_shape"]
            f = torch.abs(feature_dict["focal"][:, 0, 0])
            # H = 270; W = 480; f=418.9622; near=1.0

            # Transfer NDC to world.
            xyz_ndc = xyz
            if self.use_ndc:
                xyz = NDC2world(xyz, H, W, f)

            # Transform query points into the camera spaces of the input views
            xyz = repeat_interleave(xyz, NS)  # (SB*NS, B, 3)
            xyz_world = xyz

            if 'dynamic' in use_network:
                ode_latent = self.mlp_dynamic.ode_pool(feature_dict['latent_dict']['flow_latent']).reshape([SB, -1])
                ode_latent = self.mlp_dynamic.ode_fc(ode_latent)
                if not self.use_feat_guided_flow:
                    ode_latent = torch.zeros_like(ode_latent)
                if self.flow_mode == "discrete":
                    global_z = ode_latent.unsqueeze(1).repeat(1, xyzt_ndc.shape[1], 1)
                    xyz_forward = xyz_ndc.clone()
                    xyz_backward = xyz_ndc.clone()
                    forward_flow = 0
                    backward_flow = 0
                    steps = self.discrete_steps
                    for i in range(steps):
                        curr_forward_t = (t_+(i/steps)) / float(feature_dict['num_imgs']) * 2 -1
                        curr_backward_t = (t_-(i/steps)) / float(feature_dict['num_imgs']) * 2 -1
                        
                        curr_forward_xyzt = self.mlp_dynamic.vector_field.embed_fn_d(torch.cat([xyz_forward, curr_forward_t[..., None]], dim=-1))
                        curr_backward_xyzt = self.mlp_dynamic.vector_field.embed_fn_d(torch.cat([xyz_backward, curr_backward_t[..., None]], dim=-1))

                        curr_forward_xyzt = F.relu(self.mlp_dynamic.lin_in(F.normalize(curr_forward_xyzt, dim=-1)))
                        curr_backward_xyzt = F.relu(self.mlp_dynamic.lin_in(F.normalize(curr_backward_xyzt, dim=-1)))

                        for blkid in range(self.mlp_dynamic.n_global_blocks):
                            if True:
                                # tz = self.lin_global_z[blkid](global_z)
                                tz = global_z
                                if self.mlp_dynamic.origin_pipeline:
                                    tz = torch.zeros_like(tz)
                                # curr_forward_xyzt = curr_forward_xyzt + tz
                                # curr_backward_xyzt = curr_backward_xyzt + tz
                                curr_forward_xyzt = self.lin_global_z[blkid](torch.cat([curr_forward_xyzt, tz], dim=-1))
                                curr_forward_xyzt = self.lin_global_z[blkid](torch.cat([curr_backward_xyzt, tz], dim=-1))

                                # curr_forward_xyzt = F.normalize(curr_forward_xyzt, dim=-1)
                                # curr_backward_xyzt = F.normalize(curr_backward_xyzt, dim=-1)

                            curr_forward_xyzt = self.global_blocks[blkid](curr_forward_xyzt)
                            curr_backward_xyzt = self.global_blocks[blkid](curr_backward_xyzt)
                        forward_flow = torch.tanh(self.flow_fc(curr_forward_xyzt))
                        backward_flow = -torch.tanh(self.flow_fc(curr_backward_xyzt))

                        xyz_forward += forward_flow /steps
                        xyz_backward += backward_flow /steps
                    sf = torch.cat([xyz_backward, xyz_forward], dim=-1).unsqueeze(1)
                    if chain_5frames:
                        xyz_forward_forward = xyz_forward
                        xyz_backward_backward = xyz_backward
                        for i in range(steps):
                            curr_forward_t = (t_+ 1 +(i/steps)) / float(feature_dict['num_imgs']) * 2 -1
                            curr_backward_t = (t_- 1 -(i/steps)) / float(feature_dict['num_imgs']) * 2 -1
                            
                            curr_forward_xyzt = self.mlp_dynamic.vector_field.embed_fn_d(torch.cat([xyz_forward_forward, curr_forward_t[..., None]], dim=-1))
                            curr_backward_xyzt = self.mlp_dynamic.vector_field.embed_fn_d(torch.cat([xyz_backward_backward, curr_backward_t[..., None]], dim=-1))

                            curr_forward_xyzt = F.relu(self.mlp_dynamic.lin_in(F.normalize(curr_forward_xyzt, dim=-1)))
                            curr_backward_xyzt = F.relu(self.mlp_dynamic.lin_in(F.normalize(curr_backward_xyzt, dim=-1)))

                            for blkid in range(self.mlp_dynamic.n_global_blocks):
                                tz = global_z
                                if self.mlp_dynamic.origin_pipeline:
                                    tz = torch.zeros_list(tz)
                              
                                curr_forward_xyzt = self.lin_global_z[blkid](torch.cat([curr_forward_xyzt, tz], dim=-1))
                                curr_forward_xyzt = self.lin_global_z[blkid](torch.cat([curr_backward_xyzt, tz], dim=-1))

                                curr_forward_xyzt = self.global_blocks[blkid](curr_forward_xyzt)
                                curr_backward_xyzt = self.global_blocks[blkid](curr_backward_xyzt)
                            forward_flow = torch.tanh(self.flow_fc(curr_forward_xyzt, dim=-1))
                            backward_flow = -torch.tanh(self.flow_fc(curr_backward_xyzt, dim=-1))

                            xyz_forward_forward += forward_flow /steps
                            xyz_backward_backward += backward_flow /steps
                        sf_5frames = torch.cat([xyz_backward_backward, xyz_forward_forward], dim=-1).unsqueeze(1)
                        sf = torch.cat([sf, sf_5frames], dim=1)
                    sf_dict = dict(sf = sf[:, 0, ...])

                elif self.flow_mode == "ode":
                    norm_t = t_
                    norm_t_f = [1 / feature_dict["num_imgs"]]
                    norm_t_b = [1 / feature_dict["num_imgs"]]                    
                    if chain_5frames:
                        norm_t_f.append(2 / feature_dict["num_imgs"])
                        norm_t_b.append(2 / feature_dict["num_imgs"])
                    norm_t_f = torch.tensor(norm_t_f, device=device)[None, :].repeat(SB, 1)
                    norm_t_b = torch.tensor(norm_t_b, device=device)[None, :].repeat(SB, 1)

                    xyz_forward = self.mlp_dynamic.transform_from_t1_to_t2(norm_t_f, xyz_ndc, z=torch.empty([xyz_world.shape[0], 0]), c_t=ode_latent, t_batch=norm_t)
                    xyz_backward = self.mlp_dynamic.transform_from_t1_to_t2(norm_t_f, xyz_ndc, z=torch.empty([xyz_world.shape[0], 0]), c_t=ode_latent, t_batch=norm_t, invert=True)
                    sf = torch.cat([xyz_backward[:, :, :, :3], xyz_forward[:, :, :, :3]], dim=-1)
                    sf_dict = dict(sf = sf[:, 0, ...])
                    xyz_forward, xyz_backward = xyz_forward[:, 0, ...], xyz_backward[:, 0, ...]

            else:
                sf_dict = None

            #! suppose all rays are from the same image in the same batch.
            t = t[:, 0]

            #! for upper and lower bounds
            t = torch.clamp(t, 0, feature_dict["poses"].shape[1] - 1)
            t = t.cpu().numpy()

            if 'dynamic' not in use_network and self.random_static_frame:
                t = torch.randint(0, feature_dict["poses"].shape[1], (t.shape))
                # if self.unseen_frame_rendering:
                #     t = torch.randint(0, self.training_frames, (t.shape))

            if self.select_batch:
                idx = self.batch
            else:
                idx = torch.linspace(0, t.shape[0]-1, t.shape[0]).int().numpy()
            xyz_rot = torch.matmul(feature_dict["poses"][idx, t, None, :3, :3], xyz.unsqueeze(-1))[
                ..., 0
            ]
            xyz = xyz_rot + feature_dict["poses"][idx, t, None, :3, 3]

            if self.use_encoder:
                # Grab encoder's latent code.
                uv = -xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
                if self.select_batch:
                    uv *= repeat_interleave(
                        feature_dict["focal"][self.batch], NS if feature_dict["focal"].shape[0] > 1 else 1
                    )
                    uv += repeat_interleave(
                        feature_dict["c"], NS if feature_dict["c"].shape[0] > 1 else 1
                    )  # (SB*NS, B, 2)

                    # latent = self.encoder.index(
                    #     uv, t, None, self.image_shape, self.batch
                    # )  # (SB * NS, latent, B)
                else:
                    uv *= repeat_interleave(
                        feature_dict["focal"], NS if feature_dict["focal"].shape[0] > 1 else 1
                    )
                    uv += repeat_interleave(
                        feature_dict["c"], NS if feature_dict["c"].shape[0] > 1 else 1
                    )  # (SB*NS, B, 2)
                    self.batch = None

                latent_curr = self.encoder.index(
                    uv, t, None, feature_dict["image_shape"], batch=self.batch, latent_dict=feature_dict["latent_dict"],
                )
                latent = latent_curr
                if self.mlp_dynamic.use_temporal_feature and 'dynamic' in use_network:
                    forward_t =  np.clip(t+1, 0, feature_dict["poses"].shape[1] - 1)
                    backward_t =  np.clip(t-1, 0, feature_dict["poses"].shape[1] - 1)

                    # Forward Samling
                    xyz_forward_rot = torch.matmul(feature_dict["poses"][idx, forward_t, None, :3, :3], xyz_forward.unsqueeze(-1))[
                        ..., 0
                    ]
                    xyz_forward = xyz_forward_rot + feature_dict['poses'][idx, forward_t, None, :3, 3]
                    uv_forward = -xyz_forward[:, :, :2] / xyz_forward[:, :, 2:]
                    uv_forward *= repeat_interleave(
                    feature_dict["focal"], NS if feature_dict["focal"].shape[0] > 1 else 1
                    )
                    uv_forward += repeat_interleave(
                        feature_dict["c"], NS if feature_dict["c"].shape[0] > 1 else 1
                    )
                    latent_forward = self.encoder.index(
                        uv_forward, forward_t, None, feature_dict["image_shape"], batch=self.batch, latent_dict=feature_dict["latent_dict"],
                    )

                    # Backward Sampling
                    xyz_backward_rot = torch.matmul(feature_dict["poses"][idx, backward_t, None, :3, :3], xyz_backward.unsqueeze(-1))[
                        ..., 0
                    ]
                    xyz_backward = xyz_backward_rot + feature_dict['poses'][idx, backward_t, None, :3, 3]
                    uv_backward = -xyz_backward[:, :, :2] / xyz_backward[:, :, 2:]
                    uv_backward *= repeat_interleave(
                        feature_dict["focal"], NS if feature_dict["focal"].shape[0] > 1 else 1
                    )
                    uv_backward += repeat_interleave(
                    feature_dict["c"], NS if feature_dict["c"].shape[0] > 1 else 1
                    )
                    latent_backward = self.encoder.index(
                        uv_backward, backward_t, None, feature_dict["image_shape"], batch=self.batch, latent_dict=feature_dict["latent_dict"],
                    )


                    for i in range(len(t)):
                        if t[i] + 1 >= feature_dict["num_imgs"]:
                            latent_forward[i] = latent_curr[i]
                        if t[i] - 1 < 0:
                            latent_backward[i] = latent_curr[i]



                    latent_temporal = torch.stack([latent_backward, latent_forward], dim=-1)
                    latent_temporal = latent_temporal.permute(0, 1, 3, 2).reshape(
                        SB, -1, 2, self.latent_size
                    ).contiguous()
                    if self.stop_encoder_grad:
                        latent_temporal = latent_temporal.detach()
                else:
                    latent_temporal = None

                if self.stop_encoder_grad:
                    latent = latent.detach()
                latent = latent.transpose(1, 2).reshape(
                    SB, -1, self.latent_size
                ).contiguous()  # (SB * NS * B, latent)

                if self.use_encoder2d:
                    # t_2d = torch.randint(0, feature_dict["poses"].shape[1], (t.shape))
                    t_2d = t
                    latent2d = self.encoder2d.index(
                        uv, t_2d, None, feature_dict["image_shape"], batch=self.batch, latent_dict=feature_dict["latent2d_dict"],
                    )
                    latent2d = latent2d.transpose(1, 2).reshape(
                        SB, -1, self.d_latent2d
                    ).contiguous()  # (SB * NS * B, latent)
                if 'dynamic' in use_network:
                    if self.d_in == 0:
                        # z_feature not needed
                        mlp_input = latent
                    else:
                        mlp_input = torch.cat([xyzt_ndc, latent, ode_latent.unsqueeze(1).repeat(1, latent.shape[1], 1)], dim=-1)

            if (self.mlp_static.pixelnerf_mode or self.mlp_static.add_features) and 'static' in use_network:
                input_pts, input_views = torch.split(xyzt_ndc, [self.mlp_dynamic.input_ch, self.mlp_dynamic.input_ch_views], dim=-1)
                xyz_embeded = input_pts.reshape(*input_pts.shape[:2], -1, 4)[..., :3]
                if self.use_encoder2d:
                    static_input = torch.cat((xyz_embeded.reshape(*xyz_embeded.shape[:2], -1), input_views, latent2d), dim=-1)
                else:
                    static_input = torch.cat((xyz_embeded.reshape(*xyz_embeded.shape[:2], -1), input_views, latent), dim=-1)
                static_output = self.mlp_static(static_input)
            else:
                static_output = None


            if 'dynamic' in use_network:
                mlp_output = self.mlp_dynamic(mlp_input, sf_dict=sf_dict, latent_temporal=latent_temporal)
                if chain_5frames:
                    # corr = torch.cat([backward_flow[:, 1, :, :3], forward_flow[:, 1, :, :3]], dim=-1)
                    mlp_output = torch.cat([mlp_output, sf[:, 1, ...]], dim=-1)
            else:
                mlp_output = None



        if self.mlp_static.pixelnerf_mode or self.mlp_static.add_features:
            rs = {}
            if mlp_output is not None:
                rs["dynamic"] = mlp_output
            if static_output is not None:
                rs["static"] = static_output 
            return rs
        return mlp_output

    def load_weights(self, args, opt_init=False, strict=True, device=None):
        """
        Helper for loading weights according to argparse arguments.
        Your can put a checkpoint at checkpoints/<exp>/pixel_nerf_init to use as initialization.
        :param opt_init if true, loads from init checkpoint instead of usual even when resuming
        """
        # TODO: make backups
        if opt_init and not args.resume:
            return
        ckpt_name = (
            'pixel_nerf_init' if opt_init or not args.resume else 'pixel_nerf_latest'
        )
        model_path = '%s/%s/%s' % (args.checkpoints_path, args.name, ckpt_name)

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print('Load', model_path)
            self.load_state_dict(
                torch.load(model_path, map_location=device), strict=strict
            )
        elif not opt_init:
            warnings.warn(
                (
                    'WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n'
                    + "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                    + 'If training, unless you are startin a new experiment, please remember to pass --resume.'
                ).format(model_path)
            )
        return self

    def save_weights(self, args, opt_init=False):
        """
        Helper for saving weights according to argparse arguments
        :param opt_init if true, saves from init checkpoint instead of usual
        """
        from shutil import copyfile

        ckpt_name = 'pixel_nerf_init' if opt_init else 'pixel_nerf_latest'
        backup_name = 'pixel_nerf_init_backup' if opt_init else 'pixel_nerf_backup'

        ckpt_path = osp.join(args.checkpoints_path, args.name, ckpt_name)
        ckpt_backup_path = osp.join(args.checkpoints_path, args.name, backup_name)

        if osp.exists(ckpt_path):
            copyfile(ckpt_path, ckpt_backup_path)
        torch.save(self.state_dict(), ckpt_path)
        return self
