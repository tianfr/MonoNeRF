import torch
#  import torch_scatter
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from torch import nn

from src import util
from src.model.velocity_field import VelocityField
from torchdiffeq import odeint, odeint_adjoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def worldpts2NDCpts(pts, H=270., W=480., f=418.9622):
    """Convert world points to NDC points.

    Args:
        pts (array of shape [..., 3]): World coordinates.
        H (float, optional): Height in pixels. Defaults to 270..
        W (float, optional): Width in pixels. Defaults to 480..
        f (float, optional): Focal length of pinhole camera. Defaults to 418.9622.

    Returns:
        ndc: NDC coordinates of shape [..., 3].
    """
    #! near 1.0 far inf.
    ax = -2 * f / W
    ay = -2 * f / H
    az = 1
    bz = 2
    worldx, worldy, worldz = torch.split(pts, [1, 1, 1], dim=-1)
    ndcx = ax * worldx / worldz
    ndcy = ay * worldy / worldz
    ndcz = az + bz / worldz

    return torch.cat([ndcx, ndcy, ndcz], dim=-1)

# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0, use_layer_norm=False, use_kaiming_init=False):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(self.size_h)
            # self.norm1 = lambda x: F.normalize(x, dim=-1)
            self.norm2 = nn.LayerNorm(self.size_h)
            # self.norm2 = lambda x: F.normalize(x, dim=-1)
        else:
            self.norm1 = self.norm2 = lambda x: x

        # Init
        self.use_kaiming_init = use_kaiming_init
        if self.use_kaiming_init:
            nn.init.constant_(self.fc_0.bias, 0.0)
            nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
            nn.init.constant_(self.fc_1.bias, 0.0)
            nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            if self.use_kaiming_init:
                nn.init.constant_(self.shortcut.bias, 0.0)
                nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        with profiler.record_function('resblock'):
            net = self.fc_0(self.activation(self.norm1(x)))
            dx = self.fc_1(self.activation(self.norm2(net)))

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx


class ResnetFC(nn.Module):
    def __init__(
        self,
        d_in,
        d_out=4,
        n_blocks=3,
        n_global_blocks=0,
        d_latent=2048,
        d_global_latent=256,
        d_hidden=128,
        beta=0.0,
        combine_layer=1000,
        combine_type='average',
        use_spade=False,
        # D=8,
        # W=256,
        input_ch=84,
        input_ch_views=27,
        output_ch=4,
        # skips=[4],
        use_viewdirsDyn=True,
        add_features=False,
        pixelnerf_mode=False,
        feature_fuse_mode="addition",
        use_temporal_feature=False,
        temporal_propotion=1.0,
        use_sf_nonlinear=False,
        origin_pipeline=False,
        flow_guided_feat=True,
        use_layer_norm=False,
        flow_mode="discrete",
        discrete_steps=1,
        debug=False,
        vector_field=VelocityField,
        ode_step_size=None,
        use_adjoint=False,
        rtol=0.01, # relative error. error cannot be larger than rtoal * max(y_n, y_n+1)       
        # rtol=0.001,
        atol=0.001, # absolute error. error cannot be larger than atol.
        # atol=0.00001,
        ode_solver='dopri5',
        # ode_solver='rk4',
        p0_z=None,
        c_dim=256,
    ):
        """
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param d_latent latent size, added in each resnet block (0 = disable)
        :param d_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        """
        super().__init__()
        # if use_viewdirsDyn:
        #     d_in = input_ch
        # else:
        #     d_in = input_ch + input_ch_views
        d_in = input_ch
        d_out = output_ch


        self.n_blocks = n_blocks
        self.n_global_blocks = n_global_blocks
        self.d_latent = d_latent
        self.d_global_latent = d_global_latent
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden

        self.combine_layer = combine_layer
        self.combine_type = combine_type
        self.use_spade = use_spade

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        # self.skips = skips
        self.use_viewdirsDyn = use_viewdirsDyn
        self.add_features = add_features
        self.use_sf_nonlinear = use_sf_nonlinear
        self.pixelnerf_mode = pixelnerf_mode
        self.feature_fuse_mode = feature_fuse_mode
        self.use_temporal_feature = use_temporal_feature
        self.use_layer_norm = use_layer_norm
        if self.pixelnerf_mode:
            self.add_features = False
            print('PixelNeRF mode.')
        # import ipdb; ipdb.set_trace()
        if self.add_features and not self.pixelnerf_mode:
            if d_latent == 0:
                raise ValueError('d_latent is ZERO!')
            self.feature_embedding = nn.Linear(d_latent, 128)
            self.d_in += 128
        else:
            self.feature_embedding = None

        self.global_blocks = nn.ModuleList(
            [ResnetBlockFC(d_hidden, beta=beta, use_layer_norm=use_layer_norm, use_kaiming_init=False) for i in range(n_global_blocks)]
        )

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(d_hidden, beta=beta, use_layer_norm=use_layer_norm, use_kaiming_init=False) for i in range(n_blocks)]
        )



        if d_in > 0:
            self.lin_in = nn.Linear(self.d_in, d_hidden)
            # nn.init.constant_(self.lin_in.bias, 0.0)
            # nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")


        if self.use_viewdirsDyn:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + d_hidden, d_hidden//2)])
            self.feature_linear = nn.Linear(d_hidden, d_hidden)
            self.alpha_linear = nn.Linear(d_hidden, 1)
            self.rgb_linear = nn.Linear(d_hidden//2, 3)
            # nn.init.constant_(self.views_linears[0].bias, 0.0)
            # nn.init.kaiming_normal_(self.views_linears[0].weight, a=0, mode="fan_in")
            # nn.init.constant_(self.feature_linear.bias, 0.0)
            # nn.init.kaiming_normal_(self.feature_linear.weight, a=0, mode="fan_in")
            # nn.init.constant_(self.alpha_linear.bias, 0.0)
            # nn.init.kaiming_normal_(self.alpha_linear.weight, a=0, mode="fan_in")
            # nn.init.constant_(self.rgb_linear.bias, 0.0)
            # nn.init.kaiming_normal_(self.rgb_linear.weight, a=0, mode="fan_in")

        else:
            # self.output_linear = nn.Linear(d_hidden, output_ch)
            self.lin_out = nn.Linear(d_hidden, d_out)
            # nn.init.constant_(self.lin_out.bias, 0.0)
            # nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.encoder_features = None

        if self.use_sf_nonlinear:
            self.sf_linear = nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                # nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, 6),
            )
        else:
            self.sf_linear = nn.Linear(d_hidden, 3)
            # For numerical stability
            # self.sf_linear.weight =  torch.nn.Parameter(self.sf_linear.weight * 0.025)

        self.weight_linear = nn.Linear(d_hidden, 1)

        if d_latent != 0 and self.pixelnerf_mode:
            n_lin_z = min(combine_layer, n_blocks)
            if self.feature_fuse_mode == "concat":
                self.lin_fuse_z = nn.ModuleList(
                [nn.Linear(d_hidden + d_hidden, d_hidden) for i in range(n_lin_z)]
            )
            elif self.feature_fuse_mode != "addition":
                raise NotImplementedError("Only Support addition and concat modes.")
            self.lin_z = nn.ModuleList(
                [nn.Linear(d_latent, d_hidden) for i in range(n_lin_z)]
            )
            self.lin_global_z = nn.ModuleList(
                [nn.Linear(d_global_latent+d_hidden, d_hidden) for i in range(n_global_blocks)]
            )
            # for i in range(n_lin_z):
            #     nn.init.constant_(self.lin_z[i].bias, 0.0)
            #     nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")
            if self.use_temporal_feature:
                self.temporal_propotion = temporal_propotion
                d_temporal_hidden = int(d_hidden * self.temporal_propotion)
                self.temporal_linear = nn.ModuleList(
                    [nn.Linear(d_latent, d_temporal_hidden) for i in range(n_lin_z)]
                )
                self.temporal_linear_fuse = nn.ModuleList(
                    [nn.Linear(d_hidden + d_temporal_hidden, d_hidden) for i in range(n_lin_z)]
                )

            if self.use_spade:
                self.scale_z = nn.ModuleList(
                    [nn.Linear(d_latent, d_hidden) for _ in range(n_lin_z)]
                )
                # for i in range(n_lin_z):
                #     nn.init.constant_(self.scale_z[i].bias, 0.0)
                #     nn.init.kaiming_normal_(self.scale_z[i].weight, a=0, mode="fan_in")

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        self.origin_pipeline = origin_pipeline
        self.flow_guided_feat = flow_guided_feat
        self.debug = debug
        if self.debug:
            self.clock = 0
            self.print_clock = 100

        ### ODE related
        assert flow_mode in ["ode", "discrete"]
        self.ode_flow = (flow_mode in ["ode", "discrete"])
        if self.ode_flow:
            ode_part = dict(
                lin_in=self.lin_in,
                # blocks=self.blocks,
                blocks=self.global_blocks,
                lin_global_z=self.lin_global_z,
                sf_linear=self.sf_linear,
            )
            self.vector_field = vector_field(embed_dim=self.d_in, c_dim=c_dim, z_dim=0, **ode_part)
            print(self.vector_field)
            self.ode_fc = nn.Sequential(
                nn.Linear(2048, 4096),
                nn.ReLU(),
                nn.Linear(4096, c_dim),
            )
            
            self.ode_pool = nn.AdaptiveAvgPool3d(1)

            self.p0_z = p0_z
            self.rtol = rtol
            self.atol = atol
            self.ode_solver = ode_solver

            if use_adjoint:
                self.odeint = odeint_adjoint
            else:
                self.odeint = odeint

            self.ode_options = {}
            if ode_step_size:
                self.ode_options['step_size'] = ode_step_size

    def forward(self, x, combine_inner_dims=(1,), combine_index=None, dim_size=None, sf_dict=None, latent_temporal=None):
        """
        :param zx (..., d_latent + d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """
        if self.d_latent != 0:
            x, latent, global_latent = torch.split(x, [self.input_ch + self.input_ch_views, self.d_latent, self.d_global_latent], dim=-1)

        # if self.use_viewdirsDyn:
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        x = input_pts
        if self.add_features:
            encoder_features = self.feature_embedding(latent)
            x = torch.cat([x, encoder_features.expand([input_pts.size(0),-1])], -1)
        with profiler.record_function('resnetfc_infer'):
            if self.d_latent > 0 and self.pixelnerf_mode:
                z = latent
                global_z = global_latent
                x = x
            else:
                x = x
            x = F.normalize(x, dim=-1)
            if self.d_in > 0:
                x = self.lin_in(x)
                x = F.relu(x)
            else:
                x = torch.zeros(self.d_hidden, device=x.device)

            for blkid in range(self.n_global_blocks):
                if True:
                    # tz = self.lin_global_z[blkid](global_z)
                    tz = global_z
                    # tz = torch.zeros_like(tz)
                    # print("tz absmean, std: ", tz.abs().mean().item(), tz.std().item())
                    # print("x absmean, std: ", x.abs().mean().item(), x.std().item())
                    # import ipdb; ipdb.set_trace()
                    if self.origin_pipeline:
                        tz = torch.zeros_like(tz)
                    # x = x + tz
                    x = self.lin_global_z[blkid](torch.cat([x, tz], dim=-1))
                    # x = F.normalize(x, dim=-1)
                    
                x = self.global_blocks[blkid](x)


            for blkid in range(self.n_blocks):
                if blkid == self.combine_layer:
                    # The following implements camera frustum culling, requires torch_scatter
                    #  if combine_index is not None:
                    #      combine_type = (
                    #          "mean"
                    #          if self.combine_type == "average"
                    #          else self.combine_type
                    #      )
                    #      if dim_size is not None:
                    #          assert isinstance(dim_size, int)
                    #      x = torch_scatter.scatter(
                    #          x,
                    #          combine_index,
                    #          dim=0,
                    #          dim_size=dim_size,
                    #          reduce=combine_type,
                    #      )
                    #  else:
                    x = util.combine_interleaved(
                        x, combine_inner_dims, self.combine_type
                    )

                if self.d_latent > 0 and blkid < self.combine_layer and self.pixelnerf_mode:
                    tz = self.lin_z[blkid](z)
                    if self.use_temporal_feature and blkid == 0:
                        tz_temporal = self.temporal_linear[blkid](latent_temporal)
                        tz_temporal = tz_temporal.mean(dim=-2)
                        # tz_temporal = tz_temporal.max(dim=-2).values
                        tz_fuse = torch.cat([tz, tz_temporal], dim=-1)
                        tz = self.temporal_linear_fuse[blkid](tz_fuse)

                    if self.origin_pipeline or (not self.flow_guided_feat):
                        tz = torch.zeros_like(tz)
                        # tz_temporal = torch.zeros_like(tz)
                    if self.use_spade:
                        sz = self.scale_z[blkid](z)
                        x = sz * x + tz
                    elif self.feature_fuse_mode == "addition":
                        # x = x + tz + tz_temporal
                        x = x + tz
                    else:
                        x = self.lin_fuse_z[blkid](torch.cat([x, tz], dim=-1))

                x = self.blocks[blkid](x)

            if sf_dict is None and not self.ode_flow:
                # sf = torch.tanh(self.sf_linear(F.normalize(x)))
                sf = torch.tanh(self.sf_linear(x))
            elif "sf" not in sf_dict.keys():
                t = sf_dict['t']
                xyz = sf_dict['xyz']
                t_batch = sf_dict['t_batch']
                ode_x = F.normalize(x)
                # ode_x = x
                if self.debug:
                    self.clock += 1
                    if self.clock % self.print_clock == 0:
                        from .gpu_mem_track import MemTracker
                        gpu_tracker = MemTracker()
                        gpu_tracker.track()
                forward_flow = self.transform_from_t1_to_t2(t, xyz, z=torch.empty([xyz.shape[0], 0]), c_t=ode_x, t_batch=t_batch)
                backward_flow = self.transform_from_t1_to_t2(t, xyz, z=torch.empty([xyz.shape[0], 0]), c_t=ode_x, t_batch=t_batch, invert=True)
                del sf_dict['t_batch'], sf_dict['xyz'], sf_dict['t']
                del sf_dict
                sf = torch.cat([backward_flow[:, 0, :, :3], forward_flow[:, 0, :, :3]], dim=-1)
                if self.debug and self.clock % self.print_clock == 0:
                    gpu_tracker.track()
            else:
                sf = sf_dict["sf"]

            blending = torch.sigmoid(self.weight_linear(x))
            # print("blending: ", torch.abs(blending).mean().item())
            if self.use_viewdirsDyn:
                # raise NotImplementedError
                alpha = self.alpha_linear(x)

                feature = self.feature_linear(x)
                x = torch.cat([feature, input_views], -1)

                for i, l in enumerate(self.views_linears):
                    x = self.views_linears[i](x)
                    x = F.relu(x)

                rgb = self.rgb_linear(x)
                out = torch.cat([rgb, alpha], -1)
            else:
                out = self.lin_out(x)
            return torch.cat([out, sf, blending], dim=-1)
            # return torch.cat([out, sf, blending, x], dim=-1)

    def eval_velocity_field(self, t, p, z=None, c_t=None):
        ''' Evaluates the velocity field at points p and time values t.

        Args:
            t (tensor): time values
            p (tensor): points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t
        '''
        z_dim = z.shape[-1]
        c_dim = c_t.shape[-1]

        p = self.concat_vf_input(p, c=c_t, z=z)
        t_steps_eval = torch.tensor(0).float().to(t.device)
        out = self.vector_field(t_steps_eval, p, T_batch=t).unsqueeze(0)
        p_out = self.disentangle_vf_output(
            out, c_dim=c_dim, z_dim=z_dim, return_start=True)
        p_out = p_out.squeeze(1)

        return out

    # ######################################################
    # #### Forward and Backward Flow functions #### #

    def transform_to_t_backward(self, t, p, z=None, c_t=None):
        ''' Transforms points p from time 1 (multiple) t backwards.

        For example, for t = [0.5, 1], it transforms the points from the
        coordinate system t = 1 to coordinate systems t = 0.5 and t = 0.

        Args:
            t (tensor): time values
            p (tensor): points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned code c
        '''
        device = self.device
        batch_size = p.shape[0]

        p_out, _ = self.eval_ODE(t, p, c_t=c_t, z=z,
                                 t_batch=torch.ones(batch_size).to(device),
                                 invert=True, return_start=(0 in t))

        return p_out

    def transform_to_t(self, t, p, z=None, c_t=None):
        '''  Transforms points p from time 0 to (multiple) time values t.

        Args:
            t (tensor): time values
            p (tensor); points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t

        '''
        p_out, _ = self.eval_ODE(t, p, c_t=c_t, z=z, return_start=(0 in t))

        return p_out

    def transform_from_t1_to_t2(self, t, p, z=None, c_t=None, t_batch=None, invert=False):
        '''  Transforms points p from time t1 to time value t2.

        Args:
            t (tensor): time values
            p (tensor); points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t

        '''
        p_out, t_order = self.eval_ODE(t, p, c_t=c_t, z=z, return_start=True, t_batch=t_batch, invert=invert)
        # Select respective time value for each item from batch
        batch_size = len(t_order)
        p_out = p_out[torch.arange(batch_size)[:, None], t_order]
        return p_out

    def transform_to_t0(self, t, p, z=None, c_t=None):
        ''' Transforms the points p at time t to time 0.

        Args:
            t (tensor): time values of the points
            p (tensor): points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t
        '''

        p_out, t_order = self.eval_ODE(t, p, c_t=c_t, z=z, t_batch=t,
                                       invert=True, return_start=True)

        # Select respective time value for each item from batch
        batch_size = len(t_order)
        p_out = p_out[torch.arange(batch_size), t_order]
        return p_out

    # ######################################################
    # #### ODE related functions and helper functions #### #

    def eval_ODE(self, t, p, c_t=None, z=None, t_batch=None, invert=False,
                 return_start=False):
        ''' Evaluates the ODE for points p and time values t.

        Args:
            t (tensor): time values
            p (tensor): points tensor
            c_t (tensor): latent conditioned temporal code
            z (tensor): latent code
            t_batch (tensor): helper time tensor for batch processing of points
                with different time values when going backwards
            invert (bool): whether to invert the velocity field (used for
                batch processing of points with different time values)
            return_start (bool): whether to return the start points
        '''
        c_dim = c_t.shape[-1]
        z_dim = z.shape[-1]

        t_steps_eval, t_order = self.return_time_steps(t)
        if len(t_steps_eval) == 1:
            return p.unsqueeze(1), t_order

        f_options = {'T_batch': t_batch, 'invert': invert}
        p = self.concat_vf_input(p, c=c_t, z=z)
        # p = torch.rand_like(p).cuda()
        # s = self.vector_field(torch.tensor(0).cuda(), p, **f_options)
        # s = s[None, ...].repeat(2,1,1)
        
        s = self.odeint(
            self.vector_field, p, t_steps_eval,
            method=self.ode_solver, rtol=self.rtol, atol=self.atol,
            options=self.ode_options, f_options=f_options)

        # s = torch.zeros([2, *p.shape]).cuda()

        p_out = self.disentangle_vf_output(
            s, c_dim=c_dim, z_dim=z_dim, return_start=return_start)

        return p_out, t_order

    def return_time_steps(self, t):
        ''' Returns time steps for the ODE Solver.
        The time steps are ordered, duplicates are removed, and time 0
        is added for the start.

        Args:
            t (tensor): time values
        '''
        # device = self.device
        t_steps_eval, t_order = torch.unique(
            torch.cat([torch.zeros(1).to(device)[None, :].expand(t.shape[0], 1), t], dim=1), sorted=True,
            return_inverse=True)
        return t_steps_eval, t_order[:, 1:]

    def disentangle_vf_output(self, v_out, p_dim=3, c_dim=None,
                              z_dim=None, return_start=False):
        ''' Disentangles the output of the velocity field.

        The inputs and outputs for / of the velocity network are concatenated
        to be able to use the adjoint method.

        Args:
            v_out (tensor): output of the velocity field
            p_dim (int): points dimension
            c_dim (int): dimension of conditioned code c
            z_dim (int): dimension of latent code z
            return_start (bool): whether to return start points
        '''
        n_steps, batch_size, = v_out.shape[:2]

        if z_dim is not None and z_dim != 0:
            v_out = v_out[..., :-z_dim]

        if c_dim is not None and c_dim != 0:
            v_out = v_out[..., :-c_dim]

        v_out = v_out.contiguous().view(n_steps, batch_size, -1, p_dim)

        if not return_start:
            v_out = v_out[1:]

        v_out = v_out.transpose(0, 1)

        return v_out

    def concat_vf_input(self, p, c=None, z=None):
        ''' Concatenate points p and latent code c to use it as input for ODE Solver.

        p of size (B x T x dim) and c of size (B x c_dim) and z of size
        (B x z_dim) is concatenated to obtain a tensor of size
        (B x (T*dim) + c_dim + z_dim).

        This is done to be able to use to the adjont method for obtaining
        gradients.

        Args:
            p (tensor): points tensor
            c (tensor): latent conditioned code c
            c (tensor): latent code z
        '''
        batch_size, ray_size = p.shape[:2]
        if len(c.shape) == 2:
            p_out = p.contiguous().view(batch_size, -1)
        else:
            p_out = p.contiguous().view(batch_size, ray_size, -1)
        if c is not None and c.shape[-1] != 0:
            assert(c.shape[0] == batch_size)
            # c = c.contiguous().view(batch_size, -1)
            p_out = torch.cat([p_out, c], dim=-1)

        if z is not None and z.shape[-1] != 0:
            assert(z.shape[0] == batch_size)
            # z = z.contiguous().view(batch_size, -1)
            p_out = torch.cat([p_out, z], dim=-1)

        return p_out
    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            n_blocks=conf.get_int('n_blocks', 5),
            n_global_blocks=conf.get_int('n_global_blocks', 0),
            d_hidden=conf.get_int('d_hidden', 128),
            beta=conf.get_float('beta', 0.0),
            combine_layer=conf.get_int('combine_layer', 1000),
            combine_type=conf.get_string('combine_type', 'average'),  # average | max
            use_viewdirsDyn=conf.get_bool('use_viewdirsDyn', True),
            use_spade=conf.get_bool('use_spade', False),
            pixelnerf_mode=conf.get_bool('pixelnerf_mode', False),
            feature_fuse_mode=conf.get_string('feature_fuse_mode', "addition"),  # addition | concat
            use_temporal_feature=conf.get_bool('use_temporal_feature', False),
            temporal_propotion=conf.get_float('temporal_propotion', 1.0),
            use_layer_norm=conf.get_bool('use_layer_norm', False),
            flow_mode=conf.get_string('flow_mode', 'discrete'),
            use_adjoint=conf.get_bool('use_adjoint', False),
            origin_pipeline=conf.get_bool('origin_pipeline', False),
            flow_guided_feat=conf.get_bool('flow_guided_feat', True),
            rtol=conf.get_float('rtol', 0.01),
            atol=conf.get_float('atol', 0.001),
            ode_solver=conf.get_string('ode_solver', 'dopri5'),
            ode_step_size=conf.get('ode_step_size', None),
            **kwargs
        )

if __name__ == '__main__':
    resnetfc = ResnetFC(
        d_in=84,
        n_blocks=5,
        d_hidden=512,
        beta=0.0,
        combine_layer=3,
        combine_type='average',  # average | max
        use_spade=False,
        d_latent=2048,
        d_out=4,
        )
    bs = 16
    x = torch.randn([bs, 84+2048])
    import ipdb; ipdb.set_trace()
    y = resnetfc(x)
