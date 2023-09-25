import torch
#  import torch_scatter
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from torch import nn

from src import util


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0):
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

        # Init
        # nn.init.constant_(self.fc_0.bias, 0.0)
        # nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        # nn.init.constant_(self.fc_1.bias, 0.0)
        # nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            # nn.init.constant_(self.shortcut.bias, 0.0)
            # nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        with profiler.record_function('resblock'):
            net = self.fc_0(self.activation(x))
            dx = self.fc_1(self.activation(net))

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx
            # return dx


class ResnetFC_static(nn.Module):
    def __init__(
        self,
        d_in,
        d_out=4,
        n_blocks=5,
        d_latent=2048,
        d_hidden=128,
        beta=0.0,
        combine_layer=1000,
        combine_type='average',
        use_spade=False,
        # D=8,
        # W=256,
        input_ch=63,
        input_ch_views=27,
        output_ch=4,
        # skips=[4],
        use_viewdirs=True,
        add_features=False,
        pixelnerf_mode=False,
        origin_pipeline=False,
        debug=False,
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
        # if use_viewdirs:
        #     d_in = input_ch
        # else:
        #     d_in = input_ch + input_ch_views
        d_in = input_ch
        d_out = output_ch
        if (not add_features) and (not pixelnerf_mode):
            d_latent = 0


        self.n_blocks = n_blocks
        self.d_latent = d_latent
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden

        self.combine_layer = combine_layer
        self.combine_type = combine_type
        self.use_spade = use_spade

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        # self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.add_features = add_features
        self.pixelnerf_mode = pixelnerf_mode
        if self.pixelnerf_mode:
            self.add_features = False
            print('PixelNeRF mode in Static NeRF.')
        if self.add_features:
            if d_latent == 0:
                raise ValueError('d_latent is ZERO!')
            self.feature_embedding = nn.Linear(d_latent, 128)
            self.d_in += 128
        else:
            self.feature_embedding = None

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(d_hidden, beta=beta) for i in range(n_blocks)]
        )



        if d_in > 0:
            self.lin_in = nn.Linear(self.d_in, d_hidden)
            # nn.init.constant_(self.lin_in.bias, 0.0)
            # nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")


        if self.use_viewdirs:
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

        self.weight_linear = nn.Linear(d_hidden, 1)

        if d_latent != 0 and self.pixelnerf_mode:
            n_lin_z = min(combine_layer, n_blocks)
            self.lin_z = nn.ModuleList(
                [nn.Linear(d_latent, d_hidden) for i in range(n_lin_z)]
            )
            # for i in range(n_lin_z):
            #     nn.init.constant_(self.lin_z[i].bias, 0.0)
            #     nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

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
        self.debug = debug
        if self.debug:
            self.clock = 0
            self.print_clock = 100

    def forward(self, x, combine_inner_dims=(1,), combine_index=None, dim_size=None):
        """
        :param zx (..., d_latent + d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """
        # import ipdb; ipdb.set_trace()
        if self.d_latent != 0:
            x, latent = torch.split(x, [self.input_ch + self.input_ch_views, self.d_latent], dim=-1)
        # if self.use_viewdirs:
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        x = input_pts
        if self.add_features:
            encoder_features = self.feature_embedding(latent)
            x = torch.cat([x, encoder_features.expand([input_pts.size(0),-1])], -1)
        with profiler.record_function('resnetfc_infer'):
            if self.d_latent > 0 and self.pixelnerf_mode:
                z = latent
                x = x
            else:
                x = x
            # x = F.normalize(x, dim=-1)
            if self.d_in > 0:
                x = self.lin_in(x)
                x = F.relu(x)
            else:
                x = torch.zeros(self.d_hidden, device=x.device)

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
                    if self.origin_pipeline:
                        tz = torch.zeros_like(tz)
                    if self.use_spade:
                        sz = self.scale_z[blkid](z)
                        x = sz * x + tz
                    else:
                        x = x + tz

                x = self.blocks[blkid](x)

            blending = torch.sigmoid(self.weight_linear(x))
            # print("blending: ", torch.abs(blending).mean().item())
            if self.use_viewdirs:
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
            return torch.cat([out, blending], dim=-1)

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            n_blocks=conf.get_int('n_blocks', 5),
            d_hidden=conf.get_int('d_hidden', 128),
            beta=conf.get_float('beta', 0.0),
            combine_layer=conf.get_int('combine_layer', 1000),
            combine_type=conf.get_string('combine_type', 'average'),  # average | max
            use_viewdirs=conf.get_bool('use_viewdirs', True),
            use_spade=conf.get_bool('use_spade', False),
            pixelnerf_mode=conf.get_bool('pixelnerf_mode', False),
            origin_pipeline=conf.get_bool('origin_pipeline', False),
            **kwargs
        )

if __name__ == '__main__':
    resnetfc = ResnetFC_static(
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
