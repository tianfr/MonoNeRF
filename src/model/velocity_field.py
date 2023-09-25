import torch
import torch.nn as nn
import torch.nn.functional as F


# Using origin ResnetBlockFC from occupancy flow.
# Slightly different from src.model.resnetfc.ResnetBlockFC
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None, use_layer_norm=False, use_kaiming_init=False):
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
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return dx


class VelocityField(nn.Module):
    ''' Velocity Field network class.

    It maps input points and time values together with (optional) conditioned
    codes c and latent codes z to the respective motion vectors.

    Args:
        in_dim (int): input dimension of points concatenated with the time axis
        out_dim (int): output dimension of motion vectors
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): size of the hidden dimension
        leaky (bool): whether to use leaky ReLUs as activation
        n_blocks (int): number of ResNet-based blocks
    '''

    def __init__(self, in_dim=4, out_dim=3, embed_dim=84, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, n_blocks=3, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # calculate embed_dim
        # if embed_dim != 0:
        #     embed_dim = embed_dim * 2 * 4 + 4
        self.embed_dim = embed_dim
        # Submodules
        self.embed_fn_d = None
        # if embed_dim == 0:
        #     self.fc_p = nn.Linear(in_dim, hidden_size)
        # else:
        #     self.fc_p = nn.Linear(embed_dim, hidden_size)
        self.fc_p = kwargs['lin_in']

        # self.blocks = nn.ModuleList([
        #     ResnetBlockFC(hidden_size) for i in range(n_blocks)
        # ])
        self.blocks = kwargs['blocks']
        self.n_blocks = len(self.blocks)
        self.fc_c = kwargs['lin_global_z']


        # if z_dim != 0:
        #     self.fc_z = nn.ModuleList([
        #         nn.Linear(z_dim, hidden_size) for i in range(self.n_blocks)])

        # if c_dim != 0:
        #     self.fc_c = nn.ModuleList([
        #         nn.Linear(c_dim, hidden_size) for i in range(self.n_blocks)])

        # self.blocks = nn.ModuleList([
        #     ResnetBlockFC(hidden_size) for i in range(n_blocks)
        # ])


        self.fc_out = kwargs['sf_linear']

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.independent_feature = False

    def disentangle_inputs(self, inputs):
        ''' Disentangles the inputs and returns the points and latent code
        tensors separately.

        The input consists of the latent code z, the latent conditioned code c,
        and the points. They are concatenated before using as input for the
        velocity field to be able to use the adjoint method to obtain
        gradients. Here, the full input tensor is disentangled again into its
        components.

        Args:
            inputs (tensor): velocity field inputs
        '''
        c_dim = self.c_dim
        z_dim = self.z_dim
        batch_size, device = inputs.shape[0], inputs.device

        if z_dim is not None and z_dim != 0:
            z = inputs[..., -z_dim:]
            p = inputs[..., :-z_dim]
        else:
            z = torch.empty(batch_size, 0).to(device)
            p = inputs

        if c_dim is not None and c_dim != 0:
            c = p[..., -c_dim:]
            p = p[..., :-c_dim]
        else:
            c = torch.empty(batch_size, 0).to(device)

        p = p.view(batch_size, -1, self.out_dim)

        return p, c, z

    def concat_time_axis(self, t, p, t_batch=False, invert=False):
        ''' Concatenates the time axis to the points tenor.

        Args:
            t (tensor); time values
            p (tensor): points
            t_batch (tensor): time help tensor for batch processing
            invert (bool): whether to go backwards
        '''
        batch_size, n_points, _ = p.shape
        t = t * 12.

        t = t.repeat(batch_size)
        if t_batch is not None:
            assert(len(t_batch) == batch_size)
            if invert:
                t = t_batch - t[:, None]
            else:
                t = t_batch + t[:, None]
        t = t / 12. * 2. -1
        # Add Temporal Axis
        t = t.view(batch_size, n_points, 1).expand(batch_size, n_points, 1)
        p_out = torch.cat([p, t], dim=-1)
        assert(p_out.shape[-1] == self.in_dim)

        return p_out

    def concat_output(self, out):
        ''' Returns the output of the velocity network.

        The points, the conditioned codes c, and the latent codes z are
        concatenated to produce a single output of similar size as the input
        tensor. Zeros are concatenated with respect to the dimensions of the
        hidden vectors c and z. (This ensures that the "motion vectors" for
        these "fake points" are 0, and thus are not used by the adjoint method
        to calculate the step size.)

        Args:
            out (tensor): output points
        '''
        batch_size, ray_size = out.shape[:2]
        device = out.device
        c_dim = self.c_dim
        z_dim = self.z_dim

        if self.independent_feature:
            batch_size = [batch_size, ray_size]
        else:
            batch_size = [batch_size]

        out = out.contiguous().view(*batch_size, -1)
        if c_dim != 0:
            c_out = torch.zeros(*batch_size, c_dim).to(device)
            out = torch.cat([out, c_out], dim=-1)
        if z_dim != 0:
            z_out = torch.zeros(*batch_size, z_dim).to(device)
            out = torch.cat([out, z_out], dim=-1)

        return out

    def forward(self, t, p, T_batch=None, invert=False, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            t (tensor): time values
            p (tensor): points
            T_batch (tensor): time helper tensor to perform batch processing
                when going backwards in time
            invert (bool): whether to go backwards
        '''
        p, c, z = self.disentangle_inputs(p)
        p = self.concat_time_axis(t, p, T_batch, invert)

        if self.embed_dim != 0:
            p = self.embed_fn_d(p)
        p = F.normalize(p, dim=-1)
        if len(c.shape) == 3:
            self.independent_feature = True

        net = self.fc_p(p)
        net = self.actvn(net)

        # Layer loop
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                # net_c = self.fc_c[i](c)
                net_c = c
                if self.independent_feature:
                    # net = net + net_c
                    net = self.fc_c[i](torch.cat([net, net_c], dim=-1))
                else:
                    # net = net + net_c[:, None, :]
                    net = self.fc_c[i](torch.cat([net, net_c[:, None, :].expand(*net.shape)], dim=-1))


            # if self.z_dim != 0:
            #     net_z = self.fc_z[i](z)
            #     if self.independent_feature:
            #         net = net + net_c
            #     else:
            #         net = net + net_z[:, None, :]
            net = self.blocks[i](net)
        # import ipdb; ipdb.set_trace()
        motion_vectors = torch.tanh(self.fc_out(F.normalize(net)))

        # when going backwards in time, return -v
        sign = -1 if invert else 1
        motion_vectors = sign * motion_vectors
        out = self.concat_output(motion_vectors)

        return out
