"""
Implements image encoders
"""
import torch
import torch.autograd.profiler as profiler
import torch.nn.functional as F
import torchvision
from mmaction import __version__
from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from torch import nn
from torchvision import models



def normalize_imagenet(x):
    ''' Normalize input images according to ImageNet standards.

    Args:
        x (tensor): input images
    '''
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x

def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)

@torch.no_grad()
def cal_feature_dis(features):
    import ipdb; ipdb.set_trace()
    import tqdm
    N, C, T, H, W = features.shape
    features = features.permute(0, 2, 3, 4, 1).reshape(N, T*H*W, C)
    f0 = features[0]
    f1 = features[1]
    count = 0
    minn, maxn = 1e9, 0
    for i in tqdm.tqdm(range(0, T*H*W, 5)):
        currf0 = f0[i: i+5]
        currdis = torch.abs(f1[:, None, :] - currf0[None, :, :])
        currdis = torch.sum(currdis, dim=-1)
        currdis = torch.min(currdis)
        if currdis < 20:
            count += 1
            print(currdis)
        minn = min(currdis, minn)
        maxn = max(currdis, maxn)
    import ipdb; ipdb.set_trace()
    print(count, maxn, minn)

    return dis01

class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(
        self,
        num_layers=4,
        index_interp='bilinear',
        index_padding='border',
        upsample_interp='bilinear',
        normalize=True,
    ):
        super().__init__()
        self.normalize = normalize
        # self.use_linear = use_linear
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        # if use_linear:
        #     self.fc = nn.Linear(512, c_dim)
        # elif c_dim == 512:
        #     self.fc = nn.Sequential()
        # else:
        #     raise ValueError('c_dim must be 512 if use_linear is False')
        self.latent_size = sum([64, 64, 128, 256, 512][:num_layers])

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp

        # self.register_buffer('latent', torch.empty(1, 1, 1, 1), persistent=False)
        # self.register_buffer(
        #     'latent_scaling', torch.empty(2, dtype=torch.float32), persistent=False
        # )
    def forward(self, x):
        N, T, H, W, C = x.shape
        x = x.reshape(N*T, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.normalize:
            x = normalize_imagenet(x)

        encoder = self.features
        x = encoder.conv1(x)
        x = encoder.bn1(x)
        x = encoder.relu(x)
        x = encoder.maxpool(x)

        latents = [x]
        for i in range(self.num_layers):
            if self.num_layers <= (i+1):
                break
            layer_name = 'layer' + str(i+1)
            res_layer = getattr(encoder, layer_name)
            x = res_layer(x)
            latents.append(x)

        # self.latents = latents # 64 + 256 + 512 + 1024 = 1856
        align_corners = None if self.index_interp == 'nearest' else True
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode=self.upsample_interp,
                align_corners=align_corners,
            )
        latent = torch.cat(latents, dim=1)

        latent_scaling = torch.empty(2, dtype=torch.float32)
        latent_scaling[0] = latent.shape[-1]
        latent_scaling[1] = latent.shape[-2]
        latent_scaling = latent_scaling / (latent_scaling - 1) * 2.0
        # import ipdb; ipdb.set_trace()
        # cal_feature_dis(self.latent)
        latent = latent.reshape(N, T, *latent.shape[1:])
        latent = latent.permute(0, 2, 1, 3, 4).contiguous()

        latent_dict = dict(
            latent = latent,
            latent_scaling = latent_scaling
        )
        return latent_dict

    def index(self, uv, t, cam_z, image_size, z_bounds=None, batch=None, mask_latent=None, latent_dict=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """
        # import ipdb; ipdb.set_trace()
        # import copy
        # ori_uv = copy.deepcopy(uv)
        # tmp = torch.arange(0, 240*135).reshape([1,1,240, 135]).float().cuda()
        with profiler.record_function('encoder_index'):
            # if uv.shape[0] == 1 and self.latent.shape[0] > 1:
            #     uv = uv.expand(self.latent.shape[0], -1, -1)
            latent_scaling = latent_dict["latent_scaling"]
            latent = latent_dict["latent"]

            with profiler.record_function('encoder_index_pre'):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            if batch:
                idx = batch
            else:
                idx = list(range(t.shape[0]))
            # uv = uv[None, :, None, :]  # (1, N, 1, 2)
            if mask_latent is None:
                mask_latent = latent[idx, :, t, ...]
            samples = F.grid_sample(
                mask_latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )
            return samples[:, :, :, 0]  # (B, C, N)


class SpatialTemporalEncoder(nn.Module):
    """
    3D (Spatial/Pixel-aligned/local) video encoder
    """

    def __init__(
        self,
        backbone='slowonly',
        pretrained=True,
        num_layers=4,
        index_interp='bilinear',
        index_padding='border',
        upsample_interp='trilinear',
        feature_scale=1.0,
        use_first_pool=True,
        norm_type='batch',
        config_path = 'mmaction_configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb.py',
        pretrained_path = 'checkpoints/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912-3f9ce182.pth',
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        config_options = {}
        cfg = Config.fromfile(config_path)
        cfg.merge_from_dict(config_options)
        turn_off_pretrained(cfg.model)
        self.model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        self.pretrained = pretrained
        if self.pretrained:
            self.pretrained_path = pretrained_path
            load_checkpoint(self.model, self.pretrained_path, map_location='cpu',)

        # if norm_type != "batch":
        #     assert not pretrained

        # self.use_custom_resnet = backbone == "custom"
        # self.feature_scale = feature_scale
        # self.use_first_pool = use_first_pool
        # norm_layer = util.get_norm_layer(norm_type)

        
        # self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]
        self.layer_latent_size = [64, 256, 512, 1024, 2048]

        if type(num_layers) == int:
            self.used_latent_layers = list(range(num_layers))
            self.latent_size = sum(self.layer_latent_size[:num_layers])

        elif type(num_layers) == list:
            self.used_latent_layers = num_layers
            self.latent_size = 0
            for i in range(len(self.layer_latent_size)):
                if i in num_layers:
                    self.latent_size += self.layer_latent_size[i]

        self.max_latent_layers = max(self.used_latent_layers)
        self.max_latent_layers = 5
                    

        # self.latent_size = [64, 256, 512, 1024, 2048][num_layers-1]

        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        # self.register_buffer('latent', torch.empty(1, 1, 1, 1), persistent=False)
        # self.register_buffer(
        #     'latent_scaling', torch.empty(2, dtype=torch.float32), persistent=False
        # )
        # self.latent (B, L, H, W)

    def index(self, uv, t, cam_z, image_size, z_bounds=None, batch=None, mask_latent=None, latent_dict=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """
        # import ipdb; ipdb.set_trace()
        # import copy
        # ori_uv = copy.deepcopy(uv)
        # tmp = torch.arange(0, 240*135).reshape([1,1,240, 135]).float().cuda()
        with profiler.record_function('encoder_index'):
            # if uv.shape[0] == 1 and self.latent.shape[0] > 1:
            #     uv = uv.expand(self.latent.shape[0], -1, -1)
            latent_scaling = latent_dict["latent_scaling"]
            latent = latent_dict["latent"]

            with profiler.record_function('encoder_index_pre'):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            if batch:
                idx = batch
            else:
                idx = list(range(t.shape[0]))
            # uv = uv[None, :, None, :]  # (1, N, 1, 2)
            if mask_latent is None:
                mask_latent = latent[idx, :, t, ...]
            samples = F.grid_sample(
                mask_latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )
            return samples[:, :, :, 0]  # (B, C, N)

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        encoder = self.model.backbone
        x = encoder.conv1(x.cuda())
        if encoder.with_pool1: # false
            x = encoder.maxpool(x)
        latents = []
        if 0 in self.used_latent_layers:
            latents.append(x)
        for i, layer_name in enumerate(encoder.res_layers):
            if self.max_latent_layers <= (i+1):
                break

            res_layer = getattr(encoder, layer_name)
            x = res_layer(x)
            if i == 0 and encoder.with_pool2:
                x = encoder.pool2(x)
            if (i + 1) in self.used_latent_layers:
                latents.append(x)
        flow_latent = x
        # self.latents = latents # 64 + 256 + 512 + 1024 = 1856
        align_corners = None if self.index_interp == 'nearest' else True
        latent_sz = latents[0].shape[-3:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode=self.upsample_interp,
                align_corners=align_corners,
            )
        latent = torch.cat(latents, dim=1)
        latent_scaling = torch.empty(2, dtype=torch.float32)
        latent_scaling[0] = latent.shape[-1]
        latent_scaling[1] = latent.shape[-2]
        latent_scaling = latent_scaling / (latent_scaling - 1) * 2.0
        # import ipdb; ipdb.set_trace()
        # cal_feature_dis(self.latent)
        latent_dict = dict(
            latent = latent,
            latent_scaling = latent_scaling,
            flow_latent = flow_latent,
        )
        return latent_dict


    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string('backbone'),
            pretrained=conf.get_bool('pretrained', True),
            num_layers=conf.get_list('num_layers', [0, 1, 2, 3]),
            index_interp=conf.get_string('index_interp', 'bilinear'),
            index_padding=conf.get_string('index_padding', 'border'),
            upsample_interp=conf.get_string('upsample_interp', 'trilinear'),
            feature_scale=conf.get_float('feature_scale', 1.0),
            use_first_pool=conf.get_bool('use_first_pool', True),
        )



class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone='resnet34', pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer('latent', torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string('backbone'),
            pretrained=conf.get_bool('pretrained', True),
            latent_size=conf.get_int('latent_size', 128),
        )
