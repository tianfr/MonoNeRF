from .encoder import (ImageEncoder, Resnet18, SpatialTemporalEncoder)
from .resnetfc import ResnetFC
from .resnetfc_static import ResnetFC_static


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)

def make_mlp(conf, d_in, d_latent=0, allow_empty=False, **kwargs):
    mlp_type = conf.get_string('type', 'mlp')  # mlp | resnet
    if mlp_type == 'resnet':
        net = ResnetFC.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
    elif mlp_type == 'resnet_static':
        net = ResnetFC_static.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
    elif mlp_type == 'empty' and allow_empty:
        net = None
    else:
        raise NotImplementedError('Unsupported MLP type')
    return net


def make_encoder(conf, **kwargs):
    enc_type = conf.get_string('type', 'slowonly')  # spatial | global
    print(f'Use {enc_type} for backbone.')
    if enc_type == 'spatial':
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == 'global':
        net = ImageEncoder.from_conf(conf, **kwargs)
    elif enc_type == 'slowonly':
        net = SpatialTemporalEncoder(
            num_layers=conf['num_layers'],
            config_path=conf['config_path'],
            pretrained=conf['pretrained'],
            pretrained_path=conf['pretrained_path'],

        )
    elif enc_type == 'resnet18':
        net = Resnet18()

    else:
        raise NotImplementedError('Unsupported encoder type')
    return net
