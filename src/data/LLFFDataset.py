import glob
import os

import cv2
import imageio
import ipdb
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.pyplot import grid
from mmaction.datasets.pipelines import Compose
from mmcv import Config

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
DEFAULT_PIPELINE = [
    # dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    # dict(type='RawFrameDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='RandomResizedCrop'),
    #! Here is a bug where the scale should be (480, 270) but it remains for reimplementation.
    dict(type='Resize', scale=(270, 480), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='FormatShape', input_format='NTHWC'),
    # dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    # dict(type='ToTensor', keys=['imgs'])
]



def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def normalize(x):
    return x / np.linalg.norm(x)

def resize_flow(flow, H_new, W_new):
    H_old, W_old = flow.shape[0:2]
    flow_resized = cv2.resize(flow, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
    flow_resized[:, :, 0] *= H_new / H_old
    flow_resized[:, :, 1] *= W_new / W_old
    return flow_resized

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    # print('factor ', factor)
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        if width % 2 == 1:
            width -= 1
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        if height % 2 == 1:
            height -= 1
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return

    sh = imageio.imread(imgfiles[0]).shape #(270, 480, 3)
    num_img = len(imgfiles)
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    encoder_imgs = [imread(f)[..., :3] for f in imgfiles]
    imgs = np.stack(imgs, -1) # (270, 480, 3, 12)
    encoder_imgs = np.stack(encoder_imgs, -1) # (270, 480, 3, 12)

    assert imgs.shape[0] == sh[0]
    assert imgs.shape[1] == sh[1]

    disp_dir = os.path.join(basedir, 'disp')

    dispfiles = [os.path.join(disp_dir, f) \
                for f in sorted(os.listdir(disp_dir)) if f.endswith('npy')]

    disp = [cv2.resize(np.load(f),
                    (sh[1], sh[0]),
                    interpolation=cv2.INTER_NEAREST) for f in dispfiles]
    disp = np.stack(disp, -1) # (270, 480, 12)

    mask_dir = os.path.join(basedir, 'motion_masks')
    maskfiles = [os.path.join(mask_dir, f) \
                for f in sorted(os.listdir(mask_dir)) if f.endswith('png')]

    masks = [cv2.resize(imread(f)/255., (sh[1], sh[0]),
                        interpolation=cv2.INTER_NEAREST) for f in maskfiles]
    masks = np.stack(masks, -1)
    masks = np.float32(masks > 1e-3) # (270, 380, 12)

    flow_dir = os.path.join(basedir, 'flow')
    flows_f = []
    flow_masks_f = []
    flows_b = []
    flow_masks_b = []
    for i in range(num_img):
        if i == num_img - 1:
            fwd_flow, fwd_mask = np.zeros((sh[0], sh[1], 2)), np.zeros((sh[0], sh[1]))
        else:
            fwd_flow_path = os.path.join(flow_dir, '%03d_fwd.npz'%i)
            fwd_data = np.load(fwd_flow_path)
            fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
            fwd_flow = resize_flow(fwd_flow, sh[0], sh[1])
            fwd_mask = np.float32(fwd_mask)
            fwd_mask = cv2.resize(fwd_mask, (sh[1], sh[0]),
                                interpolation=cv2.INTER_NEAREST)
        flows_f.append(fwd_flow)
        flow_masks_f.append(fwd_mask)

        if i == 0:
            bwd_flow, bwd_mask = np.zeros((sh[0], sh[1], 2)), np.zeros((sh[0], sh[1]))
        else:
            bwd_flow_path = os.path.join(flow_dir, '%03d_bwd.npz'%i)
            bwd_data = np.load(bwd_flow_path)
            bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
            bwd_flow = resize_flow(bwd_flow, sh[0], sh[1])
            bwd_mask = np.float32(bwd_mask)
            bwd_mask = cv2.resize(bwd_mask, (sh[1], sh[0]),
                                interpolation=cv2.INTER_NEAREST)
        flows_b.append(bwd_flow)
        flow_masks_b.append(bwd_mask)

    flows_f = np.stack(flows_f, -1)
    flow_masks_f = np.stack(flow_masks_f, -1)
    flows_b = np.stack(flows_b, -1)
    flow_masks_b = np.stack(flow_masks_b, -1)

    # print(imgs.shape)
    # print(disp.shape)
    # print(masks.shape)
    # print(flows_f.shape)
    # print(flow_masks_f.shape)
    # (270, 480, 3, 12)
    # (270, 480, 12)
    # (270, 480, 12)
    # (270, 480, 2, 12)
    # (270, 480, 12)

    assert(imgs.shape[0] == disp.shape[0])
    assert(imgs.shape[0] == masks.shape[0])
    assert(imgs.shape[0] == flows_f.shape[0])
    assert(imgs.shape[0] == flow_masks_f.shape[0])

    assert(imgs.shape[1] == disp.shape[1])
    assert(imgs.shape[1] == masks.shape[1])

    return poses, bds, imgs, encoder_imgs, disp, masks, flows_f, flow_masks_f, flows_b, flow_masks_b

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def get_grid(H, W, num_img, flows_f, flow_masks_f, flows_b, flow_masks_b):

    # |--------------------|  |--------------------|
    # |       j            |  |       v            |
    # |   i   *            |  |   u   *            |
    # |                    |  |                    |
    # |--------------------|  |--------------------|

    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')

    grid = np.empty((0, H, W, 8), np.float32)
    for idx in range(num_img):
        grid = np.concatenate((grid, np.stack([i,
                                               j,
                                               flows_f[idx, :, :, 0],
                                               flows_f[idx, :, :, 1],
                                               flow_masks_f[idx, :, :],
                                               flows_b[idx, :, :, 0],
                                               flows_b[idx, :, :, 1],
                                               flow_masks_b[idx, :, :]], -1)[None, ...]))
    return grid


class LLFFDataset(torch.utils.data.Dataset):
    """
    Dataset from LLFF.
    """

    def __init__(
        self,
        path,
        pipeline=DEFAULT_PIPELINE,
        frame2dolly=-1,
        factor=8,
        spherify=False,
        num_novelviews=60,
        focal_decrease=200,
        z_trans_multiplier=5.,
        x_trans_multiplier=1.,
        y_trans_multiplier=0.33,
        no_ndc=False,
        stage='train',
        list_prefix='softras_',
        image_size=None,
        sub_format='shapenet',
        scale_focal=True,
        max_imgs=100000,
        z_near=0.0,
        z_far=1.0,
        skip_step=None,
        file_lists=None,
    ):
        """
        :param path dataset root path, contains metadata.yml
        :param stage train | val | test
        :param list_prefix prefix for split lists: <list_prefix>[train, val, test].lst
        :param image_size result image size (resizes if different); None to keep original size
        :param sub_format shapenet | dtu dataset sub-type.
        :param scale_focal if true, assume focal length is specified for
        image of side length 2 instead of actual image size. This is used
        where image coordinates are placed in [-1, 1].
        """
        super().__init__()

        self.base_path = path
        # assert os.path.exists(self.base_path)

        if file_lists == None:
            self.file_lists = [x for x in glob.glob(os.path.join(path, '*')) if os.path.isdir(x)]
        elif len(file_lists) == 1 and file_lists[0] == "ALL":
            path = "./data"
            self.file_lists = [x for x in glob.glob(os.path.join(path, '*')) if os.path.isdir(x)]
        else:
            self.file_lists = file_lists

        self.frame2dolly = frame2dolly
        self.factor = factor
        self.spherify = spherify
        self.no_ndc = no_ndc
        self.num_novelviews =  num_novelviews
        self.focal_decrease = focal_decrease
        self.x_trans_multiplier = x_trans_multiplier
        self.y_trans_multiplier = y_trans_multiplier
        self.z_trans_multiplier =  z_trans_multiplier


        self.image_size = image_size
        if sub_format == 'dtu':
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        else:
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        self.sub_format = sub_format
        self.scale_focal = scale_focal
        self.max_imgs = max_imgs

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False
        self.encoder_pipeline = Compose(pipeline)


    def __len__(self):
        return len(self.file_lists)

    def __getitem__(self, index):
        datadir = self.file_lists[index]
        images, encoder_imgs, invdepths, masks, poses, bds, \
        render_poses, render_focals, grids = self.load_llff_data(datadir,
                                                            self.factor,
                                                            frame2dolly=self.frame2dolly,
                                                            recenter=True, bd_factor=.9,
                                                            spherify=self.spherify)

        encoder_data = dict(
            imgs = encoder_imgs,
            modality = 'RGB'
        )
        encoder_data = self.encoder_pipeline(encoder_data)
        result = dict(
            dataname = os.path.basename(datadir),
            images = images,
            encoder_imgs = encoder_data['imgs'],
            invdepths = invdepths,
            masks = masks,
            poses = poses,
            bds = bds,
            render_poses = render_poses,
            render_focals = render_focals,
            grids = grids,
        )

        return result
    def generate_path(self, c2w):
        hwf = c2w[:, 4:5]
        num_novelviews = self.num_novelviews
        max_disp = 48.0
        H, W, focal = hwf[:, 0]

        max_trans = max_disp / focal
        output_poses = []
        output_focals = []

        # Rendering teaser. Add translation.
        for i in range(num_novelviews):
            x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_novelviews)) * self.x_trans_multiplier
            y_trans = max_trans * (np.cos(2.0 * np.pi * float(i) / float(num_novelviews)) - 1.) * self.y_trans_multiplier
            z_trans = 0.

            i_pose = np.concatenate([
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
            ],axis=0)

            i_pose = np.linalg.inv(i_pose)

            ref_pose = np.concatenate([c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

            render_pose = np.dot(ref_pose, i_pose)
            output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
            output_focals.append(focal)

        # Rendering teaser. Add zooming.
        if self.frame2dolly != -1:
            for i in range(num_novelviews // 2 + 1):
                x_trans = 0.
                y_trans = 0.
                # z_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_novelviews)) * self.z_trans_multiplier
                z_trans = max_trans * self.z_trans_multiplier * i / float(num_novelviews // 2)
                i_pose = np.concatenate([
                    np.concatenate(
                        [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
                    np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
                ],axis=0)

                i_pose = np.linalg.inv(i_pose) #torch.tensor(np.linalg.inv(i_pose)).float()

                ref_pose = np.concatenate([c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

                render_pose = np.dot(ref_pose, i_pose)
                output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
                output_focals.append(focal)
                print(z_trans / max_trans / self.z_trans_multiplier)

        # Rendering teaser. Add dolly zoom.
        if self.frame2dolly != -1:
            for i in range(num_novelviews // 2 + 1):
                x_trans = 0.
                y_trans = 0.
                z_trans = max_trans * self.z_trans_multiplier * i / float(num_novelviews // 2)
                i_pose = np.concatenate([
                    np.concatenate(
                        [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
                    np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
                ],axis=0)

                i_pose = np.linalg.inv(i_pose)

                ref_pose = np.concatenate([c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

                render_pose = np.dot(ref_pose, i_pose)
                output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
                new_focal = focal - self.focal_decrease * z_trans / max_trans / self.z_trans_multiplier
                output_focals.append(new_focal)
                print(z_trans / max_trans / self.z_trans_multiplier, new_focal)
        return output_poses, output_focals

    def load_llff_data(self, basedir,
                    factor=2,
                    recenter=True, bd_factor=.75,
                    spherify=False, path_zflat=False,
                    frame2dolly=10):

        poses, bds, imgs, encoder_imgs, disp, masks, flows_f, flow_masks_f, flows_b, flow_masks_b = \
            _load_data(basedir, factor=factor) # factor=2 downsamples original imgs by 2x

        # print('Loaded', basedir, bds.min(), bds.max())

        # Correct rotation matrix ordering and move variable dim to axis 0
        poses = np.concatenate([poses[:, 1:2, :],
                            -poses[:, 0:1, :],
                                poses[:, 2:, :]], 1)
        # move the frame (last) axis to the first axis
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = np.moveaxis(imgs, -1, 0).astype(np.float32)
        encoder_imgs = np.moveaxis(encoder_imgs, -1, 0).astype(np.float32)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32) # 12, 2
        disp = np.moveaxis(disp, -1, 0).astype(np.float32)
        masks = np.moveaxis(masks, -1, 0).astype(np.float32)
        flows_f = np.moveaxis(flows_f, -1, 0).astype(np.float32)
        flow_masks_f = np.moveaxis(flow_masks_f, -1, 0).astype(np.float32)
        flows_b = np.moveaxis(flows_b, -1, 0).astype(np.float32)
        flow_masks_b = np.moveaxis(flow_masks_b, -1, 0).astype(np.float32)

        # Rescale if bd_factor is provided
        sc = 1. if bd_factor is None else 1./(np.percentile(bds[:, 0], 5) * bd_factor)

        poses[:, :3, 3] *= sc
        bds *= sc

        if recenter:
            poses = recenter_poses(poses)

        # Only for rendering
        if frame2dolly == -1:
            c2w = poses_avg(poses)
        else:
            c2w = poses[frame2dolly, :, :]

        H, W, _ = c2w[:, -1]

        # Generate poses for novel views
        render_poses, render_focals = self.generate_path(c2w)
        render_poses = np.array(render_poses).astype(np.float32)

        grids = get_grid(int(H), int(W), len(poses), flows_f, flow_masks_f, flows_b, flow_masks_b) # [N, H, W, 8]

        return images, encoder_imgs, disp, masks, poses, bds,\
            render_poses, render_focals, grids

if __name__ == '__main__':
    '''
    Start debugging from the root path!
    '''
    config_path = 'mmaction_configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb.py'
    config_options = {}
    cfg = Config.fromfile(config_path)
    cfg.merge_from_dict(config_options)
    llff_dataset = LLFFDataset(
        path='data',
        # pipeline=cfg.data.train.pipeline,
        stage='train',
        factor=2,
    )
    train_data = torch.utils.data.DataLoader(
        llff_dataset, batch_size=2, shuffle=False, num_workers=8, pin_memory=False
    )
    for (idx, data) in enumerate(train_data):
        ipdb.set_trace()
