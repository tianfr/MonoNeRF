import configargparse
import os
import sys

from pyhocon import ConfigFactory


def parse_args(
    callback=None,
    default_conf='conf/default_mv.conf',
    default_expname='dynamic',
    default_data_format='dvr',
    default_num_epochs=10000000,
    default_gamma=1.00,
):

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='configs/config_Balloon2.txt',
                        help='config file path')
    parser.add_argument('--expname', type=str, default="Balloon1_H270_DyNeRF_pretrain",
                        help='experiment name')
    parser.add_argument('--basedir', type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument('--datadir', type=str, default='./data/Balloon1/',
                        help='input data directory')

    # adding semantic feature
    parser.add_argument('--add_features', action='store_true', default=False,
                        help='add the high level feature from SlowOnly network')

    # training options
    parser.add_argument('--netdepth', type=int, default=8,
                        help='layers in network')
    parser.add_argument('--netwidth', type=int, default=256,
                        help='channels per layer')
    parser.add_argument('--netdepth_fine', type=int, default=8,
                        help='layers in fine network')
    parser.add_argument('--netwidth_fine', type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument('--N_rand', type=int, default=1024,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument('--lrate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--lrate_decay', type=int, default=300000,
                        help='exponential learning rate decay')
    parser.add_argument('--chunk', type=int, default=1024*256,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument('--netchunk', type=int, default=1024*128,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument('--no_reload', action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument('--ft_path', type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='fix random seed for repeatability')

    # rendering options
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument('--N_importance', type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument('--perturb', type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument('--use_viewdirs', action='store_true', default=True,
                        help='use full 5D input instead of 3D')
    parser.add_argument('--use_viewdirsDyn', action='store_true',
                        help='use full 5D input instead of 3D for D-NeRF')
    parser.add_argument('--i_embed', type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument('--multires', type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument('--multires_views', type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument('--raw_noise_std', type=float, default=1e0,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument('--render_only', action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')

    # dataset options
    parser.add_argument('--dataset_type', type=str, default='llff',
                        help='options: llff')

    # llff flags
    parser.add_argument('--factor', type=int, default=2,
                        help='downsample factor for LLFF images')
    parser.add_argument('--no_ndc', action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument('--lindisp', action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument('--spherify', action='store_true',
                        help='set for spherical 360 scenes')

    # logging/saving options
    parser.add_argument('--i_print',   type=int, default=1000,
                        help='frequency of console printout and metric logging')
    parser.add_argument('--i_log',   type=int, default=100,
                        help='frequency of console printout and metric logging')
    parser.add_argument('--i_img',     type=int, default=1000,
                        help='frequency of tensorboard image logging')
    parser.add_argument('--i_weights', type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument('--i_testset', type=int, default=20000,
                        help='frequency of testset saving')
    parser.add_argument('--i_video',   type=int, default=20000,
                        help='frequency of render_poses video saving')
    parser.add_argument('--i_testset_all',   type=int, default=20000,
                        help='frequency of render_poses video saving')
    parser.add_argument('--N_iters', type=int, default=150001,
                        help='number of training iterations')
    parser.add_argument('--N_static_iters', type=int, default=150001,
                        help='number of training iterations')
                        
    # Dynamic NeRF lambdas
    parser.add_argument('--dynamic_loss_lambda', type=float, default=1.,
                        help='lambda of dynamic loss')
    parser.add_argument('--static_loss_lambda', type=float, default=1.,
                        help='lambda of static loss')
    parser.add_argument('--full_loss_lambda', type=float, default=3.,
                        help='lambda of full loss')
    parser.add_argument('--depth_loss_lambda', type=float, default=0.04,
                        help='lambda of depth loss')
    parser.add_argument('--feature_loss_lambda', type=float, default=0.01,
                        help='lambda of order loss')
    parser.add_argument('--order_loss_lambda', type=float, default=0.1,
                        help='lambda of order loss')
    parser.add_argument('--flow_loss_lambda', type=float, default=0.02,
                        help='lambda of optical flow loss')
    parser.add_argument('--slow_loss_lambda', type=float, default=0.01,
                        help='lambda of sf slow regularization')
    parser.add_argument('--smooth_loss_lambda', type=float, default=0.1,
                        help='lambda of sf smooth regularization')
    parser.add_argument('--consistency_loss_lambda', type=float, default=1.0,
                        help='lambda of sf cycle consistency regularization')
    parser.add_argument('--mask_loss_lambda', type=float, default=0.1,
                        help='lambda of the mask loss')
    parser.add_argument('--sparse_loss_lambda', type=float, default=0.001,
                        help='lambda of sparse loss')
    parser.add_argument('--DyNeRF_blending', action='store_true', default=True,
                        help='use Dynamic NeRF to predict blending weight')
    parser.add_argument('--pretrain', action='store_true', default=True,
                        help='Pretrain the StaticneRF')
    parser.add_argument('--ft_path_S', type=str, default='logs/tfr_Balloon2_H270_DyNeRF_pretrain/Pretrained_S.tar',
                        help='specific weights npy file to reload for StaticNeRF')
    parser.add_argument('--ft_S', action='store_true',
                        help='Finetune the StaticNeRF')
    # For rendering teasers
    parser.add_argument('--frame2dolly', type=int, default=-1,
                        help='choose frame to perform dolly zoom')
    parser.add_argument('--x_trans_multiplier', type=float, default=1.,
                        help='x_trans_multiplier')
    parser.add_argument('--y_trans_multiplier', type=float, default=0.33,
                        help='y_trans_multiplier')
    parser.add_argument('--z_trans_multiplier', type=float, default=5.,
                        help='z_trans_multiplier')
    parser.add_argument('--num_novelviews', type=int, default=60,
                        help='num_novelviews')
    parser.add_argument('--focal_decrease', type=float, default=200,
                        help='focal_decrease')

    # For Video Multiscenes
    parser.add_argument('--dataset_file_lists', default=['data/Balloon2', 'data/Balloon1', 'data/Truck'],
                        help='Freeze encoder weights and only train MLP',
    )
    parser.add_argument('--blending_thickness', default=0.06,
                        help='lambda of depth blending loss',
    )

    # For DDP
    parser.add_argument("--launcher", type=str, default="slurm")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12345")
    parser.add_argument("--gpus_ids", type=str, default="0")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--deterministic", type=bool, default=True)

    # From PixelNeRF
    parser.add_argument('--conf', '-c', type=str, default='pixelnerf_conf/exp/dynamic.conf')
    parser.add_argument(
        '--gpu_id', type=str, default='0', help='GPU(s) to use, space delimited'
    )
    parser.add_argument(
        '--name', '-n', type=str, default=default_expname, help='experiment name'
    )
    parser.add_argument(
        '--dataset_format',
        '-F',
        type=str,
        default='monocular',
        help='Dataset format, multi_obj | dvr | dvr_gen | dvr_dtu | srn | monocular',
    )
    parser.add_argument(
        '--exp_group_name',
        '-G',
        type=str,
        default=None,
        help='if we want to group some experiments together',
    )
    parser.add_argument(
        '--logs_path', type=str, default='logs', help='logs output directory',
    )

    parser.add_argument(
        '--visual_path',
        type=str,
        default='visuals',
        help='visualization output directory',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=default_num_epochs,
        help='number of epochs to train for',
    )
    parser.add_argument(
        '--gamma', type=float, default=default_gamma, help='learning rate decay factor'
    )

    parser.add_argument(
        '--batch_size', '-B', type=int, default=10, help="Object batch size ('SB')"
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        default=False,
        help='Eval model',
    )
    parser.add_argument(
        '--fast_render_iter', type=int, default=0, help="option for unseen scenes"
    )

    parser.add_argument(
        '--fast_render_basename', type=str, default=None, help="option for unseen scenes"
    )


    if callback is not None:
        parser = callback(parser)
    args = parser.parse_args()

    if args.exp_group_name is not None:
        args.logs_path = os.path.join(args.logs_path, args.exp_group_name)
        args.checkpoints_path = os.path.join(args.checkpoints_path, args.exp_group_name)
        args.visual_path = os.path.join(args.visual_path, args.exp_group_name)


    conf = ConfigFactory.parse_file(args.conf)

    if args.dataset_format is None:
        args.dataset_format = conf.get_string('data.format', default_data_format)

    args.gpu_id = list(map(int, args.gpu_id.split()))

    print('EXPERIMENT NAME:', args.name)

    print('* Config file:', args.conf)

    return args, conf
