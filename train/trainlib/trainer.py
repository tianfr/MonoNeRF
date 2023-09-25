import os
import os.path as osp
import torch
from torch.utils.tensorboard import SummaryWriter

import time
import json
from run_nerf_helpers import *
from src import util
from src.data import get_split_dataset
from src.model import make_model

import torch.distributed as dist
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmaction.apis import init_random_seed
#  set_random_seed, train_detector
from mmaction.utils import (collect_env,  get_root_logger, setup_multi_processes, build_ddp)
                        #  replace_cfg_vals, 
                        #  update_data_root)
class Trainer:
    def __init__(self, args, conf, device=None):
        self.args = args
        N_scenes = len(args.dataset_file_lists)

        assert args.no_ndc != conf['model']['use_ndc']

        if conf['model']['mlp_static']['origin_pipeline']  or conf['model']['mlp_dynamic']['origin_pipeline']:
            assert True
            # assert N_scenes == 1

        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            dist_params = dict(
                backend = args.backend,
                # port = args.master_port,
                port = 22446
            )
            if args.launcher == 'pytorch':
                del dist_params["port"]
            # re-set gpu_ids with distributed training mode
            init_dist(args.launcher, **dist_params)
            _, world_size = get_dist_info()
            print("world size: ", world_size)
            args.gpu_ids = range(world_size)
            args.N_iters = args.N_iters // max(1, min(world_size // N_scenes, 2))
            args.i_testset = args.i_testset // max(1, min(world_size // N_scenes, 2))
            args.i_video = args.i_video // max(1, min(world_size // N_scenes, 2))
            args.N_rand = args.N_rand // N_scenes
            args.expname = str(world_size) + "gpus" + args.expname
            if conf.get_bool("info.fast_render", False):
                assert args.fast_render_iter != 0
                args.expname = str(args.fast_render_iter) +"steps_" + args.expname
                if args.fast_render_basename is not None:
                    args.expname = str(args.fast_render_basename) +"_" + args.expname
                args.N_iters = args.fast_render_iter
                args.N_static_iters = args.fast_render_iter
                args.i_video = args.fast_render_iter - 1
                args.i_testset = args.fast_render_iter - 1
                args.i_testset_all = args.fast_render_iter - 1
            else:
                assert args.fast_render_iter == 0

        # Create log dir and copy the config file
        basedir = args.basedir
        expname = args.expname
        os.makedirs(osp.join(basedir, expname), exist_ok=True)

        timestamp = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
        log_file = args.basedir + args.expname +'/' + timestamp + '.log'
        # logger = get_logger(logger_file)
        self.logger = get_root_logger(log_file=log_file)

        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        self.logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info

        # log some basic info
        self.logger.info(f'Distributed training: {distributed}')

        args.device = util.get_device()
        # set random seeds
        seed = init_random_seed(args.random_seed, device=args.device)
        seed = seed + dist.get_rank()
        self.logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {args.deterministic}')

        util.set_random_seed(seed, deterministic=args.deterministic)
        args.seed = seed
        meta['seed'] = seed
        meta['exp_name'] = osp.basename(args.config)

        # Build PixelNerf network with SlowOnly encoder.
        self.net = make_model(conf['model'],  stop_encoder_grad=args.freeze_enc, use_static_resnet=True)

        train_dataset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir, file_lists=args.dataset_file_lists)
        print(
            'dset z_near {}, z_far {}, lindisp {}'.format(train_dataset.z_near, train_dataset.z_far, train_dataset.lindisp)
        )

        # For DDP
        self.rank = dist.get_rank()
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_resnet_nerf(args, self.net, self.rank)


        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.start = start
        self.grad_vars = grad_vars
        self.optimizer = optimizer

        self.expname = expname
        self.basedir = basedir
        self.timestamp = timestamp

        self.conf = conf


        # Summary writers
        self.writer = SummaryWriter(os.path.join(self.basedir, 'summaries', self.expname, self.timestamp))



        self.fixed_test = hasattr(args, "fixed_test") and args.fixed_test

        if not args.eval_only and self.rank == 0:
            f = os.path.join(self.basedir, self.expname, 'args.txt')
            with open(f, 'w') as file:
                for arg in sorted(vars(args)):
                    attr = getattr(args, arg)
                    file.write('{} = {}\n'.format(arg, attr))
            if args.config is not None:
                f = os.path.join(self.basedir, self.expname, 'dynamic_config.txt')
                with open(f, 'w') as file:
                    file.write(open(args.config, 'r').read())
            f = os.path.join(self.basedir, self.expname, 'pixelnerf_config.json')
            with open(f, 'w') as file:
                conf_dict = json.dumps(conf, sort_keys=False, indent=4, separators=(',', ': '))
                file.write(conf_dict)

        self.start_iter_id = 0

    def post_batch(self, epoch, batch):
        """
        Ran after each batch
        """
        pass

    def extra_save_state(self):
        """
        Ran at each save step for saving extra state
        """
        pass

    def train_step(self, data, global_step):
        """
        Training step
        """
        raise NotImplementedError()

    def eval_step(self, data, global_step):
        """
        Evaluation step
        """
        raise NotImplementedError()

    def vis_step(self, data, global_step):
        """
        Visualization step
        """
        return None, None


    def start_train(self):
        step_id = self.start_iter_id
        for data in self.train_data_loader:
            self.train_step(data, global_step=step_id)
            return 
        