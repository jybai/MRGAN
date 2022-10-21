# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/loader.py


import glob
import os
import random
import yaml
from os.path import dirname, abspath, exists, join
from torchlars import LARS

from data_utils.load_dataset import *
from metrics.inception_network import InceptionV3
from metrics.prepare_inception_moments import prepare_inception_moments
from utils.log import make_checkpoint_dir, make_logger
from utils.losses import *
from utils.load_checkpoint import load_checkpoint
from utils.misc import *
from utils.mrt import prepare_default_mmg, SplitInceptionv3, NormalizationWrapper
from utils.biggan_utils import ema, ema_DP_SyncBN
from sync_batchnorm.batchnorm import convert_model
from worker import make_worker
from models.reID import gan_proj_ft_net
from models.ae import AE
# from models.retinaface import RetinaFaceProject

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


def prepare_train_eval(local_rank, gpus_per_node, world_size, run_name, train_configs, model_configs, hdf5_path_train):
    cfgs = dict2clsattr(train_configs, model_configs)
    prev_ada_p, step, best_step = None, 0, 0
    best_fid, best_fid_checkpoint_path = None, None
    mu, sigma, inception_model = None, None, None

    if cfgs.distributed_data_parallel:
        global_rank = cfgs.nr*(gpus_per_node) + local_rank
        print("Use GPU: {} for training.".format(global_rank))
        setup(global_rank, world_size)
        torch.cuda.set_device(local_rank)
    else:
        global_rank = local_rank

    writer = SummaryWriter(log_dir=join('./logs', run_name)) if local_rank == 0 else None
    if local_rank == 0:
        logger = make_logger(run_name, None)
        logger.info('Run name : {run_name}'.format(run_name=run_name))
        logger.info(train_configs)
        logger.info(model_configs)
    else:
        logger = None

    ##### load dataset #####
    if local_rank == 0: logger.info('Load train datasets...')
    train_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=True, download=True, 
                                resize_size=cfgs.img_size, hdf5_path=hdf5_path_train,
                                random_flip=cfgs.random_flip_preprocessing)
    if cfgs.reduce_train_dataset < 1.0:
        num_train = int(cfgs.reduce_train_dataset*len(train_dataset))
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    if local_rank == 0: logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    if local_rank == 0: logger.info('Load {mode} datasets...'.format(mode=cfgs.eval_type))
    eval_mode = True if cfgs.eval_type == 'train' else False
    eval_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=eval_mode, download=True, 
                               resize_size=cfgs.img_size, hdf5_path=None, random_flip=False)
    if local_rank == 0: logger.info('Eval dataset size : {dataset_size}'.format(dataset_size=len(eval_dataset)))

    if cfgs.distributed_data_parallel:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        cfgs.batch_size = cfgs.batch_size//world_size
    else:
        train_sampler = None

    train_dataloader = DataLoader(train_dataset, batch_size=cfgs.batch_size, shuffle=(train_sampler is None), pin_memory=True,
                                  num_workers=cfgs.num_workers, sampler=train_sampler, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True,
                                 num_workers=cfgs.num_workers, drop_last=False)

    ##### prepare memorization rejection mask #####
    vanilla_train_dset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=True, download=True, 
                                     resize_size=cfgs.img_size, hdf5_path=hdf5_path_train, normalize=False) # range=[0, 1]

    if cfgs.train_configs['mr'] is not None or cfgs.train_configs['mo'] is not None:
        if cfgs.train_configs['mr'] is not None and cfgs.train_configs['mo'] is not None:
            raise Exception("Should not apply memorization rejection and optimzation at the same time")

        if cfgs.train_configs['mr_model'] == 'imagenet_inception_v3':
            proj_model = NormalizationWrapper(SplitInceptionv3(), 
                                              mu=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225], 
                                              img_dim=299).eval().to(local_rank)
        elif cfgs.train_configs['mr_model'] == 'reID_resnet50':
            with open('Person_reID_baseline_pytorch/model/ft_ResNet50/opts.yaml', 'r') as stream:
                config = yaml.safe_load(stream)
            proj_model = gan_proj_ft_net(config['nclasses'], stride=config['stride'])
            proj_model.load_state_dict(torch.load('Person_reID_baseline_pytorch/model/ft_ResNet50/net_last.pth'))
            proj_model = proj_model.eval().to(local_rank)
        elif cfgs.train_configs['mr_model'] == 'celeba128_ae':
            proj_model = AE(in_channels=3, latent_dim=128, hidden_dims=None, img_size=128, scale=True)
            proj_model.load_state_dict({'.'.join(k.split('.')[1:]): v 
                            for k, v in torch.load('./proj_model_ckpts/celeba128_ae.pth')['state_dict'].items()})
            proj_model = proj_model.eval().to(local_rank)
        else:
            raise NotImplementedError

        with torch.no_grad():
            t = cfgs.train_configs['mrt'] if cfgs.train_configs['mr'] is not None else cfgs.train_configs['mot']
            mmg = prepare_default_mmg(proj_model, mrt=t, mrq=None, device=local_rank, 
                                      vanilla_train_dset=vanilla_train_dset)
    else:
        mmg = None

    ##### build model #####
    if local_rank == 0: logger.info('Build model...')
    module = __import__('models.{architecture}'.format(architecture=cfgs.architecture), fromlist=['something'])
    if local_rank == 0: logger.info('Modules are located on models.{architecture}.'.format(architecture=cfgs.architecture))
    Gen = module.Generator(cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.g_spectral_norm, 
                           cfgs.attention, cfgs.attention_after_nth_gen_block, cfgs.activation_fn, 
                           cfgs.conditional_strategy, cfgs.num_classes, cfgs.g_init, cfgs.G_depth,
                           cfgs.mixed_precision).to(local_rank)

    Dis = module.Discriminator(cfgs.img_size, cfgs.d_conv_dim, cfgs.d_spectral_norm, cfgs.attention, 
                               cfgs.attention_after_nth_dis_block, cfgs.activation_fn, cfgs.conditional_strategy, 
                               cfgs.hypersphere_dim, cfgs.num_classes, cfgs.nonlinear_embed, cfgs.normalize_embed, 
                               cfgs.d_init, cfgs.D_depth, cfgs.mixed_precision).to(local_rank)

    if cfgs.ema:
        if local_rank == 0: logger.info('Prepare EMA for G with decay of {}.'.format(cfgs.ema_decay))
        Gen_copy = module.Generator(cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.g_spectral_norm,
                                    cfgs.attention, cfgs.attention_after_nth_gen_block, cfgs.activation_fn, 
                                    cfgs.conditional_strategy, cfgs.num_classes, initialize=False, G_depth=cfgs.G_depth,
                                    mixed_precision=cfgs.mixed_precision).to(local_rank)
        if not cfgs.distributed_data_parallel and world_size > 1 and cfgs.synchronized_bn:
            Gen_ema = ema_DP_SyncBN(Gen, Gen_copy, cfgs.ema_decay, cfgs.ema_start)
        else:
            Gen_ema = ema(Gen, Gen_copy, cfgs.ema_decay, cfgs.ema_start)
    else:
        Gen_copy, Gen_ema = None, None

    if local_rank == 0: logger.info(count_parameters(Gen))
    if local_rank == 0: logger.info(Gen)

    if local_rank == 0: logger.info(count_parameters(Dis))
    if local_rank == 0: logger.info(Dis)


    ### define loss functions and optimizers
    G_loss = {'vanilla': loss_dcgan_gen, 'least_square': loss_lsgan_gen, 'hinge': loss_hinge_gen, 'wasserstein': loss_wgan_gen}
    D_loss = {'vanilla': loss_dcgan_dis, 'least_square': loss_lsgan_dis, 'hinge': loss_hinge_dis, 'wasserstein': loss_wgan_dis}

    if cfgs.optimizer == "SGD":
        G_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, 
                                      momentum=cfgs.momentum, nesterov=cfgs.nesterov)
        D_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, 
                                      momentum=cfgs.momentum, nesterov=cfgs.nesterov)
    elif cfgs.optimizer == "RMSprop":
        G_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, 
                                          momentum=cfgs.momentum, alpha=cfgs.alpha)
        D_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, 
                                          momentum=cfgs.momentum, alpha=cfgs.alpha)
    elif cfgs.optimizer == "Adam":
        G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Gen.parameters()), 
                                       cfgs.g_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)
        D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Dis.parameters()), 
                                       cfgs.d_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)
    else:
        raise NotImplementedError

    if cfgs.LARS_optimizer:
        G_optimizer = LARS(optimizer=G_optimizer, eps=1e-8, trust_coef=0.001)
        D_optimizer = LARS(optimizer=D_optimizer, eps=1e-8, trust_coef=0.001)

    ##### load checkpoints if needed #####
    if cfgs.checkpoint_folder is None:
        checkpoint_dir = make_checkpoint_dir(cfgs.checkpoint_folder, run_name)
    else:
        when = "current" if cfgs.load_current is True else "best"
        if not exists(abspath(cfgs.checkpoint_folder)):
            raise NotADirectoryError
        checkpoint_dir = make_checkpoint_dir(cfgs.checkpoint_folder, run_name)
        g_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=G-{when}-weights-step*.pth".format(when=when)))[0]
        d_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=D-{when}-weights-step*.pth".format(when=when)))[0]
        Gen, G_optimizer, trained_seed, run_name, step, prev_ada_p = load_checkpoint(Gen, G_optimizer, g_checkpoint_dir)
        Dis, D_optimizer, trained_seed, run_name, step, prev_ada_p, best_step, best_fid, best_fid_checkpoint_path =\
            load_checkpoint(Dis, D_optimizer, d_checkpoint_dir, metric=True)
        if local_rank == 0: logger = make_logger(run_name, None)
        if cfgs.ema:
            g_ema_checkpoint_dir = glob.glob(join(checkpoint_dir, "model=G_ema-{when}-weights-step*.pth".format(when=when)))[0]
            Gen_copy = load_checkpoint(Gen_copy, None, g_ema_checkpoint_dir, ema=True)
            Gen_ema.source, Gen_ema.target = Gen, Gen_copy

        writer = SummaryWriter(log_dir=join('./logs', run_name)) if global_rank == 0 else None
        if cfgs.train_configs['train']:
            assert cfgs.seed == trained_seed, "Seed for sampling random numbers should be same!"

        if local_rank == 0: logger.info('Generator checkpoint is {}'.format(g_checkpoint_dir))
        if local_rank == 0: logger.info('Discriminator checkpoint is {}'.format(d_checkpoint_dir))
        if cfgs.freeze_layers > -1 :
            prev_ada_p, step, best_step, best_fid, best_fid_checkpoint_path = None, 0, 0, None, None


    ##### wrap models with DP and convert BN to Sync BN #####
    if world_size > 1:
        if cfgs.distributed_data_parallel:
            if cfgs.synchronized_bn:
                process_group = torch.distributed.new_group([w for w in range(world_size)])
                Gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Gen, process_group)
                Dis = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Dis, process_group)
                if cfgs.ema:
                    Gen_copy = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Gen_copy, process_group)

            Gen = DDP(Gen, device_ids=[local_rank])
            Dis = DDP(Dis, device_ids=[local_rank])
            if cfgs.ema:
                Gen_copy = DDP(Gen_copy, device_ids=[local_rank])
        else:
            Gen = DataParallel(Gen, output_device=local_rank)
            Dis = DataParallel(Dis, output_device=local_rank)
            if cfgs.ema:
                Gen_copy = DataParallel(Gen_copy, output_device=local_rank)

            if cfgs.synchronized_bn:
                Gen = convert_model(Gen).to(local_rank)
                Dis = convert_model(Dis).to(local_rank)
                if cfgs.ema:
                    Gen_copy = convert_model(Gen_copy).to(local_rank)

    ##### load the inception network and prepare first/secend moments for calculating FID #####
    if cfgs.eval:
        inception_model = InceptionV3().to(local_rank)
        if world_size > 1 and cfgs.distributed_data_parallel:
            toggle_grad(inception_model, on=True)
            inception_model = DDP(inception_model, device_ids=[local_rank], 
                                  broadcast_buffers=False, find_unused_parameters=True)
        elif world_size > 1 and cfgs.distributed_data_parallel is False:
            inception_model = DataParallel(inception_model, output_device=local_rank)
        else:
            pass

        mu, sigma = prepare_inception_moments(dataloader=eval_dataloader,
                                              generator=Gen,
                                              eval_mode=cfgs.eval_type,
                                              inception_model=inception_model,
                                              splits=1,
                                              run_name=run_name,
                                              logger=logger,
                                              device=local_rank)

    worker = make_worker(
        cfgs=cfgs,
        run_name=run_name,
        best_step=best_step,
        logger=logger,
        writer=writer,
        n_gpus=world_size,
        gen_model=Gen,
        dis_model=Dis,
        inception_model=inception_model,
        Gen_copy=Gen_copy,
        Gen_ema=Gen_ema,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        mmg=mmg,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        G_optimizer=G_optimizer,
        D_optimizer=D_optimizer,
        G_loss=G_loss[cfgs.adv_loss],
        D_loss=D_loss[cfgs.adv_loss],
        prev_ada_p=prev_ada_p,
        global_rank=global_rank,
        local_rank=local_rank,
        bn_stat_OnTheFly=cfgs.bn_stat_OnTheFly,
        checkpoint_dir=checkpoint_dir,
        mu=mu,
        sigma=sigma,
        best_fid=best_fid,
        best_fid_checkpoint_path=best_fid_checkpoint_path,
    )

    if cfgs.train_configs['train']:
        step = worker.train(current_step=step, total_step=cfgs.total_step)

    if cfgs.eval:
        is_save = worker.evaluation(step=step, 
                                    standing_statistics=cfgs.standing_statistics, 
                                    standing_step=cfgs.standing_step)

    if cfgs.save_images:
        worker.save_images(is_generate=True, png=True, npz=True, 
                           standing_statistics=cfgs.standing_statistics, 
                           standing_step=cfgs.standing_step)

    if cfgs.image_visualization:
        worker.run_image_visualization(nrow=cfgs.nrow, ncol=cfgs.ncol, 
                                       standing_statistics=cfgs.standing_statistics, 
                                       standing_step=cfgs.standing_step)

    if cfgs.k_nearest_neighbor:
        worker.run_nearest_neighbor(nrow=cfgs.nrow, ncol=cfgs.ncol, 
                                    standing_statistics=cfgs.standing_statistics, 
                                    standing_step=cfgs.standing_step)

    if cfgs.interpolation:
        assert cfgs.architecture in ["big_resnet", "biggan_deep"], "StudioGAN does not support interpolation analysis except for biggan and biggan_deep."
        worker.run_linear_interpolation(nrow=cfgs.nrow, ncol=cfgs.ncol, fix_z=True, fix_y=False,
                                        standing_statistics=cfgs.standing_statistics, 
                                        standing_step=cfgs.standing_step)
        worker.run_linear_interpolation(nrow=cfgs.nrow, ncol=cfgs.ncol, fix_z=False, fix_y=True,
                                        standing_statistics=cfgs.standing_statistics, 
                                        standing_step=cfgs.standing_step)

    if cfgs.frequency_analysis:
        worker.run_frequency_analysis(num_images=len(train_dataset)//cfgs.num_classes,
                                      standing_statistics=cfgs.standing_statistics, 
                                      standing_step=cfgs.standing_step)

    if cfgs.tsne_analysis:
        worker.run_tsne(dataloader = eval_dataloader,
                        standing_statistics=cfgs.standing_statistics, 
                        standing_step=cfgs.standing_step)
