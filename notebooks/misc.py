import os
import sys
if os.path.abspath('../src/') not in sys.path:
    sys.path.insert(0, os.path.abspath('../src/'))
import json
import glob
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from scipy.special import softmax

import torch
from torch.nn import Parameter as P
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.misc import dict2clsattr
from utils.sample import sample_latents

def build_cache(runname, sampler, cache_dir, conditional, which_epoch='best', n_samples=50000, 
                proj_model=None, verbose=False):
    cache_path = os.path.join(cache_dir, runname)
    os.makedirs(cache_path, exist_ok=True)
    
    if verbose:
        print(f"cache to {os.path.join(cache_path, f'{which_epoch}.npz')}")
    
    if proj_model is None:
        xs, ys = [], []
        n_samples_now = 0
        with torch.no_grad():
            for x, y in sampler:
                xs.append(x.detach().cpu().numpy())
                ys.append(y.detach().cpu().numpy())
                n_samples_now += x.shape[0]
                
                if n_samples_now >= n_samples:
                    break
        xs = np.concatenate(xs, axis=0)[:n_samples]
        ys = np.concatenate(ys, axis=0)[:n_samples]
        
        np.savez_compressed(os.path.join(cache_path, f'{which_epoch}.npz'), 
                            **{'X': xs, 'Y': ys})
        
        if verbose:
            print(xs.shape, ys.shape)
        
        del xs, ys
    else:   
        (xs, ys), feats, _ = project(sampler, proj_model, n_samples=n_samples, to_numpy=True, 
                                     return_logits=False if conditional else None)
        
        proj_model_name = proj_model.__class__.__name__
        np.savez_compressed(os.path.join(cache_path, f'{which_epoch}.npz'), 
                            **{'X': xs, 'Y': ys, proj_model_name: feats})
        
        if verbose:
            print(xs.shape, ys.shape, feats.shape)
            
        del xs, ys, feats, _

def wrapper_all(runname, dset_name, **kwargs):
    if len(glob.glob(f'../checkpoints/{runname}/model=G_ema-weights-step=*.pth')) > 0:
        g_paths = glob.glob(f'../checkpoints/{runname}/model=G_ema-weights-step=*.pth')
    else:
        g_paths = glob.glob(f'../checkpoints/{runname}/model=G-weights-step=*.pth')
    config_path = f"../src/configs/{dset_name}/{runname.split('-')[0]}.json"
    samplers = {int(g_path.split('/')[-1].split('=')[-1].split('.')[0]): 
                construct_sampler(g_path, config_path, **kwargs) 
                for g_path in g_paths}
    return samplers

def wrapper_best(runname, dset_name, **kwargs):
    if len(glob.glob(f'../checkpoints/{runname}/model=G_ema-best-weights-step=*.pth')) > 0:
        ema_g_path = glob.glob(f'../checkpoints/{runname}/model=G_ema-best-weights-step=*.pth')[0]
    else:
        ema_g_path = glob.glob(f'../checkpoints/{runname}/model=G-best-weights-step=*.pth')[0]
    config_path = f"../src/configs/{dset_name}/{runname.split('-')[0]}.json"
    sampler = construct_sampler(ema_g_path, config_path, **kwargs)
    return sampler

def visualize_samples(sampler):
    with torch.no_grad():
        x, y = next(sampler)
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2
    n_dims = np.floor(np.sqrt(x.shape[0])).astype(np.int)
    margin = 5
    img = np.zeros((x.shape[1] * n_dims + margin * (n_dims - 1),
                    x.shape[1] * n_dims + margin * (n_dims - 1),
                    x.shape[-1]))

    for i, x_ in enumerate(x):
        c, r = i % n_dims, i // n_dims
        img[c * (margin + x.shape[1]):c * margin + (c + 1) * x.shape[1], 
            r * (margin + x.shape[2]):r * margin + (r + 1) * x.shape[2]] = x_
        if i == n_dims**2 - 1:
            break
    plt.figure(figsize=(20, 20))
    plt.imshow(img)

def visualize_1nns(sampler, dl, proj_model, conditional=True, return_indices=False):
    ref_x, ref_feat, _ = project(dl, proj_model, n_samples=None, to_numpy=True,
                                 return_logits=None if not conditional else True)
    
    x, gen_feat, _ = project(sampler, proj_model, n_samples=50, to_numpy=True,
                             return_logits=None if not conditional else True)
    x = (x.transpose(0, 2, 3, 1) + 1) / 2
    
    nnds, nn_indices = calculate_knnd_numpy(target_feats=gen_feat, ref_feats=ref_feat)
    
    n_dims = np.floor(np.sqrt(x.shape[0])).astype(np.int)
    margin = 5
    img = np.zeros((x.shape[1] * n_dims + margin * (n_dims - 1),
                    x.shape[2] * n_dims * 2 + margin * (n_dims - 1),
                    x.shape[-1]))
    
    for i, (x_, nn_index) in enumerate(zip(x, nn_indices)):
        nn_img = (ref_x[nn_index].transpose(1, 2, 0) + 1) / 2
        c, r = i % n_dims, i // n_dims
        img[c * margin + c * x.shape[1]:c * margin + (c + 1) * x.shape[1], 
            r * margin + 2 * r * x.shape[2]:r * margin + (2 * r + 1) * x.shape[2]] = x_
        img[c * margin + c * x.shape[1]:c * margin + (c + 1) * x.shape[1], 
            r * margin + (2 * r + 1) * x.shape[2]:r * margin + (2 * r + 2) * x.shape[2]] = nn_img
        if i == n_dims**2 - 1:
            break
    plt.figure(figsize=(20, 40))
    plt.imshow(img)
    
    if return_indices:
        return nnds, nn_indices
    else:
        return nnds

def project(generator, model, n_samples, to_numpy=True, return_logits=True):
    model = model.eval()
    device = next(model.parameters()).device # https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180/9
    xs, ys, embs, logits = [], [], [], []
    n_samples_now = 0
    with torch.no_grad():
        for x, y in generator:
            x = x.to(device) # expect value range = [-1, 1]
            if return_logits is None:
                _embs = model(x).detach()
                if to_numpy:
                    x, y, _embs = x.cpu().numpy(), y.cpu().numpy(), _embs.cpu().numpy()
            else:
                _embs, _logits = model(x)
                _embs, _logits = _embs.detach(), _logits.detach()
                if to_numpy:
                    x, y, _embs, _logits = x.cpu().numpy(), y.cpu().numpy(), _embs.cpu().numpy(), _logits.cpu().numpy()
            
            xs.append(x)
            ys.append(y)
            embs.append(_embs)
            
            if return_logits is not None:
                logits.append(_logits)
            
            n_samples_now += x.shape[0]
            if n_samples is not None and n_samples_now >= n_samples:
                # xs, embs, logits = xs[:n_samples], embs[:n_samples], logits[:n_samples]
                break
        if to_numpy:
            xs = np.concatenate(xs, axis=0)
            ys = np.concatenate(ys, axis=0)
            embs = np.concatenate(embs, axis=0)
            if return_logits is not None:
                logits = np.concatenate(logits, axis=0)
        else:
            xs = torch.cat(xs, dim=0)
            ys = torch.cat(ys, dim=0)
            embs = torch.cat(embs, dim=0)
            if return_logits is not None:
                logits = torch.cat(logits, dim=0)
    if return_logits is None:
        return (xs, ys), embs, None
    elif return_logits:
        return (xs, ys), embs, logits
    else:
        if to_numpy:
            probs = softmax(logits, axis=1)
        else:
            probs = F.softmax(logits, dim=1)
        return (xs, ys), embs, probs

def calculate_knnd_numpy(target_feats, ref_feats, k=1, return_indices=True, use_partition=True):
    target_feats = target_feats / np.linalg.norm(target_feats, axis=1, keepdims=True)
    ref_feats = ref_feats / np.linalg.norm(ref_feats, axis=1, keepdims=True)
    d = 1.0 - np.matmul(target_feats, ref_feats.T)
    # already checked has same result but partition significantly faster
    if use_partition:
        idx = np.argpartition(d, k, axis=1)[:, :k]
        sorted_part_rel_idx = np.argsort(np.take_along_axis(d, idx, axis=1), axis=1)
        idx = np.take_along_axis(idx, sorted_part_rel_idx, axis=1)
        idx = np.expand_dims(idx[:, -1], -1)
        d = np.take_along_axis(d, idx, axis=1)
        idx = np.squeeze(idx)
        d = np.squeeze(d)
    else:
        idx = np.argsort(d, axis=1)[:, k - 1]
        d = np.sort(d, axis=1)[:, k - 1]
    if return_indices:
        return d, idx
    else:
        return d

def calculate_knnd_torch(target_feats, ref_feats, k=1, return_indices=True):
    with torch.no_grad():
        target_feats = torch.div(target_feats, torch.norm(target_feats, dim=1, keepdim=True))
        ref_feats = torch.div(ref_feats, torch.norm(ref_feats, dim=1, keepdim=True))
        d = 1.0 - torch.mm(target_feats, ref_feats.T)
        val, idx = torch.topk(d, k, largest=False, dim=1)
    if return_indices:
        return val[:, -1], idx[:, -1]
    else:
        return val[:, -1]

def construct_sampler(ema_g_path, config_path, device, bsize=50, sample_mode="default"):
    def _sampler():
        train_configs = _load_default_train_args()
        with open(config_path) as f:
            model_configs = json.load(f)
        cfgs = dict2clsattr(train_configs, model_configs)
        module = __import__('models.{architecture}'.format(architecture=cfgs.architecture), fromlist=['something'])
        G = module.Generator(cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.g_spectral_norm, cfgs.attention,
                             cfgs.attention_after_nth_gen_block, cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes,
                             cfgs.g_init, cfgs.G_depth, cfgs.mixed_precision)

        checkpoint = torch.load(ema_g_path)
        G.load_state_dict(checkpoint['state_dict'])
        G = G.eval().to(device)
    
        while True:
            with torch.no_grad():
                zs, fake_labels = sample_latents(dist=cfgs.prior, batch_size=bsize, dim=G.z_dim, 
                                                 truncated_factor=cfgs.truncated_factor, 
                                                 num_classes=G.num_classes, perturb=None, 
                                                 device=device, sampler=sample_mode)
                batch_images = G(zs, fake_labels, evaluation=True)
                yield batch_images, fake_labels
                
    return _sampler()

def calculate_IS(probs, splits=10):    
    scores = []
    n_images = probs.shape[0]
    for j in range(splits):
        part = probs[(j*n_images//splits): ((j+1)*n_images//splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        kl = np.exp(kl)
        scores.append(np.expand_dims(kl, 0))
    scores = np.concatenate(scores, 0)
    m_scores = np.mean(scores)
    m_std = np.std(scores)
    return m_scores, m_std

def autolabel(rects, ax, n_digits=2):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.{n_digits}f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -12),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def _load_default_train_args():
    parser = ArgumentParser(add_help=False)

    parser.add_argument('--mrt', type=float, default=0.0, help='memorization rejection threshold')

    parser.add_argument('-c', '--config_path', type=str, default='./src/configs/CIFAR10/ContraGAN.json')
    parser.add_argument('--checkpoint_folder', type=str, default=None)
    parser.add_argument('-current', '--load_current', action='store_true', help='whether you load the current or best checkpoint')
    parser.add_argument('--log_output_path', type=str, default=None)

    parser.add_argument('-DDP', '--distributed_data_parallel', action='store_true')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')

    parser.add_argument('--seed', type=int, default=-1, help='seed for generating random numbers')
    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument('-sync_bn', '--synchronized_bn', action='store_true', help='whether turn on synchronized batchnorm')
    parser.add_argument('-mpc', '--mixed_precision', action='store_true', help='whether turn on mixed precision training')
    parser.add_argument('-LARS', '--LARS_optimizer', action='store_true', help='whether turn on LARS optimizer')
    parser.add_argument('-rm_API', '--disable_debugging_API', action='store_true', help='whether disable pytorch autograd debugging mode')

    parser.add_argument('--reduce_train_dataset', type=float, default=1.0, help='control the number of train dataset')
    parser.add_argument('-stat_otf', '--bn_stat_OnTheFly', action='store_true', help='when evaluating, use the statistics of a batch')
    parser.add_argument('-std_stat', '--standing_statistics', action='store_true')
    parser.add_argument('--standing_step', type=int, default=-1, help='# of steps for accumulation batchnorm')
    parser.add_argument('--freeze_layers', type=int, default=-1, help='# of layers for freezing discriminator')

    parser.add_argument('-l', '--load_all_data_in_memory', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--eval', action='store_true')
    parser.add_argument('-s', '--save_images', action='store_true')
    parser.add_argument('-iv', '--image_visualization', action='store_true', help='select whether conduct image visualization')
    parser.add_argument('-knn', '--k_nearest_neighbor', action='store_true', help='select whether conduct k-nearest neighbor analysis')
    parser.add_argument('-itp', '--interpolation', action='store_true', help='whether conduct interpolation analysis')
    parser.add_argument('-fa', '--frequency_analysis', action='store_true', help='whether conduct frequency analysis')
    parser.add_argument('-tsne', '--tsne_analysis', action='store_true', help='whether conduct tsne analysis')
    parser.add_argument('--nrow', type=int, default=10, help='number of rows to plot image canvas')
    parser.add_argument('--ncol', type=int, default=8, help='number of cols to plot image canvas')

    parser.add_argument('--print_every', type=int, default=100, help='control log interval')
    parser.add_argument('--save_every', type=int, default=2000, help='control evaluation and save interval')
    parser.add_argument('--eval_type', type=str, default='test', help='[train/valid/test]')
    args = parser.parse_args('')
    return vars(args)

def load_model(model, pretrained_path, load_to_cpu):
    '''Load from test_widerface.py
    '''
    def check_keys(model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def remove_prefix(state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model