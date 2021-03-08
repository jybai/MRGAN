import os
import sys
if os.path.abspath('../src/') not in sys.path:
    sys.path.insert(0, os.path.abspath('../src/'))
import json
import numpy as np
from argparse import ArgumentParser
from scipy.special import softmax

import torch
from torch.nn import Parameter as P
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.misc import dict2clsattr
from utils.sample import sample_latents

def project(generator, model, to_numpy=True, n_samples=10000, return_logits=True):
    model = model.eval()
    device = next(model.parameters()).device # https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180/9
    embs, logits = [], []
    n_samples_now = 0
    with torch.no_grad():
        for x, y in generator:
            x = x.to(device) # expect value range = [-1, 1]
            _embs, _logits = model(x)
            _embs, _logits = _embs.detach(), _logits.detach()
            if to_numpy:
                _embs, _logits = _embs.cpu().numpy(), _logits.cpu().numpy()
            embs.append(_embs)
            logits.append(_logits)
            
            n_samples_now += x.shape[0]
            if n_samples_now >= n_samples:
                break
        if to_numpy:
            embs = np.concatenate(embs, axis=0)
            logits = np.concatenate(logits, axis=0)
        else:
            embs = torch.cat(embs, dim=0)
            logits = torch.cat(logits, dim=0)
    if return_logits:
        return embs[:n_samples], logits[:n_samples]
    else:
        if to_numpy:
            probs = softmax(logits, axis=1)
        else:
            probs = F.softmax(logits, dim=1)
        return embs[:n_samples], probs[:n_samples]

def calculate_knnd_numpy(target_feats, ref_feats, k=1):
    target_feats = target_feats / np.linalg.norm(target_feats, axis=1, keepdims=True)
    ref_feats = ref_feats / np.linalg.norm(ref_feats, axis=1, keepdims=True)
    d = 1.0 - np.matmul(target_feats, ref_feats.T)
    idx = np.argsort(d, axis=1)
    d = np.sort(d, axis=1)
    return d[:, 0], idx[:, 0]

def calculate_knnd_torch(target_feats, ref_feats, k=1):
    with torch.no_grad():
        target_feats = torch.div(target_feats, torch.norm(target_feats, dim=1, keepdim=True))
        ref_feats = torch.div(ref_feats, torch.norm(ref_feats, dim=1, keepdim=True))
        d = 1.0 - torch.mm(target_feats, ref_feats.T)
        val, idx = torch.topk(d, k, largest=False, dim=1)
        return val[:, -1], idx[:, -1]

def construct_sampler(ema_g_path, config_path, device, bsize=50):
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
                zs, fake_labels = sample_latents(cfgs.prior, bsize, G.z_dim, cfgs.truncated_factor, 
                                                 G.num_classes, None, device)
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