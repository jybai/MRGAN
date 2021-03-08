import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import inception_v3

import torchvision
import torchvision.transforms as transforms

class MemMaskGenerator():
    def __init__(self, t, proj_model, ref_tensor, device, q=None,
                 nnd_stats_path='/home/cybai2020/BigGAN-PyTorch/cifar10_train_nnd_stats.npz'):
        if q is None:
            self.t = t
        else:
            nnd_stats = np.load(nnd_stats_path)
            self.t = []
            for class_d in nnd_stats['d']:
                rnk = np.floor(q * len(class_d)).astype(np.int)
                self.t.append(class_d[rnk])
            print(f"Calibrated mrt = {self.t}")
            self.t = torch.tensor(self.t, requires_grad=False).to(device)

        self.proj_model = proj_model
        self.proj_model.eval()
        self.device = device
        
        self.ref_emb = self.compute_embedding(ref_tensor)
        
        self.accu_nnds = []
        self.accu_labels = []
    
    def get_running_stats(self):
        if len(self.accu_nnds) == 0:
            mar = None
            min_nnd = None
            mean_nnd = None
            median_nnd = None
        else:
            nnds = np.concatenate(self.accu_nnds, axis=0).reshape(-1)
            labels = np.concatenate(self.accu_labels, axis=0)
            t_np = self.t.cpu().numpy()
            mar = (nnds > t_np[labels]).sum() / nnds.size
            min_nnd = np.min(nnds)
            mean_nnd = np.mean(nnds)
            median_nnd = np.median(nnds)
            skew_nnd = scipy.stats.skew(nnds, axis=None)
        return {'mar': mar, 'min_nnd': min_nnd, 
                'mean_nnd': mean_nnd, 'median_nnd': median_nnd,
                'skew_nnd': skew_nnd}
        
    def reset_running_stats(self):
        self.accu_nnds = []
        self.accu_labels = []
    
    def compute_embedding(self, target_tensor, bsize=64):
        target_emb = [] 
        with torch.no_grad():
            for index in range(0, target_tensor.shape[0], bsize):
                target_batch = target_tensor[index:index + bsize].to(self.device)
                target_emb_ = self.proj_model(target_batch)
                target_emb.append(target_emb_)
            target_emb = torch.cat(target_emb, dim=0)
            target_emb = torch.div(target_emb,
                                   torch.norm(target_emb, dim=1, keepdim=True))
        return target_emb
    
    def compute_nnd(self, target_emb, return_ref_indices=False):
        with torch.no_grad():
            d = 1.0 - torch.abs(torch.mm(target_emb, self.ref_emb.T))
            min_d, min_ref_indices = torch.min(d, dim=1, keepdim=True)
        if return_ref_indices:
            return min_d, min_ref_indices
        else:
            return min_d

    def sample_nnds(self, sample, num_samples=10000):
        pass
        
    def __call__(self, target_tensor, target_labels, 
                 gated=True, accu_stats=True):
        target_emb = self.compute_embedding(target_tensor)
        nnd = self.compute_nnd(target_emb)

        if accu_stats:
            self.accu_nnds.append(nnd.detach().cpu().numpy())
            self.accu_labels.append(target_labels.detach().cpu().numpy())

        if not gated:
            return nnd
        else:
            if isinstance(self.t, float):
                t = self.t
            else:
                t = self.t[target_labels]
            mask = nnd > t
            return mask
        
class NormalizationWrapper(torch.nn.Module):
    def __init__(self, base_model, mu, std, img_dim):
        super(NormalizationWrapper, self).__init__()
        self.mu = torch.tensor(mu).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.img_dim = img_dim
        self.base_model = base_model

    def forward(self, x):
        x = (x - self.mu) / self.std
        if x.shape[2] != self.img_dim or x.shape[3] != self.img_dim:
            x = F.interpolate(x, size=(self.img_dim, self.img_dim), 
                              mode='bilinear', align_corners=True)
        return self.base_model(x)
    
    def to(self, *args, **kwargs):
        self = super(NormalizationWrapper, self).to(*args, **kwargs) 
        self.mu = self.mu.to(*args, **kwargs)
        self.std = self.std.to(*args, **kwargs)
        return self

class Identity(torch.nn.Module):
    def forward(self, x):
        return x
    
class SplitInceptionv3(torch.nn.Module):
    def __init__(self):
        super(SplitInceptionv3, self).__init__()
        self.front_half = inception_v3(pretrained=True)
        self.front_half.fc = Identity()
        self.back_half = inception_v3(pretrained=True).fc
        
    def forward(self, x, return_logits=False):
        front_out = self.front_half(x)
        if not return_logits:
            return front_out
        else:
            back_out = self.back_half(front_out)
            return front_out, back_out

def prepare_default_mmg(mrt, mrq, device, vanilla_train_dset):
    # cifar10_train = torchvision.datasets.CIFAR10(root='./data/cifar', train=True,
    #                                              download=True, transform=transforms.ToTensor())
    train_feats = torch.stack([x for x, _ in vanilla_train_dset], dim=0)
    
    inceptionv3_proj = SplitInceptionv3()
    inceptionv3_proj.eval()
    proj_model = NormalizationWrapper(inceptionv3_proj, mu=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225], img_dim=299).to(device)
    
    mmg = MemMaskGenerator(mrt, proj_model, train_feats, device, q=mrq)
    return mmg

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mmg = prepare_cifar10_default_mmg(0.12, device)
    
    cifar10_test = torchvision.datasets.CIFAR10(root='./data/cifar', train=False,
                                            download=True, transform=transforms.ToTensor())
    cifar10_test_feats = torch.stack([x for x, _ in cifar10_test], dim=0)
    print(cifar10_test_feats.shape)

    nnd = mmg(cifar10_test_feats, gated=False).cpu().numpy()
    assert(nnd.mean() == 0.15423344)

if __name__ == '__main__':
    main()
