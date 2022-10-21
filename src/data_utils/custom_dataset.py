import os
import numpy as np
from glob import glob
import PIL
from PIL import Image

from collections import namedtuple
from typing import Any, Callable, List, Optional, Union, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CelebA

class PatchedCelebA(CelebA):
    # patch incorrect selection of valid and test in 0.10.0
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        assert(split in ["train", "valid", "test", "all"])
        split_ = split_map[split]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header
    

class ImageNetAnimal50(Dataset):
    
    def __init__(self, root_dir, split, transform=None):
        assert(split in ['train', 'valid', 'test'])
        imagenet_dir = os.path.join(root_dir, 'imagenet')
        
        with open(os.path.join(imagenet_dir, 'subset.txt'), 'r') as f:
            self.subset_classes = np.sort([l.split(' ')[0] for l in f.readlines()])
        
        self.data = []
        
        if split == 'train':
            train_dir = os.path.join(imagenet_dir, 'ILSVRC/Data/CLS-LOC/train')
            for i, subset_class in enumerate(self.subset_classes):
                img_fnames = glob(os.path.join(train_dir, subset_class, '*.JPEG'))
                self.data += [(img_fname, i) for img_fname in img_fnames] 
        elif split == 'valid':
            valid_dir = os.path.join(imagenet_dir, 'ILSVRC/Data/CLS-LOC/val')
            with open(os.path.join(imagenet_dir, 'LOC_val_solution.csv'), 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    image_id, pred_string = line.split(',')
                    image_class = pred_string.split(' ')[0]
                    if image_class in self.subset_classes:
                        image_fname = os.path.join(valid_dir, f'{image_id}.JPEG')
                        label = np.where(self.subset_classes == image_class)[0][0]
                        self.data.append((image_fname, label))
                        
        else:
            raise NotImplementedError
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_fname, label = self.data[idx]
        image = PIL.Image.open(img_fname).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label