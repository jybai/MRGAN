import torch
import torch.nn as nn
from torch.nn import Parameter as P
from torchvision.models.utils import load_state_dict_from_url
from typing import Any

class WrappedAlexNet(nn.Module):
    def __init__(self, net=None, custom_num_classes=None, img_size=224):
        super(WrappedAlexNet, self).__init__()
        if net is None:
            self.net = AlexNet(pretrained=True)
        else:
            self.net = net
        for param in self.net.parameters():
            param.requires_grad = False
            
        self.img_size = img_size
        self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1),
                      requires_grad=False)
        self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1),
                     requires_grad=False)
        
        if custom_num_classes is not None:
            # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#alexnet
            self.net.classifier[-1] = nn.Linear(4096, custom_num_classes)
        
        print('##### trainable parameters')
        for param in self.net.parameters():
            if param.requires_grad:
                print(param.name, end='\t')
        print('#####')

    def forward(self, x):
        # Normalize x
        x = (x + 1.) / 2.0
        x = (x - self.mean) / self.std
        # Upsample if necessary
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
        # forward of original net (seperate penultimate output)
        x = self.net.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        embs = self.net.classifier[:-1](x)
        logits = self.net.classifier[-1](embs)
        return embs, logits
    
'''Below directly copied from the link
   https://github.com/pytorch/vision/blob/973db14523d338fa4e772574fe4d49763fb25245/torchvision/models/alexnet.py
'''

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model