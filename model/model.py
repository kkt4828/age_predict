import torch
import torch.nn as nn
from torchvision import models

class SimpleNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, 10),
            nn.ReLU(),
            nn.Linear(10, out_ch)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, out_ch=1):
        super().__init__()
        self.net = models.resnet18()
        self.net.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.net.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, out_ch)
        )
        model_state_dict = torch.load('model/resnet18.pt')
        self.net.load_state_dict(model_state_dict)


    def forward(self, x):
        x = self.net(x)
        return x