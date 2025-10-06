import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights

class ExoplanetResNet(nn.Module):
    def __init__(self, num_classes=1, weights=ResNet18_Weights.IMAGENET1K_V1):
        super(ExoplanetResNet, self).__init__()
        self.resnet = resnet18(weights=weights)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)