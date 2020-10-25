import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.append("../")
from utils.spectralpool import HartleyPool2d, CosinePool2d
from .resnet_blocks import BasicBlock, MBasicBlock, MBasicBlock2

class ResNet_Mnist(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Mnist, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class SPResNet_Mnist(nn.Module):

    def __init__(self, block, layers, num_classes=10, s_pool='hartley'):
        super(SPResNet_Mnist, self).__init__()
        self.s_pool = s_pool
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], pool_size=14)
        self.layer3 = self._make_layer(block, 32, layers[2], pool_size=7)
        self.layer4 = self._make_layer(block, 64, layers[3], pool_size=4)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, pool_size=None, stride=1):
        downsample = None
        if self.s_pool == 'hartley':
        	pool = HartleyPool2d(pool_size)
        else:
        	pool = CosinePool2d(pool_size)
        if pool_size is not None:
            downsample = nn.Sequential(
                pool,
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet16_mnist(**kwargs):
    model = ResNet_Mnist(BasicBlock, [2, 2, 2], **kwargs)
    return model

def resnet20_mnist(**kwargs):
    model = ResNet_Mnist(BasicBlock, [3, 3, 3], **kwargs)
    return model

def spresnet18_mnist(**kwargs):
    model = SPResNet_Mnist(MBasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def spresnet15_mnist(**kwargs):
    model = SPResNet_Mnist(MBasicBlock2, [2, 2, 2, 2], **kwargs)
    return model    

def spresnet21_mnist(**kwargs):
    model = SPResNet_Mnist(MBasicBlock2, [2, 3, 3, 3], **kwargs)
    return model 