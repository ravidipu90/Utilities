#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import os
import argparse
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,  # downsample with first conv
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride!=-1 or in_channels != out_channels:
            self.shortcut.add_module('conv',nn.Conv2d(in_channels,
                                                      out_channels,
                                                      kernel_size=1,
                                                      stride=stride,  # downsample
                                                      padding=0,
                                                      bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN
            
    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)), inplace=True)
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        out=F.relu(out, inplace=True)  # apply ReLU after addition
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,  # downsample with 3x3 conv
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels,
                               out_channels*self.expansion,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.shortcut = nn.Sequential()  # identity
        if stride!=1 or in_channels != out_channels*self.expansion:
            self.shortcut.add_module('conv',nn.Conv2d(in_channels,
                                                      out_channels*self.expansion,
                                                      kernel_size=1,
                                                      stride=stride,  # downsample
                                                      padding=0,
                                                      bias=False))
            self.shortcut.add_module('bn',nn.BatchNorm2d(out_channels*self.expansion))
                                     
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))  # not apply ReLU
        out += self.shortcut(x)
        out = F.relu(out)  # apply ReLU after addition
        return out

class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_planes = 64
        
        block_type = config['block_type']
        assert block_type in ['basic', 'bottleneck'] 
        num_blocks=config['num_blocks']
        if block_type == 'basic':
            block = BasicBlock
        elif block_type=='bottleneck':
            block = Bottleneck
        num_classes=1
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.sigmoid=nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
        
    def _make_layer(self, block, planes, num_block, stride):
        strides = [stride] + [1]*(num_block-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.maxpool(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0),-1)
        out = self.sigmoid(self.fc(out))
        return out     

depth=34
    
if depth==18:
    num_blocks=[2,2,2,2]
    block='basic'
    print("Resnet18 Model")
elif depth==34:
    num_blocks=[3,4,6,3]
    block='basic'
    print("Resnet34 Model")
elif depth==50:
    num_blocks=[3,4,6,3]
    block='bottleneck'
    print("Resnet50 Model")
elif depth==101:
    num_blocks=[3,4,23,3]
    block='bottleneck'
    print("Resnet101 Model")
elif depth==152:
    num_blocks=[3,8,36,3]
    block='bottleneck'
    print("Resnet152 Model")

config=OrderedDict([('block_type', block),
                    ('num_blocks',num_blocks)])







