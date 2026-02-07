import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
import math
import io
import os

from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 256
HEATMAP_SIZE = 64
TRAIN = False

# Residual Block

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + residual)


# Hourglass Module

class HourglassModule(nn.Module):
    def __init__(self, channels, depth, num_blocks):
        super().__init__()
        self.depth = depth
        self.channels = channels
        self.num_blocks = num_blocks

        # Upper Branch
        self.upper_branch = self._make_residual_blocks(num_blocks)

        # Lower branch
        self.lower_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            *self._make_residual_blocks(num_blocks)
        )

        # Hourglass
        if depth > 1:
            self.nested_hourglass = HourglassModule(channels, depth-1, num_blocks)
        else:
            self.nested_hourglass = nn.Sequential(
                *self._make_residual_blocks(num_blocks)
            )

        # Post hourglass residual blocks
        self.post_residual = self._make_residual_blocks(num_blocks)

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_residual_blocks(self, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(self.channels, self.channels))
        return nn.Sequential(*blocks)

    def forward(self, x):
        # Upper branch)
        up1 = self.upper_branch(x)

        # Lower branch
        low1 = self.lower_branch(x)

        # Nested hourglass or residual blocks
        low2 = self.nested_hourglass(low1)

        # Post hourglass processing
        low3 = self.post_residual(low2)

        # Upsample and combine
        up2 = self.upsample(low3)

        return up1 + up2


class LinearLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Stacked Hourglass Network

class StackedHourglassNetwork(nn.Module):
    """
    https://arxiv.org/abs/1603.06937
    """
    def __init__(self, in_channels=3, num_stacks=4, num_blocks=1,
                 num_heatmaps=5, hourglass_depth=4):
        super().__init__()
        self.num_stacks = num_stacks

        # Initial processing
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.res1 = ResidualBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, 256)

        # Hourglass stacks
        self.hourglasses = nn.ModuleList()
        self.resblock_after_hourglass = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.heatmap_convs = nn.ModuleList()
        self.intermediate_convs1 = nn.ModuleList()
        self.intermediate_convs2 = nn.ModuleList()

        for i in range(num_stacks):
            # Hourglass module
            hourglass = HourglassModule(256, hourglass_depth, num_blocks)
            self.hourglasses.append(hourglass)

            # Residual blocks after hourglass
            # for j in range(num_blocks):
            #     setattr(self, f'res_stack_{i}_{j}', ResidualBlock(256, 256))
            self.resblock_after_hourglass.append(nn.Sequential(*[ResidualBlock(256, 256) for j in range(num_blocks)]))

            # Linear layer
            linear = LinearLayer(256, 256)
            self.linears.append(linear)

            # Heatmap prediction
            heatmap_conv = nn.Conv2d(256, num_heatmaps, kernel_size=1)
            self.heatmap_convs.append(heatmap_conv)

            # Intermediate connections for next stack (except last)
            if i < num_stacks - 1:
                self.intermediate_convs1.append(
                    nn.Conv2d(256, 256, kernel_size=1)
                )
                self.intermediate_convs2.append(
                    nn.Conv2d(num_heatmaps, 256, kernel_size=1)
                )

    def forward(self, x):
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.res3(x)

        # Stacked hourglasses
        heatmaps = []

        for i in range(self.num_stacks):
            # Hourglass
            x = self.hourglasses[i](x)

            # Residual blocks
            # for j in range(len(self.hourglasses[i].upper_branch)):
            #     x = getattr(self, f'res_stack_{i}_{j}')(x)
            x = self.resblock_after_hourglass[i](x)

            # Linear layer
            x_linear = self.linears[i](x)

            # Heatmap prediction
            heatmap = self.heatmap_convs[i](x_linear)
            heatmaps.append(heatmap)

            # Prepare for next stack (if not the last)
            if i < self.num_stacks - 1:
                inter1 = self.intermediate_convs1[i](x_linear)
                inter2 = self.intermediate_convs2[i](heatmap)
                x = inter1 + inter2

        return heatmaps


