import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import timm
import cv2
import math
import io

from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 400
TRAIN = False


class Backbone(nn.Module):
    """Backbone NNetwork for Face Recognition (EfficientNet B3 by default)"""
    def __init__(self, model_name="efficientnet_b3", unfreeze_last=2, feat_dim=512):
        super().__init__()
        # timm.list_models(pretrained=True)
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=1)

        # Выключаем градиенты у модели, чтобы упростить обучение
        # for param in self.backbone.parameters():
        #     param.requires_grad = False


        # Размораживаем последние k блоков:
        # if unfreeze_last > 0:
        #     blocks = list(self.backbone.blocks.children())
        #     for block in blocks[-unfreeze_last:]:
        #         for param in block.parameters():
        #             param.requires_grad = True

        self.feat_emb = nn.Linear(384 * 8 * 8, feat_dim)
        # self.feat_emb = nn.Linear(32768, feat_dim)
        # self.feat_emb = nn.Linear(131072, feat_dim)

    def forward(self, x):
        x = self.backbone(x)[0]
        x = x.view(x.size(0), -1)
        x = self.feat_emb(x)

        return x


class ArcFaceLoss(nn.Module):
    def __init__(self, feat_dim=512, num_class=NUM_CLASSES, scale=32, margin=0.5, easy_margin=False):
        """
        The input of this Module should be a Tensor which size is (N, embed_size), and the size of output Tensor is (N, num_classes).

        arcface_loss =-\sum^{m}_{i=1}log
                        \frac{e^{s\psi(\theta_{i,i})}}{e^{s\psi(\theta_{i,i})}+
                        \sum^{n}_{j\neq i}e^{s\cos(\theta_{j,i})}}
        \psi(\theta)=\cos(\theta+m)
        where m = margin, s = scale
        """
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_class, feat_dim))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
       
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sin_theta = torch.sqrt((1.0 - torch.pow(cos_theta, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)

        one_hot = torch.zeros(cos_theta.size(), device='cuda')
        one_hot.scatter_(1, ground_truth.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)  
        output *= self.scale

        if self.training:
            loss = self.ce(output, ground_truth)
            return loss

        return embedding, output


class CE_loss(nn.Module):
    def __init__(self, feat_dim=512, num_class=NUM_CLASSES, seed=None):
        super().__init__()

        # Установка seed
        if seed is not None:
            torch.manual_seed(seed)

        # Классификатор
        self.classifier = nn.Linear(feat_dim, num_class)

        # Функция потерь
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, feats, labels):
        """
        Входные данные — эмбеддинги (feats) и метки классов (labels).
        """

        # Вычисление классификатора
        logits = self.classifier(feats)

        # Вычисление потерь
        if self.training:
            loss = self.loss_fn(logits, labels)
            return loss

        return feats, logits


class FaceRecorgnizer(nn.Module):
    def __init__(self, model_name="efficientnet_b3", unfreeze_last=2, loss_class=ArcFaceLoss, feat_dim=512, num_class=NUM_CLASSES, margin_arc=0.5, margin_am=0.0, scale=32, seed=None):
        super(FaceRecorgnizer, self).__init__()

        self.backbone = Backbone(model_name=model_name, unfreeze_last=unfreeze_last, feat_dim=feat_dim)

        self.loss = loss_class(feat_dim=feat_dim, num_class=num_class)

    def forward(self, x, labels):

        x = self.backbone(x)
        x = self.loss(x, labels)

        return x




