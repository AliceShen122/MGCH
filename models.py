#!/usr/bin/env Python
# coding=utf-8

import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc_encode = nn.Linear(4096, code_len)  # 512
        self.alpha = 1.0

    def forward(self, x):
        feat1 = F.relu(self.fc1(x))  # 4096
        feat2 = F.relu(self.fc1(feat1))  # 4096
        hid = self.fc_encode(feat2)  # 64
        code = F.tanh(self.alpha * hid)
        return feat2, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat1 = F.relu(self.fc1(x))  # 2048
        feat2 = F.relu(self.fc2(feat1))  # 4096
        hid = self.fc3(feat2)  # 64
        code = F.tanh(self.alpha * hid)
        return feat2, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
