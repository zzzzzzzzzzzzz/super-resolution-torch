# coding: utf-8

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class SRcnn(nn.Module):

    def __init__(self, depth_factor=1):
        super(SRcnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64*depth_factor, 9, padding=4) # pad 4
        self.conv2 = nn.Conv2d(64, 32*depth_factor, 1)
        self.conv3 = nn.Conv2d(32, 3, 5, padding=2) # pad 2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


if __name__ == '__main__':
    net = SRcnn()
    print(net)