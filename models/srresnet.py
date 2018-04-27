import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])

    def forward(self, x):
        return self.features(x)


class residualBlock(nn.Module):

    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)
        self.prelu = nn.PReLU()

    def forward(self, x):
        y = self.prelu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class upsampleBlock(nn.Module):
    # Implements resize-convolution

    def __init__(self, in_channels, out_channels, cuda=False):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.shuffler(self.conv(x)))


class SrResNet(nn.Module):

    def __init__(self, n_residual_blocks, upsample_factor):
        super(SrResNet, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)
        self.prelu1 = nn.PReLU()

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(int(self.upsample_factor / 2)):
            self.add_module('upsample' + str(i + 1), upsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 3, 9, stride=1, padding=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(int(self.upsample_factor / 2)):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return self.conv3(x)

