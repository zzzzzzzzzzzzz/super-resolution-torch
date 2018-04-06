# coding: utf-8
from torchvision import transforms
from torchvision.transforms import functional as F


class MyRandomSample(transforms.RandomCrop):
    """
    Get's the same random sample from the pair of images
    """
    def __call__(self, sample):
        lr, hr = sample['lr'], sample['hr']
        if self.padding > 0:
            lr = F.pad(lr, self.padding)
            hr = F.pad(hr, self.padding)
        i, j, h, w = self.get_params(lr, self.size)
        return {'lr': F.crop(lr, i, j, h, w),
                'hr': F.crop(hr, i, j, h, w)}