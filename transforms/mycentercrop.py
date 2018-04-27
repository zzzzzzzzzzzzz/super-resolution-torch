# coding: utf-8
from torchvision.transforms import functional as F


class MyCenterCrop(object):
    """
    Convert ndarrays in sample to Tensors.
    Also range will be 0.0 to 1.0. They'll be FloatTensors
    """

    def __init__(self, size=0):
        self.size = size

    def __call__(self, sample):
        lr, hr = sample['lr'], sample['hr']
        if not self.size:
            lrsh = min(lr.size)
            hrsh = min(hr.size)
            return {
                'lr': F.center_crop(lr, lrsh),
                'hr': F.center_crop(hr, hrsh)
            }
        else:
            return {
                'lr': F.center_crop(lr, self.size),
                'hr': F.center_crop(hr, self.size)
            }
