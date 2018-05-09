# coding: utf-8
from torchvision.transforms import functional as F


class MyCenterCrop(object):
    """
    Convert ndarrays in sample to Tensors.
    Also range will be 0.0 to 1.0. They'll be FloatTensors
    """

    def __init__(self, size=0, upSampling=2):
        self.size = size
        self.upSampling = upSampling

    def __call__(self, sample):
        lr, hr = sample['lr'], sample['hr']
        if not self.size:
            lrsh = min(lr.size) # basically it's expected to be the same
            hrsh = min(hr.size)
            lrsh = lrsh // self.upSampling
            hrsh = hrsh // self.upSampling
            assert lrsh == hrsh, "It's expected that these to images must be the same size. Lr just become low res " \
                                 "pic and hr will remain the same "
            return {
                'lr': F.center_crop(lr, lrsh),
                'hr': F.center_crop(hr, hrsh)
            }
        else:
            return {
                'lr': F.center_crop(lr, self.size),
                'hr': F.center_crop(hr, self.size)
            }
