# coding: utf-8
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


class MyResize(transforms.Resize):
    """
    Convert ndarrays in sample to Tensors.
    Also range will be 0.0 to 1.0. They'll be FloatTensors
    """

    def __init__(self, size=0, interpolation=Image.BILINEAR, factor=0.5):
        super(MyResize, self).__init__(size, interpolation)
        self.factor = factor

    def __call__(self, sample):
        lr, hr = sample['lr'], sample['hr']
        if self.size:
            return {'lr': F.resize(lr, self.size, self.interpolation),
                    'hr': hr}
        else:
            lr_size = lr.size
            return {'lr': F.resize(lr, (int(lr_size[1]*self.factor), int(lr_size[0]*self.factor)), self.interpolation),
                    'hr': hr}