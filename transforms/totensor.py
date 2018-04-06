# coding: utf-8
from torchvision import transforms
from torchvision.transforms import functional as F


class MyToTensor(transforms.ToTensor):
    """
    Convert ndarrays in sample to Tensors.
    Also range will be 0.0 to 1.0. They'll be FloatTensors
    """

    def __call__(self, sample):
        lr, hr = sample['lr'], sample['hr']
        return {'lr': F.to_tensor(lr),
                'hr': F.to_tensor(hr)}