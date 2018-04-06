# coding: utf-8


class ToYCbCr(object):
    """
    Transforms PIL's images pair to YCbCr format
    """

    def __call__(self, sample):
        return {'lr': sample['lr'].convert('YCbCr'), 'hr': sample['hr'].convert('YCbCr')}