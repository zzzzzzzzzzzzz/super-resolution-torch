# coding: utf-8


class ToRGB(object):
    """
    Transforms PIL's image to RGB format
    """

    def __call__(self, sample):
        return sample.convert('RGB')