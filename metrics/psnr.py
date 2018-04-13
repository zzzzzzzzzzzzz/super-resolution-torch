# coding: utf-8
import numpy as np


def psnr(image_original: np.ndarray, image_restored: np.ndarray):
    """
    Computes psnr metric. More here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    :param image_original:
    :param image_restored:
    :return:
    """
    check_shape_len = lambda img: len(img.shape) == 3
    check_shape_order = lambda img: (img.shape[0] < img.shape[1]) and (img.shape[0] < img.shape[2])

    assert check_shape_len(image_original), "Original image must be in (C,H,W) format"
    assert check_shape_order(image_original), "Original image must be in (C,H,W) format"
    assert check_shape_len(image_restored), "Restored image must be in (C,H,W) format"
    assert check_shape_order(image_restored), "Restored image must be in (C,H,W) format"
    assert (image_original.shape[0] == image_restored.shape[0]) and \
           (image_original.shape[1] == image_restored.shape[1]) and \
           (image_original.shape[2] == image_restored.shape[2]), "Original and restored images shapes are not equal"

    denominator = 0.0
    for i in range(image_original.shape[0]):
        denominator += np.sum((image_original[i] - image_restored[i]) ** 2)

    denominator = np.sqrt((1.0 / 3*image_original.shape[1]*image_original.shape[2])*denominator)
    numerator = 255
    val = 20 * np.log10(numerator/denominator)
    return val
