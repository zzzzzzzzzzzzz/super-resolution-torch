# coding: utf-8
import numpy as np


def ret_max(func):
    """
    Decorator.
    Will return maximum from array-like return of another function.
    If the output of the wrapped function is not array this decorator
    does nothing
    :param func:
    :return:
    """

    def wrapper(image_original: np.ndarray, image_restored: np.ndarray, dynamic_range=255):
        res = func(image_original, image_restored, dynamic_range=255)
        if isinstance(res, list) or isinstance(res, np.array):
            return np.max(res)
        else:
            return res

    return wrapper


def ret_mean(func):
    """
    Decorator.
    Will return mean from array-like return of another function.
    If the output of the wrapped function is not array this decorator
    does nothing
    :param func:
    :return:
    """

    def wrapper(image_original: np.ndarray, image_restored: np.ndarray, dynamic_range=255):
        res = func(image_original, image_restored, dynamic_range=255)
        if isinstance(res, list) or isinstance(res, np.array):
            return np.mean(res)
        else:
            return res

    return wrapper


def ssim(image_original: np.ndarray, image_restored: np.ndarray, dynamic_range=255):
    check_shape_len = lambda img: len(img.shpae) == 3
    check_shape_order = lambda img: (img.shape[0] < img.shape[1]) and (img.shape[0] < img[2])

    assert check_shape_len(image_original), "Original image must be in (C,H,W) format"
    assert check_shape_order(image_original), "Original image must be in (C,H,W) format"
    assert check_shape_len(image_restored), "Restored image must be in (C,H,W) format"
    assert check_shape_order(image_restored), "Restored image must be in (C,H,W) format"
    assert (image_original.shape[0] == image_restored.shape[0]) and \
           (image_original.shape[1] == image_restored.shape[1]) and \
           (image_original.shape[2] == image_restored.shape[2]), "Original and restored images shapes are not equal"

    ssims = []
    ssim_lambda = lambda mux, muy, variancex, variancey, covxy, c1, c2: ((2 * mux * muy + c1) * (2 * covxy + c2)) / (
        (mux ** 2 + muy ** 2 + c1) * (variancex + variancey + c2))
    for i in range(image_original.shape[0]):
        mux = np.mean(image_original[i])
        muy = np.mean(image_restored[i])
        cov_matrix = np.cov([image_original[i].flatten(), image_restored[i].flatten()])
        variancex = cov_matrix[0][0]
        variancey = cov_matrix[1][1]
        covxy = cov_matrix[0][1]
        c1 = (0.01 * dynamic_range) ** 2
        c2 = (0.03 * dynamic_range) ** 2
        ssims.append(ssim_lambda(mux, muy, variancex, variancey, covxy, c1, c2))

    return ssims