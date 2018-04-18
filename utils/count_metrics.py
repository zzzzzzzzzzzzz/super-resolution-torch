# coding: utf-8
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

from metrics.psnr import psnr
from metrics.ssim import ssim


def count_metrics(dataset_klass, dataset_root, transform, model, cuda=False):
    dataloader = DataLoader(dataset_klass(root_dir=dataset_root, transform=transform, train=False),
                            batch_size=1, num_workers=1)
    if cuda:
        model.cuda()
    psnrs = []
    for i, data in enumerate(dataloader):
        lr, hr = data

        if cuda:
            high_res_real = Variable(hr.cuda())
            high_res_fake = model(Variable(lr).cuda())
        else:
            high_res_real = Variable(hr)
            high_res_fake = model(Variable(lr))

        psnrs.append(psnr(high_res_real.cpu().data.numpy()[0], high_res_fake.cpu().data.numpy()[0]))

    mean_psnr = np.mean(psnrs)
    ssims = []
    for i, data in enumerate(dataloader):
        lr, hr = data

        if cuda:
            high_res_real = Variable(hr.cuda())
            high_res_fake = model(Variable(lr).cuda())
        else:
            high_res_real = Variable(hr)
            high_res_fake = model(Variable(lr))

        ssims.append(ssim(high_res_real.cpu().data.numpy()[0], high_res_fake.cpu().data.numpy()[0]))

    mean_ssim = np.mean(ssims)

    return mean_psnr, mean_ssim