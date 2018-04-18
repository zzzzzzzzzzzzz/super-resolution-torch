# coding: utf-8

import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms


def save_viz(path, dataset_klass, dataset_root, transform, model, cuda=False, nEpochs=10):
    dataloader = DataLoader(dataset_klass(root_dir=dataset_root, transform=transform, train=False),
                            batch_size=1, num_workers=1)
    topil = transforms.ToPILImage(mode='YCbCr')
    if cuda:
        model.cuda()

    for epoch in range(nEpochs):
        for i, data in enumerate(dataloader):
            lr, hr = data

            if cuda:
                high_res_real = Variable(hr.cuda())
                high_res_fake = model(Variable(lr).cuda())
            else:
                high_res_real = Variable(hr)
                high_res_fake = model(Variable(lr))

            hrr = high_res_real.cpu().data[0]
            hrf = high_res_fake.cpu().data[0]
            lr = lr.cpu()[0]
            topil(hrr).save(os.path.join(path, '{}_{}_hr.jpg'.format(epoch, i)))
            topil(hrf).save(os.path.join(path, '{}_{}_generated.jpg'.format(epoch, i)))
            topil(lr).save(os.path.join(path, '{}_{}_lr.jpg'.format(epoch, i)))
