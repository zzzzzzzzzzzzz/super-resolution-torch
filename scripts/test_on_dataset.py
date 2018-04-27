# coding: utf-8
# TODO: сделать поддержку загрузки класса модели по id эксперимента с теми параметрами, которые были при обучении
import argparse
import sqlite3
import sys

import os

import numpy as np
import torch
from PIL import Image
import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader

from init import Infrastructure
from metrics.psnr import psnr
from metrics.ssim import ssim
from models.srgan import Generator
from transforms.myrandomsample import MyRandomSample
from transforms.myresize import MyResize
from transforms.totensor import MyToTensor
from transforms.toycbcr import ToYCbCr
from utils.find_klass_in_folder import find_klass
from torchvision import transforms

from utils.save_viz import save_viz
from utils.write_metrics import count_metrics

sys.path.append('.')

if __name__ == '__main__':
    infra = Infrastructure()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='set5-test-only',
                        help='One of the datasets listed in your database')
    parser.add_argument('--experimentId', type=int, help="Experiment id")
    parser.add_argument('--imageSize', type=int, default=16, help='the low resolution image size')
    parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
    parser.add_argument('--nEpochs', type=int, help="Number of epochs to run over dataset")
    parser.add_argument('--experimentIdMd5', type=str, help="Experiment idmd5")
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    opt = parser.parse_args()
    print(opt)
    id, idmd5, cr_date = infra.init_test(opt)
    print("Searching for dataset...")
    dataset_klass = None
    dataset_root = None
    try:
        c = infra.conn.cursor()
        c.execute('''
                         SELECT classname, path FROM datasets WHERE name=?
                      ''', (opt.dataset,))
        classname, dataset_root = c.fetchone()
        dataset_klass = find_klass(infra.datasets_path, classname)
    except sqlite3.Error:
        print("Error in fetching classname from db, exiting...")
        exit(-1)

    weights_file_path = ''
    if isinstance(cr_date, datetime.datetime):
        path_to_snapshot_folder = '{}/{}/{}/'.format(infra.snapshots_path, idmd5,
                                                     cr_date.strftime(infra.DATETIME_FORMAT_STR))
    if isinstance(cr_date, str):
        path_to_snapshot_folder = '{}/{}/{}/'.format(infra.snapshots_path, idmd5,
                                                     cr_date.replace(':', '-')[:-7])  # dirty hack
    else:
        print("Couldn't recognize cr_date from database, exiting...")
        exit(-1)

    print(path_to_snapshot_folder)
    if os.path.exists(path_to_snapshot_folder):
        root_dir, dirs, files = os.walk(path_to_snapshot_folder).__next__()
        if len(files) > 1:
            print("There are many files inside this folder. Which one do you want to use to extract weights for model?")
            for f in files:
                print(f)

            chosen_file = ''
            while chosen_file not in files:
                chosen_file = str(input("Choose one of them:[{}]".format(files[0])))
                if not chosen_file:
                    chosen_file = files[0]

            weights_file_path = os.path.join(path_to_snapshot_folder, chosen_file)
        else:
            weights_file_path = os.path.join(path_to_snapshot_folder, files[0])
    else:
        print("Snapshots for this experiment don't exist =(")
        exit(-1)

    path_to_save_viz = ''
    if isinstance(cr_date, datetime.datetime):
        path_to_save_viz = '{}/{}/{}/{}/'.format(infra.tests_path, idmd5, cr_date.strftime(infra.DATETIME_FORMAT_STR),
                                                 opt.dataset)
    if isinstance(cr_date, str):
        path_to_save_viz = '{}/{}/{}/{}/'.format(infra.tests_path, idmd5, cr_date.replace(':', '-')[:-7],
                                                 opt.dataset)
    else:
        print("Couldn't recognize cr_date from database, exiting...")
        exit(-1)

    try:
        os.makedirs(path_to_save_viz)
    except OSError:
        pass

    # TODO: automatic model selection here with the proper params!
    model = Generator(5, opt.upSampling)
    model.load_state_dict(torch.load(weights_file_path))

    transform = transforms.Compose([ToYCbCr(),
                                    MyRandomSample(opt.imageSize * opt.upSampling),
                                    MyResize(opt.imageSize),
                                    MyToTensor()])

    mean_psnr, mean_ssim = count_metrics(dataset_klass, dataset_root, transform, model, opt.cuda)
    if opt.nEpochs:
        save_viz(path_to_save_viz, dataset_klass, dataset_root, transform, model, opt.cuda, opt.nEpochs)
    else:
        save_viz(path_to_save_viz, dataset_klass, dataset_root, transform, model, opt.cuda)

    global_psnrs = []
    global_ssims = []
    transform = transforms.Compose([ToYCbCr(),
                                    MyResize(factor=1 / opt.upSampling),
                                    MyToTensor()])
    dataloader = DataLoader(dataset_klass(root_dir=dataset_root, transform=transform, train=False),
                            batch_size=1, num_workers=1)
    if opt.nEpochs:
        save_viz(path_to_save_viz, dataset_klass, dataset_root, transform, model, opt.cuda, opt.nEpochs, prefix='global_')
    else:
        save_viz(path_to_save_viz, dataset_klass, dataset_root, transform, model, opt.cuda, prefix='global_')

    global_mean_psnr, global_mean_ssim = count_metrics(dataset_klass, dataset_root, transform, model, opt.cuda)

    with open(os.path.join(path_to_save_viz, 'metrics.txt'), 'w') as f:
        f.write("Mean psnr on small random samples using {} passes: {}\n".format(opt.nEpochs, mean_psnr))
        f.write("Mean ssim on small random samples using {} passes: {}\n".format(opt.nEpochs, mean_ssim))
        f.write("Mean psnr on big pictures: {}\n".format(global_mean_psnr))
        f.write("Mean ssim on big pictures: {}\n".format(global_mean_ssim))

"""
    transform = transforms.Compose([ToYCbCr(),
                                    MyToTensor()])
    dataloader = DataLoader(dataset_klass(root_dir=dataset_root, transform=transform, train=False),
                            batch_size=1, num_workers=1)
    for i, data in enumerate(dataloader):
        lr, hr = data
        topil = transforms.ToPILImage(mode='YCbCr')
        hr = topil(hr[0])
        lr = topil(lr[0])
        width, height = hr.size
        hor = width//(opt.imageSize*opt.upSampling)
        ver = height//(opt.imageSize*opt.upSampling)
        totensor = transforms.ToTensor()
        crop = transforms.CenterCrop((ver*opt.imageSize*opt.upSampling, hor*opt.imageSize*opt.upSampling))
        lr = crop(lr).resize((hor*opt.imageSize, ver*opt.imageSize))
        hr = crop(hr)
        hr_particles = []
        lr_particles = []
        for k in range(hor):
            for j in range(ver):
                particle_hr = hr.crop((k*opt.imageSize*opt.upSampling, j*opt.imageSize*opt.upSampling, opt.imageSize*opt.upSampling*(k+1), opt.imageSize*opt.upSampling*(j+1)))
                particle_lr = lr.crop((k * opt.imageSize, j * opt.imageSize, opt.imageSize * (k + 1), opt.imageSize * (j + 1)))
                hr_particles.append(totensor(particle_hr))
                lr_particles.append(totensor(particle_lr))

        hr_particles, lr_particles = torch.stack(hr_particles), torch.stack(lr_particles) # making torch tensor from list
        if opt.cuda:
            high_res_real = Variable(hr_particles.cuda())
            high_res_fake = model(Variable(lr_particles).cuda())
        else:
            high_res_real = Variable(hr_particles)
            high_res_fake = model(Variable(lr_particles))

        hrs = high_res_real.cpu().data
        fakes = high_res_fake.cpu().data
        result = Image.new('YCbCr', (hor*opt.imageSize*opt.upSampling, ver*opt.imageSize*opt.upSampling))
        for k in range(hor):
            for j in range(ver):
                fakeimg = topil(fakes[k*ver+j])
                for x in range(opt.imageSize*opt.upSampling):
                    for y in range(opt.imageSize*opt.upSampling):
                        result.putpixel((k*opt.imageSize*opt.upSampling + x, j*opt.imageSize*opt.upSampling+y), fakeimg.getpixel((x,y)))

        lr.save(os.path.join(path_to_save_viz, '{}_full_lr.jpg'.format(i)))
        hr.save(os.path.join(path_to_save_viz, '{}_full_hr.jpg'.format(i)))
        result.save(os.path.join(path_to_save_viz, '{}_full_generated.jpg'.format(i)))

        global_psnrs.append(
            np.mean(psnr(
                np.array(hr).reshape((-1, hr.size[0], hr.size[1])),
                np.array(result).reshape((-1, result.size[0], result.size[1]))
            ))
        )
        global_ssims.append(
            np.mean(ssim(
                np.array(hr).reshape((-1, hr.size[0], hr.size[1])),
                np.array(result).reshape((-1, result.size[0], result.size[1]))
            ))
        )

    with open(os.path.join(path_to_save_viz, 'metrics.txt'), 'w') as f:
        f.write("Mean psnr on small random samples using {} passes: {}\n".format(opt.nEpochs, mean_psnr))
        f.write("Mean ssim on small random samples using {} passes: {}\n".format(opt.nEpochs, mean_ssim))
        f.write("Mean psnr on big pictures, restored using small pieces: {}\n".format(np.array(global_psnrs).mean()))
        f.write("Mean ssim on big pictures, restored using small pieces: {}\n".format(np.array(global_ssims).mean()))
"""
