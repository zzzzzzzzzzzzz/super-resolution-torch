# coding: utf-8

import argparse

import os
import sqlite3
import time
import sys

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from tensorboard_logger import configure, log_value

from init import Infrastructure
from metrics.psnr import psnr
from metrics.ssim import ssim
from models.srgan import Generator, Discriminator, FeatureExtractor
from transforms.myrandomsample import MyRandomSample
from transforms.myresize import MyResize
from transforms.totensor import MyToTensor
from transforms.toycbcr import ToYCbCr
from torch.utils.data import DataLoader

from utils.find_klass_in_folder import find_klass
from visualizers.srgan_vis import Visualizer

infra = Infrastructure()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mixed-flowers-berkley', help='One of the datasets listed in your database')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=16, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='', help="path to generator weights (to continue training)")
parser.add_argument('--discriminatorWeights', type=str, default='',
                    help="path to discriminator weights (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')
parser.add_argument('--genResLayersNumber', type=int, default=5, help='number of residual blocks within generator')
parser.add_argument('--pretrainGenerator', type=int, default=1, help='Pretrain generator?')
parser.add_argument('--pretrainGeneratorEpochs', type=int, default=2, help='Number of epochs for generator pretrain')
parser.add_argument('--generatorLatentLossWeight', type=float, default=0.006, help='The weight for the loss within features extracted from vgg19')
parser.add_argument('--generatorAdversarialLossWeight', type=float, default=1e-3, help='The weight of adversarial loss (when fake data is input for discriminator with labels saying that the data is real')

opt = parser.parse_args()
print(opt)
experiment_id = infra.init_experiment(opt)
start = time.time()
print("Searching for dataset...")
dataset_klass = None
dataset_root = None
try:
    c = infra.conn.cursor()
    c.execute('''
                 SELECT classname, path FROM datasets WHERE name=?
              ''', opt.dataset)
    classname, dataset_root = c.fetchone()
    dataset_klass = find_klass(infra.datasets_path, classname)
    infra.conn.commit()
except sqlite3.Error:
    print("Error in fetching classname from db, exiting...")
    exit(-1)

print("Starting experiment {}".format(experiment_id))

try:
    os.makedirs(opt.out)
except OSError:
    pass

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = transforms.Compose([ToYCbCr(),
                                MyRandomSample(opt.imageSize*opt.upSampling),
                                MyResize(opt.imageSize),
                                MyToTensor()])

dataloader = DataLoader(dataset_klass(root_dir=dataset_root, transform=transform),
                        batch_size=opt.batchSize,
                        shuffle=True, num_workers=int(opt.workers))

generator = Generator(opt.genResLayersNumber, opt.upSampling)
if opt.generatorWeights != '':
    generator.load_state_dict(torch.load(opt.generatorWeights))
print(generator)

discriminator = Discriminator()
if opt.discriminatorWeights != '':
    discriminator.load_state_dict(torch.load(opt.discriminatorWeights))
print(discriminator)

# For the content loss
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
print(feature_extractor)
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()  # cross-entropy

ones_const = Variable(torch.ones(opt.batchSize, 1))

# if gpu is to be used
if opt.cuda:
    generator.cuda()
    discriminator.cuda()
    feature_extractor.cuda()
    content_criterion.cuda()
    adversarial_criterion.cuda()
    ones_const = ones_const.cuda()

optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)


configure('{}/{}/{}/'.format(infra.logs_path, os.path.basename(__file__), experiment_id), flush_secs=5)
visualizer = Visualizer(image_size=opt.imageSize*opt.upSampling)

# Pre-train generator using raw MSE loss
if opt.pretrainGenerator:
    sys.stdout.write('Generator pre-training')
    for epoch in range(opt.pretrainGeneratorEpochs):
        mean_generator_content_loss = 0.0

        for i, data in enumerate(dataloader):

            lr, hr = data
            # Generate real and fake inputs
            if opt.cuda:
                high_res_real = Variable(hr.cuda())
                high_res_fake = generator(Variable(lr).cuda())
            else:
                high_res_real = Variable(hr)
                high_res_fake = generator(Variable(lr))

            # Train generator
            generator.zero_grad()

            generator_content_loss = content_criterion(high_res_fake, high_res_real)
            mean_generator_content_loss += generator_content_loss.data[0]

            generator_content_loss.backward()
            optim_generator.step()

            # Logging and visualization
            sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f' % (
            epoch, opt.pretrainGeneratorEpochs, i, len(dataloader), generator_content_loss.data[0]))
            visualizer.show(lr, high_res_real.cpu().data, high_res_fake.cpu().data)

        sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f\n' % (
        epoch, opt.pretrainGeneratorEpochs, i, len(dataloader), mean_generator_content_loss / len(dataloader)))
        log_value('generator_mse_loss', mean_generator_content_loss / len(dataloader), epoch)
else:
    sys.stdout.write("Skipping generator pretrain phase")

# Do checkpointing
torch.save(generator.state_dict(), '{}/{}/generator_pretrain.pth'.format(infra.snapshots_path, experiment_id))


# SRGAN training
optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)

print('SRGAN training')
for epoch in range(opt.nEpochs):
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0

    for i, data in enumerate(dataloader):

        lr, hr = data

        # Generate real and fake inputs
        if opt.cuda:
            high_res_real = Variable(high_res_real.cuda())
            high_res_fake = generator(Variable(lr).cuda())
            # target_real = Variable(torch.rand(opt.batchSize, 1) * 0.5 + 0.7).cuda()
            target_real = Variable(torch.ones(opt.batchSize, 1)).cuda()
            # target_fake = Variable(torch.rand(opt.batchSize, 1) * 0.3).cuda()
            target_fake = Variable(torch.zeros(opt.batchSize, 1)).cuda()
        else:
            high_res_real = Variable(high_res_real)
            high_res_fake = generator(Variable(lr))
            # target_real = Variable(torch.rand(opt.batchSize, 1) * 0.5 + 0.7)
            target_real = Variable(torch.ones(opt.batchSize, 1))
            # target_fake = Variable(torch.rand(opt.batchSize, 1) * 0.3)
            target_fake = Variable(torch.zeros(opt.batchSize, 1))

        ######### Train discriminator #########
        discriminator.zero_grad()

        discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                             adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
        mean_discriminator_loss += discriminator_loss.data[0]

        discriminator_loss.backward()
        optim_discriminator.step()

        ######### Train generator #########
        generator.zero_grad()

        real_features = Variable(feature_extractor(high_res_real).data)
        fake_features = feature_extractor(high_res_fake)

        generator_content_loss = content_criterion(high_res_fake, high_res_real) + opt.generatorLatentLossWeight * content_criterion(
            fake_features, real_features)
        mean_generator_content_loss += generator_content_loss.data[0]
        generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
        mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

        generator_total_loss = generator_content_loss + opt.generatorAdversarialLossWeight * generator_adversarial_loss
        mean_generator_total_loss += generator_total_loss.data[0]

        generator_total_loss.backward()
        optim_generator.step()

        ######### Status and display #########
        sys.stdout.write(
            '\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (
            epoch, opt.nEpochs, i, len(dataloader),
            discriminator_loss.data[0], generator_content_loss.data[0], generator_adversarial_loss.data[0],
            generator_total_loss.data[0]))
        visualizer.show(lr, high_res_real.cpu().data, high_res_fake.cpu().data)

    sys.stdout.write(
        '\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (
        epoch, opt.nEpochs, i, len(dataloader),
        mean_discriminator_loss / len(dataloader), mean_generator_content_loss / len(dataloader),
        mean_generator_adversarial_loss / len(dataloader), mean_generator_total_loss / len(dataloader)))

    log_value('generator_content_loss', mean_generator_content_loss / len(dataloader), epoch)
    log_value('generator_adversarial_loss', mean_generator_adversarial_loss / len(dataloader), epoch)
    log_value('generator_total_loss', mean_generator_total_loss / len(dataloader), epoch)
    log_value('discriminator_loss', mean_discriminator_loss / len(dataloader), epoch)

    # Do checkpointing
    torch.save(generator.state_dict(), '{}/{}/generator_final.pth'.format(infra.snapshots_path, experiment_id))
    torch.save(discriminator.state_dict(), '{}/{}/discriminator_final.pth'.format(infra.snapshots_path, experiment_id))

end = time.time()

dataloader_test = DataLoader(dataset_klass(root_dir=dataset_root, transform=transform, train=False),
                        batch_size=1, num_workers=1)

psnrs = []
for i, data in enumerate(dataloader_test):
    lr, hr = data

    if opt.cuda:
        high_res_real = Variable(hr.cuda())
        high_res_fake = generator(Variable(lr).cuda())
    else:
        high_res_real = Variable(hr)
        high_res_fake = generator(Variable(lr))

    psnrs.append(psnr(high_res_real.data.numpy(), high_res_fake.data.numpy()))

mean_psnr = np.mean(psnrs)
ssims = []
for i, data in enumerate(dataloader_test):
    lr, hr = data

    if opt.cuda:
        high_res_real = Variable(hr.cuda())
        high_res_fake = generator(Variable(lr).cuda())
    else:
        high_res_real = Variable(hr)
        high_res_fake = generator(Variable(lr))

    ssims.append(ssim(high_res_real.data.numpy(), high_res_fake.data.numpy()))

mean_ssim = np.mean(ssims)

try:
    c = infra.conn.cursor()
    c.execute('''
                 INSERT INTO metric_values (experiment_id, metric_id, value) 
                 SELECT
                  t1.id as experiment_id,
                  t2.id as metric_id,
                  ?
                 FROM
                  (SELECT 
                       id 
                   FROM
                       experiments
                   WHERE idmd5=?) AS t1,
                  (SELECT 
                       id 
                   FROM
                       metrics
                   WHERE name=?) AS t2
              ''', (mean_psnr, experiment_id, 'psnr'))
    c.execute('''
                 INSERT INTO metric_values (experiment_id, metric_id, value) 
                 SELECT
                  t1.id as experiment_id,
                  t2.id as metric_id,
                  ?
                 FROM
                  (SELECT 
                       id 
                   FROM
                       experiments
                   WHERE idmd5=?) AS t1,
                  (SELECT 
                       id 
                   FROM
                       metrics
                   WHERE name=?) AS t2
              ''', (mean_ssim, experiment_id, 'ssim'))
    c.execute('''
                 INSERT INTO metric_values (experiment_id, metric_id, value) 
                 SELECT
                  t1.id as experiment_id,
                  t2.id as metric_id,
                  ?
                 FROM
                  (SELECT 
                       id 
                   FROM
                       experiments
                   WHERE idmd5=?) AS t1,
                  (SELECT 
                       id 
                   FROM
                       metrics
                   WHERE name=?) AS t2
              ''', (end-start, experiment_id, 'time'))
    infra.conn.commit()
except sqlite3.Error:
    print("Error in writing metrics to db, exiting...")
    exit(-1)


# Avoid closing
while True:
    pass