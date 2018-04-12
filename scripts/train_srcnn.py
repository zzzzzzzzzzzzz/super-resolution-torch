# coding: utf-8

import argparse
import sqlite3

import os
import time
import sys

import numpy as np
import torch.optim as optim

import torch
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from init import Infrastructure
from metrics.psnr import psnr
from metrics.ssim import ssim
from models.srcnn import SRcnn
from transforms.myrandomsample import MyRandomSample
from transforms.myresize import MyResize
from transforms.totensor import MyToTensor
from transforms.toycbcr import ToYCbCr
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
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for model')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--weights', type=str, default='', help="path to model weights (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')


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
                                MyRandomSample(opt.imageSize*opt.upSampling), # crop size
                                MyResize(opt.imageSize), # crop size / scale factor
                                MyResize(opt.imageSize*opt.upSampling),
                                MyToTensor()])

dataloader = DataLoader(dataset_klass(root_dir=dataset_root, transform=transform),
                        batch_size=opt.batchSize,
                        shuffle=True, num_workers=int(opt.workers))

criterion = torch.nn.MSELoss()

model = SRcnn()

# if gpu is to be used
if opt.cuda:
    model.cuda()
    criterion.cuda()

optim_srcnn = optim.Adam(model.parameters(), lr=opt.lr)

configure('{}/{}/{}/'.format(infra.logs_path, os.path.basename(__file__), experiment_id), flush_secs=5)
visualizer = Visualizer(image_size=opt.imageSize*opt.upSampling)

for epoch in range(opt.nEpochs):  # loop over the dataset multiple times

    mean_loss = 0.0
    for i, data in enumerate(dataloader):
        # get the inputs
        lr, hr = data

        if opt.cuda:
            high_res_real = Variable(hr.cuda())
            high_res_fake = model(Variable(lr).cuda())
        else:
            high_res_real = Variable(hr)
            high_res_fake = model(Variable(lr))

        # zero the parameter gradients
        optim_srcnn.zero_grad()

        # forward + backward + optimize
        loss = criterion(high_res_fake, high_res_real)
        mean_loss += loss.data[0]
        loss.backward()
        optim_srcnn.step()

        # Logging and visualization
        sys.stdout.write('\r[%d/%d][%d/%d] MSE_Loss: %.4f' % (
        epoch, opt.nEpochs, i, len(dataloader), loss.data[0]))
        visualizer.show(lr, high_res_real.cpu().data, high_res_fake.cpu().data)

    sys.stdout.write('\r[%d/%d][%d/%d] Epoch mean MSE_Loss: %.4f\n' % (
    epoch, opt.pretrainGeneratorEpochs, i, len(dataloader), mean_loss / len(dataloader)))
    log_value('model_mse_loss_epoch', mean_loss / len(dataloader), epoch)
    # Do checkpointing every epoch
    torch.save(model.state_dict(), '{}/{}/generator_pretrain.pth'.format(infra.snapshots_path, experiment_id))

end = time.time()

dataloader_test = DataLoader(dataset_klass(root_dir=dataset_root, transform=transform, train=False),
                        batch_size=1, num_workers=1)

psnrs = []
for i, data in enumerate(dataloader_test):
    lr, hr = data

    if opt.cuda:
        high_res_real = Variable(hr.cuda())
        high_res_fake = model(Variable(lr).cuda())
    else:
        high_res_real = Variable(hr)
        high_res_fake = model(Variable(lr))

    psnrs.append(psnr(high_res_real.data.numpy(), high_res_fake.data.numpy()))

mean_psnr = np.mean(psnrs)
ssims = []
for i, data in enumerate(dataloader_test):
    lr, hr = data

    if opt.cuda:
        high_res_real = Variable(hr.cuda())
        high_res_fake = model(Variable(lr).cuda())
    else:
        high_res_real = Variable(hr)
        high_res_fake = model(Variable(lr))

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