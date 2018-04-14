# coding: utf-8

import argparse
import os
import sqlite3
import sys
import time
sys.path.append('.')
import torch
import torch.optim as optim
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from init import Infrastructure
from models.srcnn import SRcnn
from transforms.myrandomsample import MyRandomSample
from transforms.myresize import MyResize
from transforms.totensor import MyToTensor
from transforms.toycbcr import ToYCbCr
from utils.find_klass_in_folder import find_klass
from utils.write_metrics import write_metrics
from visualizers.srgan_vis import Visualizer

if __name__ == '__main__':
    infra = Infrastructure()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mixed-flowers-berkley',
                        help='One of the datasets listed in your database')
    parser.add_argument('--model', type=str, help="One of the models listed in your database")
    parser.add_argument('--depthFactor', type=int, default=1, help="Multiply number of filter by this number")
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=16, help='the low resolution image size')
    parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
    parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for model')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--weights', type=str, default='', help="path to model weights (to continue training)")
    parser.add_argument('--description', type=str, help="The description of experiment. It's purpose")

    opt = parser.parse_args()
    print(opt)
    experiment_id, cr_date = infra.init_experiment(opt)
    start = time.time()
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

    try:
        os.makedirs('{}/{}/{}/{}/'.format(infra.snapshots_path, os.path.basename(__file__), experiment_id, cr_date.strftime(infra.DATETIME_FORMAT_STR)))
    except OSError:
        pass

    print("Starting experiment {}".format(experiment_id))

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    transform = transforms.Compose([ToYCbCr(),
                                    MyRandomSample(opt.imageSize * opt.upSampling),  # crop size
                                    MyResize(opt.imageSize),  # crop size / scale factor
                                    MyResize(opt.imageSize * opt.upSampling),
                                    MyToTensor()])

    dataloader = DataLoader(dataset_klass(root_dir=dataset_root, transform=transform),
                            batch_size=opt.batchSize,
                            shuffle=True, num_workers=int(opt.workers))

    criterion = torch.nn.MSELoss()

    model = SRcnn()
    if opt.weights != '':
        model.load_state_dict(torch.load(opt.weights))

    # if gpu is to be used
    if opt.cuda:
        model.cuda()
        criterion.cuda()

    optim_srcnn = optim.Adam(model.parameters(), lr=opt.lr)

    configure('{}/{}/{}/{}/'.format(infra.logs_path, os.path.basename(__file__), experiment_id, cr_date.strftime(infra.DATETIME_FORMAT_STR)), flush_secs=5)
    visualizer = Visualizer(image_size=opt.imageSize * opt.upSampling)
    try:
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
                epoch, opt.nEpochs, len(dataloader), len(dataloader), mean_loss / len(dataloader)))
            log_value('model_mse_loss_epoch', mean_loss / len(dataloader), epoch)
            # Do checkpointing every epoch
            torch.save(model.state_dict(),
                       '{}/{}/{}/{}/model_pretrain.pth'.format(infra.snapshots_path, os.path.basename(__file__), experiment_id, cr_date.strftime(infra.DATETIME_FORMAT_STR)))
    except KeyboardInterrupt:
        print("Keyboard interrupt. Writing metrics...")
        write_metrics(infra, model, dataset_klass, dataset_root, transform, time.time() - start, experiment_id, cr_date, opt.cuda)
        print("Exiting...")
        exit(0)

    end = time.time()

    write_metrics(infra, model, dataset_klass, dataset_root, transform, time.time() - start, experiment_id, cr_date, opt.cuda)

    # Avoid closing
    text = ''
    while text != 'y':
        text = str(
            input("The experiment is ended. Would you like to stop it? [y]:")).lower()
    if text == 'y':
        print("Stopping...")
        exit(0)
