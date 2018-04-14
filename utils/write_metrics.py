# coding: utf-8
import sqlite3

from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from metrics.psnr import psnr
from metrics.ssim import ssim


def write_metrics(infra, model, dataset_klass, dataset_root, transform, time_delta, experiment_id, cr_date, cuda=False):
    dataloader_test = DataLoader(dataset_klass(root_dir=dataset_root, transform=transform, train=False),
                                 batch_size=1, num_workers=1)
    psnrs = []
    for i, data in enumerate(dataloader_test):
        lr, hr = data

        if cuda:
            high_res_real = Variable(hr).cuda()
            high_res_fake = model(Variable(lr)).cuda()
        else:
            high_res_real = Variable(hr)
            high_res_fake = model(Variable(lr))

        psnrs.append(psnr(high_res_real.data.numpy()[0], high_res_fake.data.numpy()[0]))

    mean_psnr = np.mean(psnrs)
    ssims = []
    for i, data in enumerate(dataloader_test):
        lr, hr = data

        if cuda:
            high_res_real = Variable(hr).cuda()
            high_res_fake = model(Variable(lr)).cuda()
        else:
            high_res_real = Variable(hr)
            high_res_fake = model(Variable(lr))

        ssims.append(ssim(high_res_real.data.numpy()[0], high_res_fake.data.numpy()[0]))

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
                       WHERE idmd5=? AND dt=?) AS t1,
                      (SELECT 
                           id 
                       FROM
                           metrics
                       WHERE name=?) AS t2
                  ''', (mean_psnr, experiment_id, cr_date, 'psnr'))
        infra.conn.commit()
    except sqlite3.Error:
        print("Error in writing psnr to db, exiting...")
        exit(-1)

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
                       WHERE idmd5=? AND dt=?) AS t1,
                      (SELECT 
                           id 
                       FROM
                           metrics
                       WHERE name=?) AS t2
                  ''', (mean_ssim, experiment_id, cr_date, 'ssim'))
        infra.conn.commit()
    except sqlite3.Error:
        print("Error in writing ssim to db, exiting...")
        exit(-1)

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
                       WHERE idmd5=? AND dt=?) AS t1,
                      (SELECT 
                           id 
                       FROM
                           metrics
                       WHERE name=?) AS t2
                  ''', (time_delta, experiment_id, cr_date, 'time'))
        infra.conn.commit()
    except sqlite3.Error:
        print("Error in writing time to db, exiting...")
        exit(-1)

    print("Mean psnr {}".format(mean_psnr))
    print("Mean ssim {}".format(mean_ssim))
    print("Time elapsed {}".format(time_delta))