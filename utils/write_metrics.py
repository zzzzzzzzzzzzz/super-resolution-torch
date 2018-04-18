# coding: utf-8
import sqlite3
from utils.count_metrics import count_metrics


def write_metrics(infra, model, dataset_klass, dataset_root, transform, time_delta, experiment_id, cr_date, cuda=False):

    mean_psnr, mean_ssim = count_metrics(dataset_klass, dataset_root, transform, model, cuda)

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