# coding: utf-8

# TODO: hope someday it will do some useful work. It's pretty useless for now, just collects some metrics

import argparse
import json
import os
import re

from torch.utils.data import Dataset

from init import Infrastructure
from utils.find_klass_in_folder import find_klass

infra = Infrastructure()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='name of the new dataset', required=True)
    parser.add_argument('--path', type=str, help='path to folder with train and test folders inside', required=True)
    parser.add_argument('--description', type=str, help="some words about this dataset, maybe there is specific "
                                                        "purpose for it's usage")
    parser.add_argument('--classname', type=str, help="Classname for this dataset (Dataset child)", required=True)
    opt = parser.parse_args()
    print(opt)
    dataset_additional_info = {}
    train_folder_rel = os.path.join(opt.path, 'train/')
    test_folder_rel = os.path.join(opt.path, 'test/')
    abs_train_path = os.path.join(infra.GLOBAL_PATH, train_folder_rel)
    abs_test_path = os.path.join(infra.GLOBAL_PATH, test_folder_rel)
    train_extensions = set()
    test_extensions = set()
    train_size_bytes = 0
    test_size_bytes = 0
    train_count_files = 0
    test_count_files = 0
    extension_regexp = re.compile('\.\w+')
    if os.path.exists(abs_train_path):
        if os.path.isdir(abs_train_path):
            print("train folder found")
            filenames = os.listdir(abs_train_path)
            for elem in filenames:
                try:
                    ext = extension_regexp.search(elem).group(0)
                    train_extensions.add(ext)
                except AttributeError:
                    print("Got file without extension, skipping")

            train_size_bytes = sum(os.path.getsize(os.path.join(abs_train_path, f)) for f in filenames if
                                   os.path.isfile(os.path.join(abs_train_path, f)))

            train_count_files = len(filenames)
    else:
        raise Exception("Train folder must be inside of the dataset folder")

    if os.path.exists(abs_test_path):
        if os.path.isdir(abs_train_path):
            print("test folder found")
            filenames = os.listdir(abs_test_path)
            for elem in filenames:
                try:
                    ext = extension_regexp.search(elem).group(0)
                    test_extensions.add(ext)
                except AttributeError:
                    print("Got file without extension, skipping")

            test_size_bytes = sum(os.path.getsize(os.path.join(abs_test_path, f)) for f in filenames if
                                  os.path.isfile(os.path.join(abs_test_path, f)))

            test_count_files = len(filenames)
    else:
        raise Exception("Test folder must be inside of the dataset folder")

    dataset_additional_info['train'] = {}
    dataset_additional_info['train']['train_extensions'] = list(train_extensions)
    dataset_additional_info['train']['train_size_bytes'] = train_size_bytes
    dataset_additional_info['test'] = {}
    dataset_additional_info['test']['test_extensions'] = list(test_extensions)
    dataset_additional_info['test']['test_size_bytes'] = test_size_bytes

    dataset_klass = find_klass(infra.datasets_path, opt.classname)

    c = infra.conn.cursor()
    c.execute('''
                INSERT INTO datasets (name, path, description, additional, classname, train_count_files, test_count_files) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
             ''', (
    opt.name, opt.path, opt.description, json.dumps(dataset_additional_info), opt.classname, train_count_files, test_count_files))
    infra.conn.commit()
