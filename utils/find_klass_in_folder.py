# coding: utf-8
import os

from torch.utils.data import Dataset


def find_klass(where:str, classname:str):
    class_found_flag = False
    if os.path.isdir(where):
        for elem in os.listdir(where):
            if os.path.isfile(os.path.join(where, elem)) and ('.py' in elem) and ('__init__' not in elem):
                try:
                    mod = __import__('{}.{}'.format(os.path.basename(where), elem.split('.')[0]),
                                     fromlist=[classname])
                    dataset_klass = getattr(mod, classname)
                    if issubclass(dataset_klass, Dataset):
                        print("Dataset class found. Nice.")
                        class_found_flag = True
                        return dataset_klass
                except ImportError as e:
                    continue

        if not class_found_flag:
            raise ImportError("Couldn't import dataloader class. Exiting...")
    else:
        raise ValueError("provide folder path! My purpose is to search for class inside folder!")