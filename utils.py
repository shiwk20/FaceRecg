import os
import numpy as np
import time
import logging
import importlib
import torch


def get_folder_num(path):
    count1 = 0
    count2 = 0
    name_list = os.listdir(path)
    count1 = len(name_list)
    for name in name_list:
        img_list = os.listdir(os.path.join(path, name))
        for img in img_list:
            count2 += 1
    return count1, count2   

# compute the area of bounding box
def area(mtcnn):
    return -np.prod(mtcnn['box'][-2:])

def L2_dist(vec1, vec2):
    return torch.sum(torch.square(vec1 - vec2), dim = 1)

def get_logger(type, log_path = 'log'):
    logger = logging.getLogger()
    logfile = os.path.join(log_path, '{}_{}.log'.format(type, time.strftime('%m-%d-%H-%M-%S')))
    logging.basicConfig(level = logging.INFO, format = \
        '%(asctime)s - %(levelname)s %(filename)s(%(lineno)d): %(message)s', filename = logfile)
    logging.root.addHandler(logging.StreamHandler())
    return logger

def instantiation(config):
    assert 'dest' in config, 'No dest key in config'
    dest, name = config["dest"].rsplit(".", 1)
    module = importlib.import_module(dest)
    return getattr(module, name)(**config.get("paras", dict()))
    
def get_train_indexes(train_img_path = 'data/train/training_set'):
    tmp = {}
    for root, dirs, files in os.walk(train_img_path):
        if root == train_img_path:
            continue
        tmp[root] = files
    tmp = sorted(tmp.items(), key = lambda x: x[0].lower())
    train_indexes = {}
    for train_data in tmp:
        train_indexes[train_data[0]] = train_data[1]
    return train_indexes
