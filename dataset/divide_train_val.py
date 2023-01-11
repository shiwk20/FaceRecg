import random
import argparse
import json
from utils import get_train_indexes

def divide_train_val(seed):
    all_indexes = get_train_indexes()
    
    random.seed(seed)
    train_indexes = random.sample(list(all_indexes.keys()), int(len(all_indexes) * 0.8))
    val_indexes = list(set(all_indexes.keys()) - set(train_indexes))
    
    train_indexes_tmp = {}
    for key in train_indexes:
        train_indexes_tmp[key] = all_indexes[key]
        
    val_indexes_tmp = {}
    for key in val_indexes:
        val_indexes_tmp[key] = all_indexes[key]
    
    json.dump(train_indexes_tmp, open('data/train/train_indexes.json', 'w'), indent=4)
    json.dump(val_indexes_tmp, open('data/train/val_indexes.json', 'w'),  indent=4)
    
    return train_indexes_tmp, val_indexes_tmp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='seed', type=int, default=0)
    args = parser.parse_args()
    
    divide_train_val(args.seed)
