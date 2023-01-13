from mtcnn import MTCNN
import cv2
import os
import json
from tqdm import tqdm
import sys


if __name__ == '__main__':
    gpu = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    gpu_num = 8
    mode = 'train'

    if mode == 'train':
        data_path = 'data/train/training_set'
        save_path = 'data/train/mtcnn'
    elif mode == 'test':
        data_path = 'data/test/test_pair'
        save_path = 'data/test/mtcnn'
    
    print('data_path: ', data_path)
    detector = MTCNN()
    results_all = {}
    name_list = os.listdir(data_path)
    name_list.sort()
    count = 0
    print('start iter')
    for name_idx, name in enumerate(tqdm(name_list)):
        if name_idx % gpu_num != int(gpu):
            continue
        img_list = os.listdir(os.path.join(data_path, name))
        img_list.sort()
        results_name = {}
        for img in img_list:
            count += 1
            image = cv2.cvtColor(cv2.imread(os.path.join(data_path, name, img)), cv2.COLOR_BGR2RGB)
            result = detector.detect_faces(image)
            results_name[img] = result

        results_all[name] = results_name
    os.makedirs(save_path, exist_ok=True)
    print(f'gpu {gpu} process {count} images')
    json.dump(results_all, open(os.path.join(save_path, f'mtcnn_{gpu}.json'), 'w'), indent=4)