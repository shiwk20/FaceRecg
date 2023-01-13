import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/home/zwl/swk/PRML')
from utils import area

train_img_path = 'data/train/training_set'
train_align_path = 'data/train/align'
train_mtcnn_path = 'data/train/mtcnn/mtcnn.json'
train_eval_mtcnn_path = 'data/train/eval_mtcnn'

test_img_path = 'data/test/test_pair'
test_align_path = 'data/test/align'
test_mtcnn_path = 'data/test/mtcnn/mtcnn.json'
test_eval_mtcnn_path = 'data/test/eval_mtcnn'

# compute the area of bounding box
def area(mtcnn):
    return -np.prod(mtcnn['box'][-2:])

def merge_mtcnn():
    for type in ['train', 'test']:
        mtcnn_path = os.path.join('data', type, 'mtcnn')
        mtcnns = {}
        for i in range(8):
            tmp = json.load(open(os.path.join(mtcnn_path, 'mtcnn_{}.json'.format(i)), 'r'))
            mtcnns.update(tmp)
            
        if type == 'test':
            list_mtcnns = sorted(mtcnns.items(), key=lambda x: int(x[0]))
        elif type == 'train':
            list_mtcnns = sorted(mtcnns.items(), key=lambda x: x[0])
    
        mtcnns = {}
        for i, j in list_mtcnns:
            mtcnns[i] = j
            
        for i, j in mtcnns.items():
            print(i)
        json.dump(mtcnns, open(os.path.join(mtcnn_path, 'mtcnn.json'), 'w'), indent=4)

def eval_mtcnn():
    for data_path, save_path, mtcnn_path in [(train_img_path, train_eval_mtcnn_path, train_mtcnn_path),\
                                            (test_img_path, test_eval_mtcnn_path, test_mtcnn_path)]:
        mtcnns = json.load(open(mtcnn_path, 'r'))
        name_list = os.listdir(data_path)
        name_list.sort()
        for name in tqdm(name_list):
            img_list = os.listdir(os.path.join(data_path, name)) 
            img_list.sort()
            for img in img_list:
                # get the face with the largest area
                keypoints = mtcnns[name][img]
                keypoints = sorted(keypoints, key = area)[0]['keypoints'].values()
                
                keypoints = np.array(list(keypoints))
                image = cv2.imread(os.path.join(data_path, name, img))
                for i in range(len(keypoints)):
                    cv2.circle(image, (keypoints[i][0], keypoints[i][1]),1,(0,255,0), -1, 8)

                os.makedirs(os.path.join(save_path, name), exist_ok=True)
                cv2.imwrite(os.path.join(save_path, name, img), image)

if __name__ == '__main__':
    merge_mtcnn()
    
    eval_mtcnn()
    pass