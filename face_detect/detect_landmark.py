import dlib
import cv2
import json
import random
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from imutils import face_utils
from  utils import get_folder_num

train_img_path = 'data/train/training_set'
train_lmk_path = 'data/train/landmark/landmarks'
train_eval_lmk_path = 'data/train/landmark/eval_landmark'
train_error_lmk_path = 'data/train/landmark/error_lmk.json'

test_img_path = 'data/test/test_pair'
test_lmk_path = 'data/test/landmark/landmarks'
test_eval_lmk_path = 'data/test/landmark/eval_landmark'
test_error_lmk_path = 'data/test/landmark/error_lmk.json'

detector = dlib.get_frontal_face_detector()
lmk_detector = dlib.shape_predictor('checkpoints/shape_predictor_68_face_landmarks.dat')

# original training_set (5353, 12448)；images that are able to detect landmark: (5346, 12425)
# test_pair only 84/B.jpg can't be detected，special treatment to it. final: (600, 1200)
def get_all_lmk():
    def get_lmk(data_path = train_img_path, save_path = train_lmk_path, error_lmk_path = train_error_lmk_path):
        def detect(image, iter):
            for i in range(iter): # the bigger, the slower
                faces = detector(image, i)
                if len(faces) >= 1:
                    break
            return faces
        
        # images that dlib is unable to detect
        error_lmk_all = {}
        img_count = 0
        name_list = os.listdir(data_path)
        name_list.sort()
        for name in tqdm(name_list, desc='name'):
            error_lmk_name = []
            img_list = os.listdir(os.path.join(data_path, name))
            img_list.sort()
            for img in img_list:
                image = cv2.imread(os.path.join(data_path, name, img))
                size = image.shape[0] # 250
                faces = detector(image, 1)
                if len(faces) == 0:
                    faces = detect(image, 5)
                    if len(faces) == 0:
                        print('{} 0 face'.format(name + '/' + img))
                        error_lmk_name.append(img)
                        continue
                if len(faces) == 1:
                    face = dlib.rectangle(faces[0].left() , faces[0].top() , faces[0].right() , faces[0].bottom())
                elif len(faces) > 1:
                    # find face that include center point 
                    for i in range(len(faces)):
                        if faces[i].left() < size / 2 and faces[i].right() > size / 2\
                        and faces[i].top() < size / 2 and faces[i].bottom() > size / 2:
                            face = dlib.rectangle(faces[i].left() , faces[i].top() , faces[i].right() , faces[i].bottom() )
                
                landmark = lmk_detector(image, face)
                landmark = face_utils.shape_to_np(landmark)
                os.makedirs(os.path.join(save_path, name), exist_ok=True)
                np.save(os.path.join(save_path, name, img[:-4] + '.npy'), landmark)
                img_count += 1
            if error_lmk_name != []:
                error_lmk_all[name] = error_lmk_name
        json.dump(error_lmk_all, open(error_lmk_path, 'w'), indent=4)
        print('find {} error images: {}'.format(get_folder_num(data_path)[1] - img_count, error_lmk_all))
    
    get_lmk()
    print(get_folder_num(train_img_path))
    print(get_folder_num(train_lmk_path))
    
    get_lmk(test_img_path, test_lmk_path, test_error_lmk_path)
    # special treatment to test_pair/84/B.jpg
    image = cv2.imread('data/test/test_pair/84/B.jpg')
    face = dlib.rectangle(80, 80, 200, 200)
    landmark = lmk_detector(image, face)
    landmark = face_utils.shape_to_np(landmark)
    for i in range(landmark.shape[0]):
        cv2.circle(image, (landmark[i,0], landmark[i,1]),1,(0,255,0), -1, 8)
    cv2.imwrite('test.png', image)
    np.save('data/test/landmark/84/B.npy', landmark)
    
    print(get_folder_num(test_img_path))
    print(get_folder_num(test_lmk_path))
    os.remove(test_error_lmk_path)
    
def eval_all_lmk():
    def eval_lmk(data_path = train_img_path, lmk_path = train_lmk_path, save_path = train_eval_lmk_path):
        name_list = os.listdir(lmk_path)
        for name in tqdm(name_list):
            img_list = os.listdir(os.path.join(lmk_path, name)) # .npy
            for img in img_list:
                image = cv2.imread(os.path.join(data_path, name, img.replace('npy', 'jpg')))
                landmark = np.load(os.path.join(lmk_path, name, img))
                for i in range(landmark.shape[0]):
                    cv2.circle(image, (landmark[i,0], landmark[i,1]),1,(0,255,0), -1, 8)

                os.makedirs(os.path.join(save_path, name), exist_ok=True)
                cv2.imwrite(os.path.join(save_path, name, img.replace('npy', 'jpg')), image)
    eval_lmk()
    eval_lmk(test_img_path, test_lmk_path, test_eval_lmk_path)

if __name__ == '__main__':
    get_all_lmk()
    
    eval_all_lmk()