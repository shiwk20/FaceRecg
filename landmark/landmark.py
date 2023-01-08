import dlib
import cv2
import json
import random
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from imutils import face_utils
from logger import setup_logger
from  utils import get_folder_num

train_img_path = 'data/train/training_set'
train_align_path = 'data/train/align'
train_lmk_path = 'data/train/landmark'
train_eval_lmk_path = 'data/train/eval_landmark'
train_error_lmk_path = 'data/train/error_lmk.json'

test_img_path = 'data/test/test_pair'
test_align_path = 'data/test/align'
test_lmk_path = 'data/test/landmark'
test_eval_lmk_path = 'data/test/eval_landmark'
test_error_lmk_path = 'data/test/error_lmk.json'

detector = dlib.get_frontal_face_detector()
lmk_detector = dlib.shape_predictor('checkpoints/shape_predictor_68_face_landmarks.dat')

# 最初的training_set (5353, 12448)；检测出landmark的training_set (5346, 12425)
# test_pair除84/B.jpg外全部检测成功，可对其做特殊处理。最终是(600, 1200)
def get_all_lmk():
    '''
    通过dlib获取所有图片的landmark并保存
    这样可以为之后align做准备
    '''
    def get_lmk(data_path = train_img_path, save_path = train_lmk_path, error_lmk_path = train_error_lmk_path):
        def detect(image, iter):
            for i in range(iter): # the bigger, the slower
                faces = detector(image, i)
                if len(faces) >= 1:
                    break
            return faces
        
        # 存储dlib无法找到人脸的图片索引
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
                    # 最后确认是否无法检测
                    faces = detect(image, 5)
                    if len(faces) == 0:
                        print('{} 0 face'.format(name + '/' + img))
                        error_lmk_name.append(img)
                        continue
                if len(faces) == 1:
                    face = dlib.rectangle(faces[0].left() , faces[0].top() , faces[0].right() , faces[0].bottom())
                elif len(faces) > 1:
                    # 只要有一个人脸包括中心点，则认为其为准确人脸
                    for i in range(len(faces)):
                        if faces[i].left() < size / 2 and faces[i].right() > size / 2\
                        and faces[i].top() < size / 2 and faces[i].bottom() > size / 2:
                            face = dlib.rectangle(faces[i].left() , faces[i].top() , faces[i].right() , faces[i].bottom() )
                
                landmark = lmk_predictor(image, face)
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
    # 针对test_pair/84/B.jpg的特殊处理
    image = cv2.imread('data/test/test_pair/84/B.jpg')
    face = dlib.rectangle(80, 80, 200, 200)
    landmark = lmk_predictor(image, face)
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

# 250x250 -> 128x128
def face_align_all():
    def face_align(data_path = train_img_path, lmk_path = train_lmk_path, save_path = train_align_path, output_size = 128, transform_size = 512):
        img_count = 0
        name_list = os.listdir(lmk_path)
        name_list.sort()
        for name_idx, name in enumerate(tqdm(name_list, desc = 'name')):
            img_list = os.listdir(os.path.join(lmk_path, name)) # .npy
            img_list.sort()
            for img in img_list: 
                landmark = np.load(os.path.join(lmk_path, name, img))
                
                lmk_eye_left, lmk_eye_right, lmk_mouth_outer = landmark[36 : 42], landmark[42 : 48], landmark[48 : 60] 

                # Calculate auxiliary vectors.
                eye_left     = np.mean(lmk_eye_left, axis=0)
                eye_right    = np.mean(lmk_eye_right, axis=0)
                eye_avg      = (eye_left + eye_right) * 0.5
                eye_to_eye   = eye_right - eye_left
                mouth_left   = lmk_mouth_outer[0]
                mouth_right  = lmk_mouth_outer[6]
                mouth_avg    = (mouth_left + mouth_right) * 0.5
                eye_to_mouth = mouth_avg - eye_avg

                # Choose oriented crop rectangle.
                x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
                x /= np.hypot(*x)
                x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
                y = np.flipud(x) * [-1, 1]
                c0 = eye_avg + eye_to_mouth * 0.1

                image = Image.open(os.path.join(data_path, name, img.replace('npy', 'jpg')))
                align_points = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
                qsize = np.hypot(*x) * 2

                # Shrink.
                shrink = int(np.floor(qsize / output_size * 0.5))
                if shrink > 1:
                    rsize = (int(np.rint(float(image.size[0]) / shrink)), int(np.rint(float(image.size[1]) / shrink)))
                    # print(f'first opretion: resize, from {image.size} to {rsize}')
                    image = image.resize(rsize, Image.ANTIALIAS)
                    align_points /= shrink
                    qsize /= shrink

                # Crop.
                border = max(int(np.rint(qsize * 0.1)), 3)
                crop = (int(np.floor(min(align_points[:,0]))), int(np.floor(min(align_points[:,1]))), int(np.ceil(max(align_points[:,0]))), int(np.ceil(max(align_points[:,1]))))
                crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, image.size[0]), min(crop[3] + border, image.size[1]))
                IsCrop = False
                if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
                    IsCrop = True
                    crop = tuple(map(round, crop))
                    # print(f'second operation: crop, {crop}')
                    image = image.crop(crop) # (left, upper, right, lower)
                    # location = [crop[0], crop[1], crop[2], crop[3]]
                    align_points -= crop[0:2]

                # Transform(with rotation)
                align_points = (align_points + 0.5).flatten()
                # 表明它是一个平行四边形
                assert(abs((align_points[2] - align_points[0]) - (align_points[4] - align_points[6])) < 1e-6 and abs((align_points[3] - align_points[1]) - (align_points[5] - align_points[7])) < 1e-6)
                
                ## 使用 matlab 计算 affine matrix:
                # syms transform_size align_points0 align_points1 align_points2 align_points3 align_points4 align_points5 align_points6 align_points7 a b c d e f;
                # % affine
                # [a, b, c] = solve(c == align_points0, b * transform_size + c == align_points2, a * transform_size + c == align_points6, a, b, c);
                # [d, e, f] = solve(f == align_points1, e * transform_size + f == align_points3, d * transform_size + f == align_points7, d, e, f);
            
                # 使用 affine 转换图片
                affine = (-(align_points[0] - align_points[6]) / transform_size, -(align_points[0] - align_points[2]) / transform_size, align_points[0],
                        -(align_points[1] - align_points[7]) / transform_size, -(align_points[1] - align_points[3]) / transform_size, align_points[1])
                image = image.transform((transform_size, transform_size), Image.Transform.AFFINE, affine, Image.Resampling.BICUBIC) # a, b, c, d, e, f
                
                # 使用 quad 转换图片
                # image = image.transform((transform_size, transform_size), Image.Transform.QUAD, align_points, Image.Resampling.BICUBIC) # left-upper, left-bottom, right-bottom, right-upper
                
                if output_size < transform_size:
                    image = image.resize((output_size, output_size), Image.Resampling.BICUBIC)
                
                faces = detector(np.array(image), 1)
                if len(faces) != 1:
                    print('error', name + '/' + img)
                
                os.makedirs(os.path.join(save_path, name), exist_ok=True)
                image.save(os.path.join(save_path, name, img.replace('npy', 'jpg')))
                
                img_count += 1

        print('处理{}张图片，结果保存至{}'.format(img_count, save_path))
    
    face_align()

    
if __name__ == '__main__':
    get_all_lmk()
    
    eval_all_lmk()
    
    face_align_all()
