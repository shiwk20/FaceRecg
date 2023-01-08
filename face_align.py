# 250x250 -> 112x96(HxW)
import numpy as np
import cv2
import json
import random
import os
from PIL import Image
from tqdm import tqdm
from utils import area

train_img_path = 'data/train/training_set'
train_align_path = 'data/train/align'
train_mtcnn_path = 'data/train/mtcnn/mtcnn.json'

test_img_path = 'data/test/test_pair'
test_align_path = 'data/test/align'
test_mtcnn_path = 'data/test/mtcnn/mtcnn.json'

crop_size = (96, 112) # (W, H)

def face_align():

    for data_path, align_path, mtcnn_path in [(train_img_path, train_align_path, train_mtcnn_path),\
                                            (test_img_path, test_align_path, test_mtcnn_path)]:
        name_list = os.listdir(data_path)
        name_list.sort()
        mtcnns = json.load(open(mtcnn_path, 'r'))

        for name_idx, name in enumerate(tqdm(name_list, desc = 'name')):
            img_list = os.listdir(os.path.join(data_path, name))
            img_list.sort()
            for img in img_list: 
                image = cv2.imread(os.path.join(data_path, name, img))

                keypoints = mtcnns[name][img]
                keypoints = sorted(keypoints, key = area)[0]['keypoints']
                
                left_eye = np.array(keypoints['left_eye'])
                right_eye = np.array(keypoints['right_eye'])
                
                # theta of rotation (anticlockwise)
                theta = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
                # center of rotation
                eye_center = (left_eye + right_eye) // 2
                rotate_matrix = cv2.getRotationMatrix2D(tuple(eye_center), theta * 180. / np.pi, 1)
                rotated_img = cv2.warpAffine(image, rotate_matrix, (250, 250))
                
                # find the new Coords of keypoints after rotation (https://blog.csdn.net/sinat_33425327/article/details/78333946)
                rotate_keypoints = {}
                for i in keypoints.keys():
                    rotate_keypoints[i] = np.dot(rotate_matrix, np.append(keypoints[i], 1))[:2]
                
                # find crop rectangle[x0, y0, x1, y1]
                H, W = image.shape[0], image.shape[1]
                rotate_eye_center = (rotate_keypoints['left_eye'] + rotate_keypoints['right_eye']) / 2
                rotate_mouth_center = (rotate_keypoints['mouth_left'] + rotate_keypoints['mouth_right']) / 2
                
                # there are two hyper-parameter to adjust
                y0 = max(0, rotate_eye_center[1] - 15 / 16 * (rotate_mouth_center[1] - rotate_eye_center[1]))
                y1 = min(H, rotate_mouth_center[1] - 10 / 16 * (rotate_eye_center[1] - rotate_mouth_center[1]))
                x0 = max(0, rotate_eye_center[0] - 1 / 2 * crop_size[0] / crop_size[1] * (y1 - y0))
                x1 = x0 + crop_size[0] / crop_size[1] * (y1 - y0)
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                
                aligned_img = rotated_img[y0:y1, x0:x1]
                aligned_img = cv2.resize(aligned_img, crop_size)

                os.makedirs(os.path.join(align_path, name), exist_ok=True)
                cv2.imwrite(os.path.join(align_path, name, img), aligned_img)
            
if __name__ == '__main__':
    face_align()
