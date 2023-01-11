# 250x250 -> 112x96(HxW)
import numpy as np
import cv2
import json
import random
import os
from PIL import Image
from tqdm import tqdm
import argparse
from utils import area

train_img_path = 'data/train/training_set'
train_align_mtcnn_path = 'data/train/mtcnn/align112x96'
train_mtcnn_path = 'data/train/mtcnn/mtcnns/mtcnn.json'
train_align_lmk_path = 'data/train/landmark/align112x112'
train_lmk_path = 'data/train/landmark/landmarks'

test_img_path = 'data/test/test_pair'
test_align_mtcnn_path = 'data/test/mtcnn/align112x96'
test_mtcnn_path = 'data/test/mtcnn/mtcnns/mtcnn.json'
test_align_lmk_path = 'data/test/landmark/align112x112'
test_lmk_path = 'data/test/landmark/landmarks'

mtcnn_crop_size = (96, 112) # (W, H)
lmk_crop_size = 112 # (W, H)

template = np.float32([ [0.        , 0.17856914], [0.00412831, 0.31259227],
                        [0.0196793 , 0.44770938], [0.04809872, 0.5800727 ],
                        [0.10028344, 0.70349526], [0.17999782, 0.81208664],
                        [0.27627307, 0.90467805], [0.38463727, 0.98006284],
                        [0.5073561 , 1.        ], [0.63014114, 0.9761118 ],
                        [0.7386777 , 0.89921385], [0.8354747 , 0.80513287],
                        [0.91434467, 0.6945623 ], [0.9643504 , 0.56826204],
                        [0.9887058 , 0.432444  ], [0.9993123 , 0.29529294],
                        [1.        , 0.15909716], [0.09485531, 0.07603313],
                        [0.15534875, 0.02492465], [0.2377474 , 0.01139098],
                        [0.32313403, 0.02415778], [0.4036699 , 0.05780071],
                        [0.56864655, 0.0521157 ], [0.65128165, 0.01543965],
                        [0.7379608 , 0.        ], [0.82290924, 0.01191543],
                        [0.88739765, 0.06025707], [0.48893312, 0.15513189],
                        [0.48991537, 0.24343018], [0.49092147, 0.33176517],
                        [0.49209353, 0.422107  ], [0.397399  , 0.48004663],
                        [0.4442625 , 0.49906778], [0.4949509 , 0.5144414 ],
                        [0.54558265, 0.49682876], [0.59175086, 0.47722608],
                        [0.194157  , 0.16926692], [0.24600308, 0.13693026],
                        [0.31000495, 0.13735634], [0.36378494, 0.17794687],
                        [0.3063696 , 0.19082251], [0.24390514, 0.19138186],
                        [0.6189632 , 0.17277813], [0.67249435, 0.12988105],
                        [0.7362857 , 0.1279085 ], [0.7888591 , 0.15817115],
                        [0.74115133, 0.18155812], [0.6791372 , 0.18370388],
                        [0.30711025, 0.6418497 ], [0.3759703 , 0.6109595 ],
                        [0.44670257, 0.5970508 ], [0.49721557, 0.60872644],
                        [0.5500201 , 0.5954327 ], [0.6233016 , 0.6070911 ],
                        [0.69541407, 0.6341429 ], [0.628068  , 0.70906836],
                        [0.5573954 , 0.7434471 ], [0.50020397, 0.7505844 ],
                        [0.44528747, 0.74580276], [0.37508208, 0.7145425 ],
                        [0.3372878 , 0.64616466], [0.44701463, 0.64064664],
                        [0.49795204, 0.6449633 ], [0.5513943 , 0.6385937 ],
                        [0.6650228 , 0.63955915], [0.5530556 , 0.67647934],
                        [0.4986481 , 0.68417645], [0.44657204, 0.6786047 ]])

# train: [5353, 12448]
# test: [600, 1200]
def face_align_mtcnn():
    print(f'start mtcnn align, align size: {mtcnn_crop_size}')
    for data_path, align_path, mtcnn_path in [(train_img_path, train_align_mtcnn_path, train_mtcnn_path),\
                                            (test_img_path, test_align_mtcnn_path, test_mtcnn_path)]:
        name_list = os.listdir(data_path)
        name_list.sort()
        mtcnns = json.load(open(mtcnn_path, 'r'))
        count = [0, 0]

        for name_idx, name in enumerate(tqdm(name_list, desc = 'name')):
            img_list = os.listdir(os.path.join(data_path, name))
            img_list.sort()
            count[0] += 1
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
                x0 = max(0, rotate_eye_center[0] - 1 / 2 * mtcnn_crop_size[0] / mtcnn_crop_size[1] * (y1 - y0))
                x1 = x0 + mtcnn_crop_size[0] / mtcnn_crop_size[1] * (y1 - y0)
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                
                aligned_img = rotated_img[y0:y1, x0:x1]
                aligned_img = cv2.resize(aligned_img, mtcnn_crop_size)

                os.makedirs(os.path.join(align_path, name), exist_ok=True)
                cv2.imwrite(os.path.join(align_path, name, img), aligned_img)
                count[1] += 1
        print(f'process ({count}) images, save the result to f{align_path}')

# train: [5346, 12425]
# test: [600, 1200]
def face_align_lmk():
    print(f'start landmark align, align size: {lmk_crop_size}, {lmk_crop_size}')
    for data_path, align_path, lmk_path in [(train_img_path, train_align_lmk_path, train_lmk_path),\
                                            (test_img_path, test_align_lmk_path, test_lmk_path)]:
        name_list = os.listdir(lmk_path)
        name_list.sort()
        count = [0, 0]
        
        for name_idx, name in enumerate(tqdm(name_list, desc = 'name')):
            img_list = os.listdir(os.path.join(lmk_path, name))
            img_list.sort()
            count[0] += 1
            for img in img_list: # *.npy
                image = cv2.imread(os.path.join(data_path, name, img.replace('npy', 'jpg')))
                landmark = np.float32(np.load(os.path.join(lmk_path, name, img)))
                landmark_indexes = np.array([39, 42, 57]) # inner eyes and bottom lip
                
                affine_matrix = cv2.getAffineTransform(landmark[landmark_indexes], lmk_crop_size * template[landmark_indexes])
                aligned_img = cv2.warpAffine(image, affine_matrix, (lmk_crop_size, lmk_crop_size))

                os.makedirs(os.path.join(align_path, name), exist_ok=True)
                cv2.imwrite(os.path.join(align_path, name, img.replace('npy', 'jpg')), aligned_img)
                count[1] += 1
        print(f'process ({count}) images, save the result to {align_path}')
                
    
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-t', '--type', help = 'align type(mtcnn or landmark), default: mtcnn', type = str, default = 'mtcnn')
    args = parse.parse_args()
    if args.type == 'mtcnn':
        face_align_mtcnn()
    elif args.type == 'landmark':
        face_align_lmk()
    else:
        raise NameError('Expected type to be mtcnn or landmark')
