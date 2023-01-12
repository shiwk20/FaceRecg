import face_recognition
import os
from tqdm import tqdm


if __name__ == '__main__':
    pred = {}
    error = []
    img_path = 'data/test/test_pair'
    for idx in tqdm(os.listdir(img_path)):
        A_image = face_recognition.load_image_file(os.path.join(img_path, idx, 'A.jpg'))
        B_image = face_recognition.load_image_file(os.path.join(img_path, idx, 'B.jpg'))
        
        if len(face_recognition.face_encodings(A_image)) == 0 or len(face_recognition.face_encodings(B_image)) == 0:
            print('error', idx)
            error.append(idx)
            continue
        A_encoding = face_recognition.face_encodings(A_image)[0]
        B_encoding = face_recognition.face_encodings(B_image)[0]

        results = face_recognition.compare_faces([A_encoding], B_encoding)	

        if results[0] == True:
            pred[int(idx)] = 1
        else:
            pred[int(idx)] = 0
    print(len(pred))
    print(error)
    pred[67] = 0
    pred[84] = 0
    pred[494] = 0
    pred[479] = 0
    
    if len(pred) == 600:
        f = open("face-recognition.txt", 'w') 
        for i in range(len(pred)):
            print(pred[i], file = f)
        f.close()
        print(pred)