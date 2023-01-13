for i in 0 1 2 3 4 5 6
do
    echo $i
    python face_detect/detect_mtcnn.py $i &
done
python face_detect/detect_mtcnn.py 7