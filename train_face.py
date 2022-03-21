import cv2
import numpy as np
import matplotlib.pyplot as plt

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

data = []

count = 0

while True:

    successful_frame_read , frame = webcam.read()

    greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cordinates = trained_face_data.detectMultiScale(greyscaled_img)

    print(face_cordinates)

    for (x,y,w,h) in face_cordinates:
        cv2.rectangle(frame,(x ,y),(x+w,y+h), (0,255,0),2)
        count = count+1
        name = './images/face without mask/'+str(count) + '.jpg'
        print('creating images...',name)
        cv2.imwrite(name , frame[y:y+h,x:x+w])
        

    cv2.imshow('face detector', frame)
    key = cv2.waitKey(1)

    if key == 32 or count > 500:
        break

np.save('without_mask.npy',data)
np.save('with_mask.npy',data)


webcam.release()