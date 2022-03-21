import cv2
from cv2 import putText
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')

with_mask = with_mask.reshape(201 , 50*50*3)
without_mask = without_mask.reshape(201,50*50*3)

x = np.r_[with_mask , without_mask]

labels = np.zeros(x.shape[0])

labels[201:] = 1.0
names = {0 : 'mask' , 1 : 'no mask'}

x_train , x_test , y_train , y_test = train_test_split(x , labels , test_size=0.25)

svm = SVC()
svm.fit(x_train , y_train)

y_pred = svm.predict(x_test)

acc = accuracy_score(y_test , y_pred)

# print(accuracy_score)

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read , frame = webcam.read()

    greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cordinates = trained_face_data.detectMultiScale(greyscaled_img)

    # print(face_cordinates)

    for (x,y,w,h) in face_cordinates:

        cv2.rectangle(frame,(x ,y),(x+w,y+h), (0,255,0),2)
        face = frame[y:y+h , x:x+w :]
        face = cv2.resize(face , (50,50))
        face = face.reshape(1 , -1)
        pred = svm.predict(face)[0]
        n = names[int(pred)]
        print(n)
        if n == names[0]:
            putText(frame , names[0],  (x , y-5) , fontScale = 1 , fontFace=cv2.FONT_HERSHEY_SIMPLEX , color=(0,255,0))
            # putText(frame , acc*100,  (x , y-h+40) , fontScale = 1 , fontFace=cv2.FONT_HERSHEY_SIMPLEX , color=(0,255,0))
        else:
            cv2.rectangle(frame,(x ,y),(x+w,y+h), (0,0,255),2)
            putText(frame , names[1] ,  (x, y-5) , fontScale = 1 , fontFace=cv2.FONT_HERSHEY_SIMPLEX , color=(0,0,255))
            # putText(frame , acc*100,  (x , y+h+40) , fontScale = 1 , fontFace=cv2.FONT_HERSHEY_SIMPLEX , color=(0,255,0))

    cv2.imshow('face detector', frame)
    key = cv2.waitKey(1)

    if key == 32:
        break

webcam.release()