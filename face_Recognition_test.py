import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image 
model=load_model(r'C:\Python37\Projects\Transfer-Learning-master/facefeatures_new_model.h5')

face_cascade=cv2.CascadeClassifier(r'C:\Python37\Projects\models\research\object_detection\data\haarcascade_frontalface_alt.xml')

def face_extractor(img):
    # function detects faces and return the cropped face
    # If no face detected, it returns the input image
    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(img,1.3,5)

    if faces is ():
        return None

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face=img[y:y+h,x:x+w]

    return cropped_face


video_capture=cv2.VideoCapture(0)
while True:
    _,frame=video_capture.read()

    face=face_extractor(frame)
    if type(face) is np.ndarray:
        face=cv2.resize(face,(224,224))
        im= Image.fromarray(face,'RGB')
        img_array=np.array(im)
        img_array=np.expand_dims(img_array,axis=0)
        pred=model.predict(img_array)
        print(pred)

        name='None matching'
        
        if(pred[0][0]>0.3):
            name='Ashish'
        cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    
    else:
        cv2.putText(frame,"No Face Found",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
    
