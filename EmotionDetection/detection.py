import cv2
import numpy as np
from keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import webbrowser
model=load_model('model_file_30epochs.h5')

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
Song_dict={0:'Rock',1:'Alternative', 2:'Metal', 3:'Pop',4:'Chill',5:'Mood',6:'Party'}
cam = cv2.VideoCapture(0)
frame=None
while True:
    ret,frame=cam.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= faceDetect.detectMultiScale(gray, 1.3, 3)
    for x,y,w,h in faces:
        sub_face_img=gray[y:y+h, x:x+w]
        resized=cv2.resize(sub_face_img,(48,48))
        normalize=resized/255.0
        reshaped=np.reshape(normalize, (1, 48, 48, 1))
        result=model.predict(reshaped)
        label=np.argmax(result, axis=1)[0]
        print(label)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('c'):
        break
cam.release()
cv2.destroyAllWindows()
print(labels_dict[label])
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
results = spotify.category_playlists(Song_dict[label])
url=results["playlists"]["items"][0]["external_urls"]["spotify"]
webbrowser.open(url, new=2)