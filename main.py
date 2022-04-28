# Facial Recognition program by Hobospider132
# please run program-trainer first when adding new
# data samples

import cv2
import pickle

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
Recogniser = cv2.face.LBPHFaceRecognizer_create()
Recogniser.read("trainner.yml")

labels = {"person name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

# define a video capture object
vid = cv2.VideoCapture(0)
while True:
    # Capture the video frame by frame
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Recogniser
        id_, conf = Recogniser.predict(roi_gray)
        if conf>=45 and conf<=85:
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            colour = (0, 255, 0)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, colour, stroke, cv2.LINE_AA)
        colour = (0,255,0) # NOT RGB, THIS IS BGR.
        stroke = 2
        endCordx = x + w
        endCordy = y + h
        cv2.rectangle(frame, (x,y), (endCordx, endCordy), colour, stroke)
        eyes = eyes_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,0,255), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()


