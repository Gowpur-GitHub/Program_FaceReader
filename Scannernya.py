import cv2
import os
from PIL import Image

camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('DataGambar/training.xml')

a = 0
while True:
    a = a + 1
    check, frame = video.read()

    if not check:
        break

    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)

    for (x, y, w, h) in wajah:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = recognizer.predict(abu[y:y + h, x:x + w])
        if id == 1:
            id = 'GOWPUR'
        elif id == 2:
            id = 'GigaChad'
        elif id == 3:
            id = 'Elon Musk'
        elif id == 4:
            id = 'EMAK'
        elif id == 5:
            id = 'SHAMIN'
        elif id == 6:
            id = 'MANIH'
        elif id == 7:
            id = 'KEMBI'
        elif id == 8:
            id = 'PEREMPUAN'
        cv2.putText(frame, str(id), (x + 40, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1)

    if key == ord('a'):
        break

video.release()
cv2.destroyAllWindows()
