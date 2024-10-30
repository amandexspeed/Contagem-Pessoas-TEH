import cv2
import os
img_path = os.path.join(os.path.dirname(__file__), '../imagem/1.png')
haardcascades_path = os.path.join(os.path.dirname(__file__), '../haarcascades/')


face_cascade = cv2.CascadeClassifier(haardcascades_path +'haarcascade_frontalface_alt.xml')
#face_cascade = cv2.CascadeClassifier(haardcascades_path +'haarcascade_frontalcatface.xml')
img_face = cv2.imread(img_path)
gray_cascade = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
faces_cascade = face_cascade.detectMultiScale(gray_cascade, 1.1, 4)

count_face = 0
for (x, y, w, h) in faces_cascade:
    cv2.rectangle(img_face, (x, y), (x+w, y+h), (255, 0, 0), 2)
    count_face += 1

print("Número contado:", count_face)
cv2.imshow('img', img_face)
cv2.waitKey(0)