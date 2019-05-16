'''
face_gatherer.py
Grava amostras do rosto de uma pessoa na tela.
Adaptado de: <https://medium.com/mjrobot-org/real-time-face-recognition-an-end-to-end-project-6a6d6173a6a3>
'''
import cv2
import os
from time import sleep

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Para cada pessoa, escreva um nome
face_id = input('\n Entre com o nome da pessoa separado por underline e tecle <enter> ==>  ')
print("\n [INFO] Inicializando captura de face. Olhe para a câmera e aguarde ...")



# Inicialize o contador de faces em 0
count = 0
while os.path.isdir("src/dataset/" +face_id):
    print("[ERROR] Nome "+face_id+" ja existe no dataset. Tente outro:")
    face_id = input()

os.mkdir("src/dataset/" +face_id)
while 1:
    ret, img = cam.read()
    img = cv2.flip(img, 1) # gire verticalmente a imagem do vídeo
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Salve a imagem capturada
        cv2.imwrite("src/dataset/"+ face_id +"/" + face_id + '_' +  
                    "{:04d}".format(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Pressione 'ESC' para sair do vídeo
    if k == 27:
        break
    elif count >= 10: # Pegue 10 amostras da face e interrompa o vídeo
         break

# Do a bit of cleanup
print("\n [INFO] Saindo do programa")
cam.release()
cv2.destroyAllWindows()
