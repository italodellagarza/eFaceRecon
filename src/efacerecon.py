import numpy as np
import cv2 as cv
import image_align

face_detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")


cap = cv.VideoCapture(0)
cap.set(3, 640) # set video width
cap.set(4, 480) # set video height
while 1:
    # Faz uma captura do frame e armazena o retângulo e a imagem
    ret, img = cap.read()
    img = cv.flip(img, 1) # flip video image vertically
    # Converte a imagem para tons de cinza.
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Detecta se há alguma face no frame usando o detector já treinado
    # e retorna as bordas que delimitam essa face.
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # Desenha um retângulo na imagem usando as bordas detectadas
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # Alinha a imagem e a guarda
        aligned = image_align.align_image('''inserir aqui a face''')
        # Classifica o 
    # Mostra o frame capturado em uma janela.    
    cv.imshow('eFaceRecon - Deteccao de faces',img)
    # Espera a tecla ESC para fechar.
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
