'''
efacerecon.py
A partir dos modelos prontos o programa reconhece faces em tempo real
'''
import numpy as np
import cv2 as cv
import image_align
from joblib import load
from image_align import align_image
from model import create_model
from person import Person

nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('./weights/nn4.small2.v1.h5')
font = cv.FONT_HERSHEY_SIMPLEX
# Carrega o detector de faces Haar Cascade
face_detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
# Carrega o modelo SVM pronto
svm_model = load('saved_model.joblib')
encoder = load('saved_encoder.joblib')
persons = load('saved_persons.joblib')

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
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # Alinhar a imagem (gray[y:y+h,x:x+w])
        cv.imwrite("image.jpg", gray[y:y+h,x:x+w])
        aligned = cv.imread("image.jpg", 1)
        aligned = aligned[...,::-1]
        aligned = align_image(aligned)
        if aligned is None:
            pass
        else:
            aligned = (aligned / 255.).astype(np.float32)
            # Extrair as features
            embedded = nn4_small2_pretrained.predict(np.expand_dims(aligned, axis=0))[0]
            # Classificar usando svm e extrair o nome da pessoa 
            prediction = svm_model.predict([embedded])
            identity = encoder.inverse_transform(prediction)[0]
            m_weights = [p.embedded for p in persons if p.name==identity]
            distance = np.amin([np.sum(np.square(embedded - w)) for w in m_weights])
            # Se a distância for menor que 0.57 (distance threshold)
            if(distance < 0.57):
                 # Escrever o nome encontrado abaixo do retângulo
                cv.putText(img, str(identity), (x+5,y-5), font, 1, (255,255,255), 2)
            else:
                # Detectou um desconhecido
                cv.putText(img, "desconhecido", (x+5,y-5), font, 1, (0,0,255), 2)
    # Mostra o frame capturado em uma janela.    
    cv.imshow('eFaceRecon - Deteccao de faces',img)
    # Espera a tecla ESC para fechar.
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
