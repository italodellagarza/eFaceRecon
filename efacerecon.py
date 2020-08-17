'''
efacerecon.py
A partir dos modelos prontos o programa reconhece faces em tempo real
'''

import sys
import numpy as np
import cv2 as cv
from joblib import load
from image_align import align_image
from nn4_small2_v1 import create_model as create_nn4
from facenet import InceptionResNetV1 as create_facenet
from vgg_face import create_model as create_vgg
import time

import dlib

from ler_configuracao import ler_configuracao


neural_net, neural_net_str, classificador, limitrofe, dist = ler_configuracao()

# Função para encontrar o elemento mais comum de uma lista
def most_common(list):
    counter = 0
    num = list[0]   
    for i in list: 
        curr_frequency = list.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
    return num 

def get_size():
    if neural_net_str == "nn4":
        return 96
    elif neural_net_str == "facenet":
        return 160
    else:
        return 224


def rgb_norm(img):
    if neural_net_str == "nn4":
        return (img/ 255.).astype(np.float32)
    else:
        return ((img / 127.5) - 1.).astype(np.float32)

def normalization(emb_vector):
    global neural_net_str
    if(neural_net_str == "facenet"):
        return emb_vector[0,:] / np.sqrt(np.sum(np.multiply(emb_vector[0,:], emb_vector[0,:])))
    elif neural_net_str == 'nn4':
        return emb_vector[0]
    else:
        return emb_vector[0,:]

def main(argv):
    global neural_net
    global neural_net_str
    global classificador
    global limitrofe
    size = get_size()
    

    font = cv.FONT_HERSHEY_SIMPLEX
    # Carrega o detector de faces Haar Cascade
    face_detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    # Carrega o modelo pronto
    encoder = None
    persons = load('saved_persons.joblib')

    # Verifica se há somente uma pessoa registrada
    one_person = False
    if len(list(set([p.name for p in persons]))) == 1:
        one_person = True
    else:
        classificador = load('models/model.joblib')
        encoder = load('saved_encoder.joblib')

    cap = cv.VideoCapture(0)
    cap.set(3, 640) # set video width
    cap.set(4, 480) # set video height
    identities = [] # cria uma lista para armazenar as identificações
    time_begin = time.time()
    while len(identities) < 30:
        # Faz uma captura do frame e armazena o retângulo e a imagem
        ret, img = cap.read()
        img = cv.flip(img, 1) # flip video image vertically
        # Converte a imagem para tons de cinza.
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Detecta se há alguma face no frame usando o detector já treinado
        # e retorna as bordas que delimitam essa face.
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            # Desenha um retângulo em volta da imagem
            cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            # Alinha a imagem em formato RGB
            aligned = align_image(rgb, dlib.rectangle(x,y,x+w,y+h), size)
            if aligned is None:
                pass
            else:
                aligned = rgb_norm(aligned)
                # Extrair as features
                embedded = normalization(neural_net.predict(np.expand_dims(aligned, axis=0)))
                
                if not one_person:
                    # Classificar e extrair o nome da pessoa
                    prediction = classificador.predict([embedded])
                    identity = encoder.inverse_transform(prediction)[0]
                else:
                    identity = persons[0].name
                
                
                m_weights = [p.load_embedded(neural_net_str) for p in persons if p.name==identity]
                distance = np.amin([dist(embedded, w) for w in m_weights])
                # Se a distância for menor que 0.57 (distance threshold)
                if(distance < limitrofe):
                     # Escrever o nome encontrado abaixo do retângulo
                    cv.putText(img, str(identity), (x+5,y-5), font, 1, (0,255,0), 2)
                    identities.append(str(identity))
                else:
                    # Detectou um desconhecido
                    cv.putText(img, "desconhecido", (x+5,y-5), font, 1, (0,0,255), 2)
                    identities.append("desconhecido")
        # Mostra o frame capturado em uma janela.
        cv.imshow('eFaceRecon - Deteccao de faces',img)
        # Espera a tecla ESC para fechar.
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    time_end = time.time()
    print("Tempo de reconhecimento = " + str(time_end - time_begin) + "s")
    com = most_common(identities)
    if(identities.count(com) >= 24):
        print('identificado como ' + com)
    else:
        print('não identificado')

    cap.release()
    cv.destroyAllWindows()


if __name__=="__main__":
    main(sys.argv[1:])
