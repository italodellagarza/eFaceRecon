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


def rgb_to_0_1(img):
    return (img/ 255.).astype(np.float32)


def rgb_to_1_1(img):
    return ((img / 127.5) - 1.).astype(np.float32)

def normalization(neural_network, emb_vector):
    if(neural_network == "facenet"):
        return emb_vector / np.sqrt(np.sum(np.multiply(emb_vector, emb_vector)))
    return emb_vector

def main(argv):

    # Carrega a rede neural e o modelo desejados
    neural_network = None
    classifier = None
    rgb_norm = None
    size = 0
    if argv[0] == "nn4":
        neural_network = create_nn4()
        neural_network.load_weights('./weights/nn4.small2.v1.h5')
        size = 96
        rgb_norm = rgb_to_0_1
    elif argv[0] == "facenet":
        neural_network = create_facenet()
        neural_network.load_weights('./weights/facenet.h5')
        size = 160
        rgb_norm = rgb_to_1_1
    elif argv[0] == "vgg":
        neural_network = create_vgg()
        neural_network.load_weights('./weights/vgg_face_weights.h5')
        size = 224
        rgb_norm = rgb_to_1_1
    else:
        print("ERRO!\nDigite da seguinte forma:\n python efacerecon.py <rede> <classificador>\n")
        sys.exit(-1)
    if argv[1] == "svm":
        classifier = load("./models/svm_" + argv[0] + ".joblib")
    elif argv[1] == "knn":
        classifier = load("./models/knn_" + argv[0] + ".joblib")
    else:
        print("ERRO!\nDigite da seguinte forma:\n python efacerecon.py <rede> <classificador>\n")
        sys.exit(-1)

    font = cv.FONT_HERSHEY_SIMPLEX
    # Carrega o detector de faces Haar Cascade
    face_detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    # Carrega o modelo pronto
    encoder = load('saved_encoder.joblib')
    persons = load('saved_persons.joblib')

    cap = cv.VideoCapture(0)
    cap.set(3, 640) # set video width
    cap.set(4, 480) # set video height
    identities = [] # cria uma lista para armazenar as identificações
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
            aligned = align_image(rgb[y:y+h,x:x+w], size)
            if aligned is None:
                pass
            else:
                aligned = rgb_norm(aligned)
                # Extrair as features
                embedded = normalization(argv[0], neural_network.predict(np.expand_dims(aligned, axis=0))[0])
                # Classificar e extrair o nome da pessoa
                prediction = classifier.predict([embedded])
                identity = encoder.inverse_transform(prediction)[0]
                m_weights = [p.load_embedded(argv[0]) for p in persons if p.name==identity]
                distance = np.amin([np.sum(np.square(embedded - w)) for w in m_weights])
                # Se a distância for menor que 0.57 (distance threshold)
                if(distance < 0.57):
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

    com = most_common(identities)
    if(identities.count(com) >= 24):
        print('identificado como ' + com)
    else:
        print('não identificado')

    cap.release()
    cv.destroyAllWindows()


if __name__=="__main__":
    main(sys.argv[1:])
