'''
trainer.py
Treina um modelo de SVM a partir de embedding vectors extraídos das imagens
Adaptado de <http://krasserm.github.io/2018/02/07/deep-face-recognition/>
'''

import numpy as np
import os
import sys
from joblib import dump
from person import Person
from image_align import align_image
import cv2 as cv

from ler_configuracao import ler_configuracao

from sklearn.preprocessing import LabelEncoder



'''Função para carregar uma imagem pelo OpenCV'''
def load_image(path):
    img = cv.imread(path, 1)
    # OpenCV carrega as imagens no canal de cores
    # na ordem BGR. Então, é necessário invertê-las
    return img[...,::-1]

'''Função para carregar as pessoas no dataset'''
def load_persons(path):
    persons = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            persons.append(Person(path, i, f))
    return np.array(persons)


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


# Cria os modelos de rede neural
neural_net, neural_net_str, classificador, limitrofe, dist  = ler_configuracao()


embedded = None

persons = load_persons("dataset/registrado")

# Carrega os pesos para cada rede neural
if neural_net_str == 'nn4':
    embedded = np.zeros((persons.shape[0], 128))
elif neural_net_str == 'facenet':
    embedded = np.zeros((persons.shape[0], 128))
else:
    embedded= np.zeros((persons.shape[0], 2622))

# carrega as pessoas (faces) registradas no dataset


# calcula os embedding vectors utilizando as redes pré-treinadas



for i, m in enumerate(persons):
    img = load_image(m.image_path())
    img1 = cv.resize(img, (96, 96))
    img2 = cv.resize(img, (160, 160))
    img3 = img
    if img1 is None or img2 is None or img3 is None:
        pass
    else:
        # escala os valores RGB no intervalo [0,1]
        img1 = (img1 / 255.).astype(np.float32)
        # escala os valores RGB no itervalo [-1,1]
        img2 = ((img2 / 127.5) - 1.).astype(np.float32)
        img3 = ((img3 / 127.5) - 1.).astype(np.float32)
        # obter os vetores embedding por imagem para cada rede neural (treinar cada uma)
        if neural_net_str == 'nn4':
            embedded[i] = neural_net.predict(np.expand_dims(img1, axis=0))[0]
        elif neural_net_str == 'facenet':
            embedded[i] = l2_normalize(neural_net.predict(np.expand_dims(img2, axis=0))[0,:])
        else:
            embedded[i] = neural_net.predict(np.expand_dims(img3, axis=0))[0,:]
        # grava no vetor persons o seu embeddign vecor
        m.set_embedded(neural_net_str, embedded[i])

targets = np.array([p.name for p in persons])

# Salva as pessoas registradas
dump(persons, 'saved_persons.joblib')

# Quando tiver somente uma pessoa registrada, não se treina os modelos.
if(len(list(set(targets))) == 1):
    sys.exit(0)

# Transforma os nomes das pessoas em números
encoder = LabelEncoder()
encoder.fit(targets)
y = encoder.transform(targets)

# Treina e salva os modelos de classificação
classificador.fit(embedded, y)
dump(classificador, 'models/model.joblib')

# Salva os modelos
dump(encoder, 'saved_encoder.joblib')

