'''
trainer.py
Treina um modelo de SVM a partir de embedding vectors extraídos das imagens
Adaptado de <http://krasserm.github.io/2018/02/07/deep-face-recognition/>
'''
from nn4_small2_v1 import create_model as create_nn4
from facenet import InceptionResNetV1 as create_facenet
from vgg_face import create_model as create_vgg

import numpy as np
import os
from joblib import dump
from person import Person
from image_align import align_image
import cv2 as cv
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
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

'''Função para treinar o conjunto de modelos e salvar'''
def train_models_and_save(vectors, labels, network):
    models = {}
    models["svm"] = LinearSVC()
    models["knn"] = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    for name, model in models.items():
        model.fit(vectors, labels)
        dump(model, 'models/'+ name + '_' + network +'.joblib')

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


# Cria os modelos de rede neural
nn4_small2_pretrained = create_nn4()
facenet = create_facenet()
vgg_face = create_vgg()

# Carrega os pesos para cada rede neural
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
facenet.load_weights('weights/facenet.h5')
vgg_face.load_weights('weights/vgg_face_weights.h5')

# carrega as pessoas (faces) registradas no dataset
persons = load_persons("dataset")

# calcula os embedding vectors utilizando as redes pré-treinadas
embedded_nn4 = np.zeros((persons.shape[0], 128))
embedded_vgg = np.zeros((persons.shape[0], 2622))
embedded_facenet = np.zeros((persons.shape[0], 128))

for i, m in enumerate(persons):
    img = load_image(m.image_path())
    img1 = align_image(img, 96)
    img2 = align_image(img, 160)
    img3 = align_image(img, 224)
    if img1 is None or img2 is None or img3 is None:
        pass
    else:
        # escala os valores RGB no intervalo [0,1]
        img1 = (img1 / 255.).astype(np.float32)
        # escala os valores RGB no itervalo [-1,1]
        img2 = ((img2 / 127.5) - 1.).astype(np.float32)
        img3 = ((img3 / 127.5) - 1.).astype(np.float32)
        # obter os vetores embedding por imagem para cada rede neural (treinar cada uma)
        embedded_nn4[i] = nn4_small2_pretrained.predict(np.expand_dims(img1, axis=0))[0]
        embedded_facenet[i] = l2_normalize(facenet.predict(np.expand_dims(img2, axis=0))[0,:])
        embedded_vgg[i] = vgg_face.predict(np.expand_dims(img3, axis=0))[0,:]
        # grava no vetor persons o seu embeddign vecor
        m.embedded_nn4 = embedded_nn4[i]
        m.embedded_facenet = embedded_facenet[i]
        m.embedded_vgg = embedded_vgg[i]

persons = [p for p in persons if (p.embedded_nn4.any() and p.embedded_facenet.any() and p.embedded_vgg.any())]
embedded_nn4 = [p.embedded_nn4 for p in persons]
embedded_vgg = [p.embedded_vgg for p in persons]
embedded_facenet = [p.embedded_facenet for p in persons]

targets = np.array([p.name for p in persons])

# Transforma os nomes das pessoas em números
encoder = LabelEncoder()
encoder.fit(targets)
y = encoder.transform(targets)

# Treina e salva os modelos de classificação
train_models_and_save(embedded_nn4, y, 'nn4')
train_models_and_save(embedded_facenet, y, 'facenet')
train_models_and_save(embedded_vgg, y, 'vgg')

# Salva os modelos
dump(encoder, 'saved_encoder.joblib')
dump(persons, 'saved_persons.joblib')
