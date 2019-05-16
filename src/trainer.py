# TODO criar um novo model a partir de um pré treinado

from model import create_model
import numpy as np
import os
from person import Person
from image_align import align_image
import cv2 as cv
from matplotlib import pyplot as plt

def load_image(path):
    img = cv.imread(path, 1)
    # OpenCV carrega as imagens no canal de cores
    # na ordem BGR. Então, é necessário invertê-las
    return img[...,::-1]


def load_persons(path):
    persons = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            persons.append(Person(path, i, f))
    return np.array(persons)


nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
persons = load_persons("dataset")
embedded = np.zeros((persons.shape[0], 128))

for i, m in enumerate(persons):
    img = load_image(m.image_path())
    img = align_image(img)
    # escala os valores RGB no intervalo [0,1]
    img = (img / 255.).astype(np.float32)
    # obter os vetores embedding por imagem
    embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    #plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(persons[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(persons[idx2].image_path()));    

show_pair(2, 3)
show_pair(2, 12)

# TODO salvar esse vetor de features extraído em um arquivo