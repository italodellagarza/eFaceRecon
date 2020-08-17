import configparser, os

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier

from person import Person

from nn4_small2_v1 import create_model as create_nn4
from facenet import InceptionResNetV1 as create_facenet
from vgg_face import create_model as create_vgg


def dist_cos(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
 
def dist_eucl(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def is_float(value):
  try:
    float(value)
    return True
  except:
    return False

def ler_configuracao():
    config = configparser.ConfigParser()
    config.read('config.ini')
    neural_net = config['DEFAULT']['rnc']
    classificador = config['DEFAULT']['classificador']
    limitrofe = config['DEFAULT']['limitrofe']
    dist = config['DEFAULT']['distancia']
    neural_net_str = neural_net
    if neural_net == 'nn4':
        neural_net = create_nn4()
        neural_net.load_weights('weights/nn4.small2.v1.h5')
    elif neural_net == 'vgg':
        neural_net = create_vgg()
        neural_net.load_weights('./weights/vgg_face_weights.h5')
    else:
        if neural_net != 'facenet':
            print("[AVISO] \'" + neural_net + "\' não é reconhecido como um nome válido para" + \
                " rede neural. Inception ResNet V1 foi escolhida por padrão.")
            neural_net_str = 'facenet'
        neural_net = create_facenet()
        neural_net.load_weights('weights/facenet.h5')
    if classificador == 'svm':
        classificador = LinearSVC()
    elif classificador == 'knn':
        classificador = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    elif classificador == 'decision_tree':
        classificador = DecisionTreeClassifier(random_state=0, min_samples_split=5)
    elif classificador == 'random_forest':
        classificador = RandomForestClassifier(random_state=0, min_samples_split=5, n_estimators=60)
    elif classificador == 'adaboost':
        classificador = AdaBoostClassifier(random_state=0, n_estimators=20, base_estimator=DecisionTreeClassifier(max_depth=5))
    elif classificador == 'naive_bayes':
        classificador = GaussianNB()
    else:
        if classificador != 'xgboost':
            print("[AVISO] \'" + classificador + "\' não é reconhecido como um nome válido para" + \
                " o classificador. Gradient Boosting foi escolhido por padrão.")
        classificador = XGBClassifier(random_state=0, objective='multi:softmax', num_class=10, n_estimators=50, max_depth=5)
    if is_float(limitrofe):
        limitrofe = float(limitrofe)
    else:
        print("[AVISO] \'" + limitrofe + "\' não é um valor válido para" + \
            " a limítrofe. 0.83 foi escolhido por padrão.")
    if dist == 'euclideana':
        dist = dist_eucl
    else:
        if dist != 'cosseno':
            print("[AVISO] \'" + dist + "\' não é um valor válido para" + \
            " a medida de distância. A diferença de cossenos foi escolhido por padrão.")
        dist = dist_cos
    return neural_net, neural_net_str, classificador, limitrofe, dist 