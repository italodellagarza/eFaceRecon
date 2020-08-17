import os
import numpy as np

class Person:
    
    
    def __init__(self, base, name, file, vp_exists=""):
        self.name = name
        self.base = base
        self.file = file
        self.embedded_nn4 = np.zeros(128)
        self.embedded_facenet = np.zeros(128)
        self.embedded_vgg = np.zeros(2622)
        self.vp_exists = vp_exists
    
    def __repr__(self):
        return self.image_path()

    def image_path(self):
        if self.vp_exists != "" :
            return os.path.join(self.base, self.name, self.vp_exists, self.file)
        return os.path.join(self.base, self.name, self.file)

    def load_embedded(self, neural_network):
        if neural_network == "nn4": return self.embedded_nn4
        elif neural_network == "facenet": return self.embedded_facenet
        return self.embedded_vgg
    def set_embedded(self, neural_network, value):
        if neural_network == "nn4":
            self.embedded_nn4 = value
        elif neural_network == "facenet":
            self.embedded_facenet = value
        else:
            self.embedded_vgg = value
