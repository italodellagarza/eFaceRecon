import os
import numpy as np

class Person:
    
    
    def __init__(self, base, name, file):
        self.name = name
        self.base = base
        self.file = file
        self.embedded_nn4 = np.zeros(128)
        self.embedded_facenet = np.zeros(128)
        self.embedded_vgg = np.zeros(2622)
    
    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)

    def load_embedded(self, neural_network):
        if neural_network == "nn4": return self.embedded_nn4
        elif neural_network == "facenet": return self.embedded_facenet
        return self.embedded_vgg
