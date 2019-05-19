import os
import numpy as np

class Person:
    
    
    def __init__(self, base, name, file):
        self.name = name
        self.base = base
        self.file = file
        self.embedded = np.zeros(128)
    
    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 

        