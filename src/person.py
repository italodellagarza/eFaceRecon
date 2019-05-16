import os

class Person:
    
    
    def __init__(self, base, name, file):
        self.name = name
        self.base = base
        self.file = file
    
    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 

        