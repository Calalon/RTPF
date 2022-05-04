import numpy as np

class FIR:
    h = []
    
    def __init__(self, h):
        self.h = h
        
    def Predict(self, data):
        return np.dot(data, self.h)
        