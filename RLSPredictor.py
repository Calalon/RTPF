import numpy as np

class RLS:
    h = []
    lamb = 1
    P = []
    
    def __init__(self, taps, lamb):
        self.lamb = lamb
        self.P = np.identity(taps) * lamb
        self.h = np.zeros((taps,1))
        
        
        
    def Update(self, q, d):
        q = np.array([q]).T
        lamb = 1 / self.lamb
        k = (lamb * np.matmul(self.P, q) 
             / (1 + lamb * np.matmul(np.matmul(q.T, self.P), q)))

        self.P = np.add(lamb * self.P,
                        -lamb * np.matmul(np.matmul(k, q.T), self.P))
        eps = d - np.matmul(q.T, self.h)
        self.h = np.add(self.h, k*eps)
        
    def Predict(self, q):
        return np.inner(q, self.h.flatten())