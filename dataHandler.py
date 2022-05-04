import pandas as pd
import numpy as np

class DataHandler:
    observedData = [] # All data present
    inputData = None # Data that goes into the predictor
    outputData = None # What the predicitonis made against.
    def __init__(self, dataLocation, taps, delay):
        self.data = pd.read_csv(dataLocation, header=None)
        data = self.data.values.tolist()[0]
        
        for i in range(len(data) - (taps+delay + 1)):
            sample = np.append(np.array(data[i:i+taps]),
                               np.array(data[i+taps+delay-1]))
            #   sample =  sample.append(inputSignal[i+taps+delay])
            if(i == 0):
                self.observedData = sample
            else:
                self.observedData = np.vstack([self.observedData, sample])
        self.inputData = np.ascontiguousarray(self.observedData[:,0:taps]) 
        self.outputData = np.ascontiguousarray(self.observedData[:,taps]) 

        
    