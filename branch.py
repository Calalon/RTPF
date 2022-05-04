
class Branch:
    isBranch = True   # If this is a desision branch or no t
    greaterThanPath = -1
    lessThanPath = -1
    threshold = -1.0   #What value to check against
    decisionSample = -1
    value = type(float)
    #If this is a branch then value will be None and everything else will be
    #filled. Else value will be filled and this will be consired a leaf
    def __init__(self, greaterThanPath = -1, lessThanPath = -1, 
                 threshold = -1.0, decisionSample = -1, value = None):
        if(value == None):
            self.greaterThanPath = greaterThanPath
            self.lessThanPath = lessThanPath
            self.threshold = threshold 
            self.decisionSample = decisionSample
        else:
            self.isBranch = False 
            self.value = value
        
    def isLeaf(self):
        return not self.isBranch

    def getValue(self):
        return self.value

    def getDirection(self, signal):
        if(signal[self.decisionSample] < self.threshold):
            return self.lessThanPath
        else:
            return self.greaterThanPath
    