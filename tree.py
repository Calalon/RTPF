from branch import Branch
import json


class Tree:
    branches = []
    age = 1
    scalingFactor = 1.0
    def __init__(self, treeDict, branchCount):
        self.branches = [None]*branchCount 
        data = json.loads(treeDict)
        self.AddBranch(data)

    # Parse Json recursivly to add branch objects to the tree
    def AddBranch(self, branchDict):
        if 'children' in branchDict:
            for subDict in branchDict['children']:
                self.AddBranch(subDict)
            branch = Branch(greaterThanPath = branchDict['no'],
                            lessThanPath = branchDict['yes'], 
                            threshold = branchDict['split_condition'],
                            decisionSample = int(
                                branchDict['split'].replace("f","")))
            self.branches[branchDict['nodeid']] = branch   
        else:
            branch = Branch(value = branchDict['leaf'])
            self.branches[branchDict['nodeid']] = branch
            
    # recursive function to get this trees prediction of the siganl  
    def Predict(self, signal, branchNumber = 0):
        branch = self.branches[branchNumber]
        if(branch.isLeaf()):
            return branch.getValue()*self.scalingFactor
        else:
            branchNumber = branch.getDirection(signal)
            return self.Predict(signal, branchNumber)
    
    # Manually set the age of the tree. Default age is 1
    def setAge(self, newAge):
        self.age = newAge
    
    #Effectivly multiply by lamda to reduce the effect of this tree
    def ageTree(self, ageingFactor):
        self.age = self.age * ageingFactor
        self.scalingFactor = self.scalingFactor * ageingFactor
        
    #Checks to see if tree is blindly applying a constant to prediction
    def isStump(self):
        if(self.branches[0].isLeaf()):
            return True
        return False