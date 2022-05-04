import json
from tree import Tree
import xgboost as xgb

def parseTrees(trees):
    listOfTrees = []
    for tree in trees:
        listOfTrees.append(json.loads(tree))
    return listOfTrees    


class Forest:
    trees = []
    groveSize = 1
    treeBranchCount = 0; # the number of branches in each tree
    lamb = 1 #how much to age the trees by
    deathAge = 0
    param = {}
    
    #Set the grove size (amount of trees in a batch)
    #and finds the expected amount of branches
    def __init__(self, groveSize, treeDepth, lamb, deathAge, param):
        self.groveSize = groveSize
        for i in range(treeDepth + 1):
            self.treeBranchCount += 2**i
        self.lamb = lamb
        self.deathAge = deathAge
        self.param = param
        
    def ClearForest(self):
        self.trees.clear()
        
    # Grows and adds treeCount amount of trees to the forest based off the training data
    def GrowNewTrees(self, trainingInput, trainingOutput, treeCount):
        D_InitialTrain = xgb.DMatrix(trainingInput, label=trainingOutput)
        model = xgb.train(self.param, D_InitialTrain, treeCount)
        jsonTrees = model.get_dump(with_stats=True, dump_format="json")
        self.AddTrees(jsonTrees)
    
        
    #Adds trees to the front of the forest
    def AddTrees(self, jsonTrees):
        for jsonTree in jsonTrees:
            self.trees.append(Tree(jsonTree, self.treeBranchCount))
           
    #uses all the availible trees to make a prediction
    def Predict(self, signal):
        prediction = 0 
        for tree in self.trees:
            prediction += tree.Predict(signal)
        return prediction
    
    #Base case of algorithm for sorting
    def IntialTreeAge(self):
        groveCount = 0
        age = 1
        for tree in self.trees:
            tree.setAge(age)
            groveCount += 1
            if(groveCount == self.groveSize):
                age *= self.lamb
                groveCount = 0
             
    #applies Scaling factor on all trees in the forest
    def AgeForest(self):
        for tree in self.trees:
            tree.ageTree(self.lamb)
            
    # Removes all trees that are at a certain age threshold
    def PruneForest(self):
        self.trees = [tree for tree in self.trees 
                      if not tree.age < self.deathAge]

    #Totals the amount of stumps in the forest
    def GetStumpCount(self):
        count = 0
        for tree in self.trees:
            if(tree.isStump()):
                count += 1

        return count
            