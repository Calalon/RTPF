print("RTPF Running")

from dataHandler import DataHandler
import numpy as np
from forest import Forest
from fir import FIR
from RLSPredictor import RLS
import matplotlib.pyplot as plt
import sys
from numpy.linalg import inv
import getopt


#Constants to play around with
###########################################################
##Predictor Parameters
adaptiveTree = False
linearPredict = True
adaptiveLinearPredict = False
##Tree Parameters
treeCount = 1000
treeDepth = 6
groveSize = 100  #how many trees to age in groups   
updateTime = 100 #How many samples to go before doing any tree updates
residueSize = 500
##Training Data Parameters
trainingSize = 1000 #how many samples should be used to train the new trees
taps = 10  #number of inputs for prediction
ageFactor = .7
lam = .98 # Aging factor
delay = 1  #How far to predict into the future
ageThreshhold = ageFactor ** (treeCount / groveSize - 1) # Age that a grove should die
##Graph Parameters
ivpStart = 0
ivpEnd = 100
mapX1Start = -10
mapX1End = 10
mapX2Start = -10
mapX2End = 10

dataLocation = 'NonLinearData.csv'  #'LinearData.csv' 'cosineData.csv' 'NonLinearData.csv'

###########################################################
argv = sys.argv[1:]
options = [
    'RegTree', 'NonAdaptLin', 'AdaptLin', 'TreeCount=', 
    'TreeDepth=', 'GroveSize=', 'UpdateTime=', 'ResidueSize=', 
    'TrainingSize=', 'Taps=', 'AgeFactor=', 'Lambda=', 'Delay=', 
    'ivpStart=', 'ivpEnd =', 'mapX1Start=', 'mapX1End=', 
    'mapX2Start=', 'mapX2End=', 'help'
    ]
optionsHelp = [
    "Will run the regression tree filter if present.",
    "Will run the linear filter if present.",
    "Will run the adaptive RLS linear filter if present.",
    "The total number of trees to have in the forest.",
    "The max depth each tree is allowed to be trained. ",
    "The number of trees for each grove. ",
    "How many samples between adapting the forest. ",
    "How many samples to use to use when adapting the forest.",
    "Number of samples to use to train the forest.",
    "How many past samples to use for predicting. ",
    "How much to scale old trees by each update time.",
    "The scaling factor to be used in the RLS algorithm.",
    "How far to predict into the future. ",
    "The sample to begin the input vs predicted signal graph",
    "The sample to end the input vs predicted signal graph",
    "The lower bound to map for samples at t = 0 ",
    "The upper bound to map for samples at t = 0 ",
    "The lower bound to map for samples at t = -1 ",
    "The upper bound to map for samples at t = -1 ",
    "Print the usage Statement"
    ]    
    
    
usage = "Usage: \nRTAP.py <Path to input signal> [Optional: changes to \
 default parameters]"
try:    
    opts, args = getopt.getopt(argv, 'h', options)
except getopt.GetoptError:
    print("Unknown Command Line arguments")
    print(usage)
    sys.exit(2)     
print(args)


if(len(argv) > 0):
    dataLocation = argv[0]
    print("datachangesd")
elif(len(args) > 1):
    print("Unknown Command Line argument \"" + args[0] +"\"")
    print(usage)
    sys.exit(2)
  
for opt, arg in opts:
    if opt == '-h':
        print(usage)
        sys.exit()   
    elif opt == '--RegTree':
        adaptiveTree = True
    elif opt == "--NonAdaptLin":
        linearPredict = True
    elif opt == "--AdaptLin":
        adaptiveLinearPredict = True
    elif opt == "--TreeCount":
        treeCount = int(arg)
    elif opt == "--TreeDepth":
        treeDepth = int(arg)
    elif opt == "--GroveSize":
        groveSize = int(arg)
    elif opt == "--UpdateTime":
        updateTime = int(arg)
    elif opt == "--ResidueSize":
        residueSize = int(arg)
    elif opt == "--TrainingSize":
        trainingSize = int(arg)
    elif opt == "--Taps":
        taps = int(arg)
    elif opt == "--AgeFactor":
        ageFactor = float(arg)
    elif opt == "--Lambda":
        lam = float(arg)
    elif opt == "--Delay":
        delay = int(arg)
    elif opt == "--ivpStart":
        ivpStart = float(arg)
    elif opt == "--ivpEnd":
        ivpEnd = float(arg)
    elif opt == "--mapX1Start":
        mapX1Start = float(arg)
    elif opt == "--mapX1End":
        mapX1End = float(arg)
    elif opt == "--mapX2Start":
        mapX2Start = float(arg)
    elif opt == "--mapX2End":
        mapX2End = float(arg)        
    elif opt == "--help":
        print(usage)
        for hp in range(len(options)):
            print("--" + options[hp].replace('=', ''))
            print("\t" + optionsHelp[hp])
            




if(treeCount % groveSize):
    sys.exit('Critical Error: TreeCount must be divisible by GroveCount')


#####Input
# Load in data 
# Format the data properly. 
data = DataHandler(dataLocation, taps, delay)


initialTrainingInput = data.inputData[0:trainingSize]
initialTrainingOutput = data.outputData[0:trainingSize] 

print("Data processed")
###########################################################

if(adaptiveTree):


    param = {
        'max_depth': treeDepth,     # How many levels the trees should have
        'base_score': 0             # additive scaling factor, this should be left at 0
        }
    #The forest to be used for the duration of the program
    redWoods = Forest(groveSize, treeDepth, ageFactor, ageThreshhold, param)
    
    
    redWoods.ClearForest()

    redWoods.GrowNewTrees(initialTrainingInput, 
                          initialTrainingOutput, treeCount)
    
    stumpCount = redWoods.GetStumpCount();
    if(stumpCount > 1):
        print("Warning: Initial Training of trees contained " 
              + str(stumpCount)+ " stumps");

    redWoods.IntialTreeAge()
    
    
    treePrdictions = []
    treeError = []
    update = 0
   
    for i in range(len(data.inputData) - trainingSize):
        index = i + trainingSize
        update += 1
        prediction = redWoods.Predict(data.inputData[index])
        treeError.append(abs(prediction - data.outputData[index]))
        treePrdictions.append(prediction)    
        if(update == updateTime):
            update = 0
            redWoods.AgeForest()
            redWoods.PruneForest()
            residue = []
            for j in range(-residueSize,0):
                prediction = redWoods.Predict(data.inputData[index + j])
                error = data.outputData[index + j] - prediction
                residue.append(error)
            redWoods.GrowNewTrees(data.inputData[index - residueSize:index], 
                                  np.array(residue), groveSize)

                
                
    stumpCount = redWoods.GetStumpCount();
    if(stumpCount > 1):
        print("Warning: Final forest contained " + str(stumpCount)
              + " stumps");            
    ### Graph Everything
    plt.figure(1)
    plt.clf()
    plt.title("Tree Prediciton vs signal")
    plt.plot(treePrdictions[ivpStart:ivpEnd], label="Tree Prediction")
    plt.plot(data.outputData[trainingSize+ivpStart:trainingSize+ivpEnd],
             label="Signal")
    plt.legend(loc="upper right")
    plt.show()
    plt.figure(2)
    plt.clf()
    plt.title("Tree Error")
    plt.plot(treeError)
    print("Regression Tree Adaptive Predictor Statistics")
    print("Varience of Error")
    print(np.var(treeError))
    print("MSE:")
    print(np.square(treeError).mean())
    
    if(taps == 2 ):
        presision = 100
        x = np.linspace(mapX1Start, mapX1End, presision)
        y = np.linspace(mapX2Start, mapX2End, presision)
    
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((presision, presision))
        for i in range(presision):
            for j in range(presision):
                Z[i, j] = redWoods.Predict([X[i,j], Y[i,j]])
        plt.figure(3)   
        
        plt.clf()
       
        ax = plt.axes(projection='3d')
        ax.set_title("Tree Predictor")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                   cmap='viridis', edgecolor='none')

##########################################################################
if (linearPredict):

    # find coefs 
    h = np.matmul(inv(np.matmul(initialTrainingInput.transpose(),
                                initialTrainingInput)),
                  np.matmul(initialTrainingInput.transpose(),
                            initialTrainingOutput))
    
    fir = FIR(h)
    
    firPredictions = []
    firError = []
    
    for i in range(len(data.inputData) - trainingSize):
        prediction = fir.Predict(data.inputData[i + trainingSize])
        firError.append(abs(prediction - data.outputData[i + trainingSize]))
        firPredictions.append(prediction)    
        
    
    plt.figure(4)
    plt.clf()
    plt.title("Linear Prediciton vs signal")
    plt.plot(firPredictions[ivpStart:ivpEnd], label="Linear Prediction")
    plt.plot(data.outputData[trainingSize+ivpStart:trainingSize+ivpEnd],
             label="Signal")
    plt.legend(loc="upper right")
    plt.show()
    plt.figure(5)
    plt.clf()
    plt.title("Tree Error")
    plt.plot(firError)
    print("Linear Predictor Statistics")
    print("Varience of Error")
    print(np.var(firError))
    print("MSE:")
    print(np.square(firError).mean())
    
    if(taps == 2):
        presision = 100
        x = np.linspace(mapX1Start, mapX1End, presision)
        y = np.linspace(mapX2Start, mapX2End, presision)
        
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((presision, presision))
        for i in range(presision):
            for j in range(presision):
                Z[i, j] = fir.Predict([X[i,j], Y[i,j]])
        plt.figure(6) 
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.set_title("Linear Predictor")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                   cmap='viridis', edgecolor='none')
        
############################################################################### 
if(adaptiveLinearPredict):
    rls = RLS(taps, lam)
    
    for i in range(trainingSize):
        rls.Update(data.inputData[i], data.outputData[i])
        
        
    rlsPredictions = []
    rlsError = []
    
    for i in range(len(data.inputData) - trainingSize):
        prediction = rls.Predict(data.inputData[i + trainingSize])
        rlsError.append(abs(prediction - data.outputData[i + trainingSize]))
        rlsPredictions.append(prediction)    
        
        
    plt.figure(7)
    plt.clf()
    plt.title("Linear Prediciton vs signal")
    plt.plot(rlsPredictions[ivpStart:ivpEnd], label="Linear Prediction")
    plt.plot(data.outputData[trainingSize+ivpStart:trainingSize+ivpEnd],
             label="Signal")
    plt.legend(loc="upper right")
    plt.show()
    plt.figure(8)
    plt.clf()
    plt.title("Linear Error")
    plt.plot(rlsError)
    print("RLS Predictor Statistics")
    print("Varience of Error")
    print(np.var(rlsError))
    print("MSE:")
    print(np.square(rlsError).mean())
    if(taps == 2):
        presision = 100
        x = np.linspace(mapX1Start, mapX1End, presision)
        y = np.linspace(mapX2Start, mapX2End, presision)
        
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((presision, presision))
        for i in range(presision):
            for j in range(presision):
                Z[i, j] = rls.Predict([X[i,j], Y[i,j]])
        plt.figure(9) 
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.set_title("Linear Predictor")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                   cmap='viridis', edgecolor='none')

print("General Prediction Parameters")
print("Delay = " + str(delay))
print("Training Size: " + str(trainingSize))
print("Taps: " + str(taps))
if(adaptiveLinearPredict):
    print("RLS Lambda: " + str(lam))
if(adaptiveTree):
    print("Tree Count: " + str(treeCount))
    print("Tree Depth: " + str(treeDepth))
    print("Grove Size: " + str(groveSize))
    print("Update Time: " + str(updateTime))
    print("Residue Size: " + str(residueSize))

    
    