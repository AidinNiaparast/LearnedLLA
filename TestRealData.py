from sortedcontainers import SortedList
import LearnedLLA
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import statistics
import Constants

#NOTE: THE SIZE OF TEST DATA MUST BE A POWER OF TWO
#throughout the code, 'Classic' and 'Adaptive' correspond to PMA and APMA, respectively

path = 'Datasets\\'

#In all datasets, we use a prefix of size 2^18 = 262144 of the dataset as our input sequence
#all the files in the Datasets folder only contain this prefix, and not the whole dataset, in timestamp order

#reads user IDs in the MOOC dataset
def readMOOC():
    with open(path + 'MOOC.tsv') as f:
        lines = f.readlines()              
    data = []
    #each line contains actionID, userID, targetID, and timestamp
    for line in lines:
        userID = int(line.split()[1])
        data.append(userID)      
    return data    

#reads target IDs in the email-Eu-core dataset
def reademail_Eu_core():
    with open(path + 'email-Eu-core.txt') as f:
        lines = f.readlines()   
    data = []
    #each line contains sourceID, targetID , and timestamp
    for line in lines:
        targetID = int(line.split()[1])
        data.append(targetID)                 
    return data    

#reads target IDs in the AdkUbuntu answers-to-questions dataset
def readAskUbuntu():
    with open(path + 'AskUbuntu.txt') as f:
        lines = f.readlines()   
    data = []
    #each line contains sourceID, targetID, and timestamp
    for line in lines:
        targetID = int(line.split()[1])
        data.append(targetID) 
    return data 

#reads location IDs and Latitudes in the Gowalla dataset
def readGowalla(type):
    with open(path + 'Gowalla.txt') as f:
        lines = f.readlines()        
    data = []
    #each line contains userID, check-in time, latitude, longitude, and locationID
    for line in lines:
        userID,timestamp,latitude,longitude,locationID = line.split()
        if type == 'latitude':
            data.append(float(latitude))
        if type == 'locationID':
            data.append(int(locationID))       
    return data

#Generates predictions for the ranks of elements in testData based on the trainingData
#predictionType can be 'Prefix', or 'LineFit', which correspond to predictor_1 and predictor_2 in the paper, respectively
#We always assume training data comes before the test data, i.e. trainingDataEnd <= testDataStart
def generatePredictions(data, trainingDataStart, trainingDataEnd, testDataStart, testDataEnd, predictionType):
    trainingData = data[trainingDataStart:trainingDataEnd]
    testData = data[testDataStart:testDataEnd]
    trainingDataSize = len(trainingData)
    testDataSize = len(testData)
    predictions = []
    #fitting a line to the training data
    slope, intercept = np.polyfit(list(range(trainingDataSize)), trainingData, 1)
    if predictionType == 'Prefix':
        scaledTrainingData = trainingData
    elif predictionType == 'LineFit':
        scaledTrainingData = []
        for i in range(trainingDataSize):
            x = trainingData[i]
            scaledTrainingData.append(x + (testDataStart - trainingDataStart) * slope + i*(testDataSize/trainingDataSize-1)*slope)       
    else:
        print('Error: predictionType is not defined!')
        quit()
    sortedScaledTrainingData = SortedList(scaledTrainingData)
    for x in testData:
        prediction = math.floor(testDataSize / len(sortedScaledTrainingData) * sortedScaledTrainingData.bisect_left(x))  
        prediction = max(prediction, 0)
        prediction = min(prediction, testDataSize - 1)
        predictions.append(prediction)
    return predictions    
    
#samples uniformly at random badPredictionPercentage percent of the predictions and makes their error as large as possible
def perturbPredictions(predictions, badPredictionPercentage):
    n = len(predictions)
    perturbedPredictions = list(predictions)
    badPredictions = random.sample(range(n), n*badPredictionPercentage//100)
    for i in badPredictions:
        if perturbedPredictions[i] < n/2:
            perturbedPredictions[i] = n-1
        else:
            perturbedPredictions[i] = 0
    return perturbedPredictions            

#this function runs an LLA on the testData with the given predictions (if the LLA is learned) 
# and returns the average number of movements.
# If LLA is not learned it does not use the predictions   
# LLAType can be 'LearnedLLA+PMA', 'LearnedLLA+APMA', 'Classic', or 'Adaptive'. 
# The first two are learnedLLA that use PMA and APMA as the blackbox LLA, respectively.
# We insert a -infinity element first in all cases to make sure the internal predictor in APMA performs as expected.
def runSingleLLA(testData, predictions, LLAType):
    testDataSize = len(testData)
    #check if testDataSize is a power of two
    if testDataSize == 0 or math.floor(math.log2(testDataSize)) != math.ceil(math.log2(testDataSize)):
        print('Error: testDataSize is not a power of two in runSingleLLA', testDataSize)
        quit()

    isLearned = False
    if LLAType == 'Classic':
        pma = LearnedLLA.LearnedLLA(testDataSize, 'Classic', False)
    elif LLAType == 'Adaptive':
        pma = LearnedLLA.LearnedLLA(testDataSize, 'Adaptive', False)  
    elif LLAType == 'LearnedLLA+PMA':
        pma = LearnedLLA.LearnedLLA(testDataSize, 'Classic', True)
        isLearned = True
    elif LLAType == 'LearnedLLA+APMA':
        pma = LearnedLLA.LearnedLLA(testDataSize, 'Adaptive', True)
        isLearned = True
    else:    
        print('Error: LLAType is not defined!')
        quit()

    #insert -infinity first
    if isLearned:
        pma.insert((-1) * Constants.INF,0)
    else:
        pma.insert((-1) * Constants.INF)

    #we don't use the last element in the testData because the number of elements should be a power of two        
    for i in range(testDataSize - 1): 
        if isLearned:
            pma.insert(testData[i], predictions[i])
        else:
            pma.insert(testData[i])

    return pma.getTotalNumOfMovemenets()/testDataSize

#trains both Prefix (predictor_1 in the paper) and LineFit (predictor_2 in the paper) predictions on the 
# first half of the training data and tests both of them on the second half
# returns the best among these two. If size of the training data is < 10, it returns all zero predictions.
# blackboxType is 'Classic' or 'Adaptive', which correspond to PMA and APMA, respectively.
def findBestPrediction(data, trainingDataStart, trainingDataEnd, testDataStart, testDataEnd, blackboxType):
    trainingDataSize = trainingDataEnd - trainingDataStart
    testDataSize = testDataEnd - testDataStart
    if trainingDataSize < 10:
        return [0] * testDataSize
    trainingDataMid = (trainingDataStart + trainingDataEnd) // 2
    #We have to make sure that the size of the second half of the training data that we are going to use to test the quality of our predictions is a power of two
    secondPortionSize = trainingDataEnd - trainingDataMid
    secondPortionSize = 2 ** math.floor(math.log2(secondPortionSize))
    trainingDataMid = trainingDataEnd - secondPortionSize
    predictionsPrefix = generatePredictions(data, trainingDataStart, trainingDataMid, trainingDataMid, trainingDataEnd, 'Prefix')
    predictionsLineFit = generatePredictions(data, trainingDataStart, trainingDataMid, trainingDataMid, trainingDataEnd, 'LineFit')
    if blackboxType == 'Classic':
        averageMovementsPrefix = runSingleLLA(data[trainingDataMid: trainingDataEnd], predictionsPrefix, 'LearnedLLA+PMA')
        averageMovementsLineFit = runSingleLLA(data[trainingDataMid: trainingDataEnd], predictionsLineFit, 'LearnedLLA+PMA')
    elif blackboxType == 'Adaptive':
        averageMovementsPrefix = runSingleLLA(data[trainingDataMid: trainingDataEnd], predictionsPrefix, 'LearnedLLA+APMA')
        averageMovementsLineFit = runSingleLLA(data[trainingDataMid: trainingDataEnd], predictionsLineFit, 'LearnedLLA+APMA')
    else:
        print('Error: invalid blackboxType!')
        quit()    

    if averageMovementsPrefix < averageMovementsLineFit:
        return generatePredictions(data, trainingDataStart, trainingDataEnd, testDataStart, testDataEnd, 'Prefix')
    else:
        return generatePredictions(data, trainingDataStart, trainingDataEnd, testDataStart, testDataEnd, 'LineFit')
    
#outputs the plots of average number of movements for PMA, APMA, LearnedLLA+PMA, 
# and LearnedLLA+APMA, each with different test data sizes.
#For k = 12,...,17, we use the first and second n = 2^k portions of the
#  input as training data and test data, respectively.
def scaleTestDataSize(data, plotName = ''):
    print(plotName+': Scaling Test Data Size')
    n = len(data)
    testDataSizes = []
    PMAAverageMovements = []
    APMAAverageMovements = []
    LearnedLLAwithPMAAverageMovements = []
    LearnedLLAwithAPMAAverageMovements = []
    testDataSize = 4096
    testDataSizeLimit = min(n//2, 131072)
    while testDataSize <= testDataSizeLimit:
        trainingDataSize = testDataSize
        trainingData = data[:trainingDataSize]  
        testData = data[trainingDataSize : trainingDataSize + testDataSize]
        
        #PMA
        averageMovements = runSingleLLA(testData, [0] * testDataSize, 'Classic') #does not use predictions
        PMAAverageMovements.append(averageMovements)   
        #APMA
        averageMovements = runSingleLLA(testData, [0] * testDataSize, 'Adaptive') #does not use predictions
        APMAAverageMovements.append(averageMovements)    
        #LearnedLLA + PMA blackbox
        predictionsClassic = findBestPrediction(data, 0, trainingDataSize, trainingDataSize, trainingDataSize + testDataSize, 'Classic')
        averageMovements = runSingleLLA(testData, predictionsClassic, 'LearnedLLA+PMA')
        LearnedLLAwithPMAAverageMovements.append(averageMovements)    
        #LearnedLLA + APMA blackbox
        predictionsAdaptive = findBestPrediction(data, 0, trainingDataSize, trainingDataSize, trainingDataSize + testDataSize, 'Adaptive')
        averageMovements = runSingleLLA(testData, predictionsAdaptive, 'LearnedLLA+APMA')
        LearnedLLAwithAPMAAverageMovements.append(averageMovements)    
        
        testDataSizes.append(testDataSize)
        testDataSize *= 2

    print('PMA Amortized Cost =', PMAAverageMovements)
    print('APMA Amortized Cost =', APMAAverageMovements)
    print('LearnedLLA+PMA Amortized Cost =', LearnedLLAwithPMAAverageMovements)
    print('LearnedLLA+APMA Amortized Cost = ', LearnedLLAwithAPMAAverageMovements)
    print('Test Data Sizes =', testDataSizes)
    print('--------------------------------------------------------------------------')
    logTestDataSizes = [round(math.log2(x)) for x in testDataSizes]
    plt.plot(logTestDataSizes, PMAAverageMovements, color='blue', linestyle='-', label='PMA')
    plt.plot(logTestDataSizes, APMAAverageMovements, color='blue', linestyle='--', label='APMA')
    plt.plot(logTestDataSizes, LearnedLLAwithPMAAverageMovements, color='darkorange', linestyle='-', label='LearnedLLA+PMA')
    plt.plot(logTestDataSizes, LearnedLLAwithAPMAAverageMovements, color='darkorange', linestyle='--', label='LearnedLLA+APMA')

    plt.xlabel("k=log(n)")
    plt.ylabel("Amortized Cost")
    plt.title(plotName+': Scaling Test Data Size')
    plt.legend(loc='lower right')
    plt.show()

#outputs the plots of average number of movements for PMA, APMA, LearnedLLA+PMA, 
# and LearnedLLA+APMA, each with different training data sizes.
#In all the experiments test data size is 2^17 = 131072, and 
#training data comes right before the test data
def scaleTrainingDataSize(data, plotName = ''):
    print(plotName + ': Scaling Training Data Size')
    n = len(data)
    PMAAverageMovements = []
    APMAAverageMovements = []
    LearnedLLAwithPMAAverageMovements = []
    LearnedLLAwithAPMAAverageMovements = []

    testDataSize = min(2 ** math.floor(math.log2(n//2)), 131072)

    trainingDataSizes = [i * testDataSize//20 for i in range(9)] #corresponding to 0,5,10,15,20,25,30,35,40 percent of the test data
    for trainingDataSize in trainingDataSizes:
        trainingData = data[testDataSize - trainingDataSize : testDataSize]  
        testData = data[testDataSize : 2*testDataSize]        
        predictionsClassic = findBestPrediction(data, testDataSize - trainingDataSize, testDataSize, testDataSize, 2*testDataSize, 'Classic')
        predictionsAdaptive = findBestPrediction(data, testDataSize - trainingDataSize, testDataSize, testDataSize, 2*testDataSize, 'Adaptive')

        #we only need to run classic and adaptive PMAs once because they do not depend on the training data
        if trainingDataSize == trainingDataSizes[0]:
            #Classic
            averageMovements = runSingleLLA(testData, [0] * testDataSize, 'Classic') #does not use predictions
            PMAAverageMovements = [averageMovements] * len(trainingDataSizes)   
            #Adaptive
            averageMovements = runSingleLLA(testData, [0] * testDataSize, 'Adaptive') #does not use predictions
            APMAAverageMovements = [averageMovements] * len(trainingDataSizes)   
            
        #LearnedLLA + PMA blackbox
        averageMovements = runSingleLLA(testData, predictionsClassic, 'LearnedLLA+PMA')
        LearnedLLAwithPMAAverageMovements.append(averageMovements)    
        
        #LearnedLLA + APMA blackbox
        averageMovements = runSingleLLA(testData, predictionsAdaptive, 'LearnedLLA+APMA')
        LearnedLLAwithAPMAAverageMovements.append(averageMovements)    

    print('PMA Amortized Cost =', PMAAverageMovements)
    print('APMA Amortized Cost =', APMAAverageMovements)
    print('LearnedLLA+PMA Amortized Cost =', LearnedLLAwithPMAAverageMovements)
    print('LearnedLLA+APMA Amortized Cost =', LearnedLLAwithAPMAAverageMovements)
    print('Training Data Sizes =', trainingDataSizes)
    print('Test Data Size =', testDataSize)
    print('--------------------------------------------------------------------------')
    
    trainingDataSizesPercentage = [x/testDataSize*100 for x in trainingDataSizes]
    plt.plot(trainingDataSizesPercentage, PMAAverageMovements, color='blue', linestyle='-', label='PMA')
    plt.plot(trainingDataSizesPercentage, APMAAverageMovements, color='blue', linestyle='--', label='APMA')
    plt.plot(trainingDataSizesPercentage, LearnedLLAwithPMAAverageMovements, color='darkorange', linestyle='-', label='LearnedLLA+PMA')
    plt.plot(trainingDataSizesPercentage, LearnedLLAwithAPMAAverageMovements, color='darkorange', linestyle='--', label='LearnedLLA+APMA')
    plt.xlabel("Training Data Size Percentage")
    plt.ylabel("Amortized Cost")
    plt.title(plotName + ': Scaling Training Data Size')
    plt.legend(loc='lower right')
    plt.show()

#outputs the plots of average number of movements for PMA, APMA, LearnedLLA+PMA, 
# and LearnedLLA+APMA, each with different percentages of "bad" predictions.
#In all the experiments we use the first and second 2^16 = 65536 portion of
#the input as training and test data, respectively.
def robustness(data, plotName = ''):
    print(plotName + ': Robustness')
    n = len(data)
    PMAAverageMovements = []
    APMAAverageMovements = []
    LearnedLLAwithPMAAverageMovementsPerturbed = []
    LearnedLLAwithAPMAAverageMovementsPerturbed = []
    LearnedLLAwithPMAAverageMovementsPerturbedSD = [] #SD= Standard Deviation
    LearnedLLAwithAPMAAverageMovementsPerturbedSD = []
    testDataSize = min(2 ** math.floor(math.log2(n//2)), 65536)
    badPredictionsPercentages = [0,5,10,15,20,25,30,35,40]
    perturbedClassicResults = {}
    perturbedAdaptiveResults = {}
    numExperiments = 5

    trainingDataSize = testDataSize
    trainingData = data[:trainingDataSize]  
    testData = data[trainingDataSize : trainingDataSize + testDataSize]
    predictionsClassic = findBestPrediction(data, 0, trainingDataSize, testDataSize, 2*testDataSize, 'Classic')
    predictionsAdaptive = findBestPrediction(data, 0, trainingDataSize, testDataSize, 2*testDataSize, 'Adaptive')

    #learned
    for percentage in badPredictionsPercentages:
        #LearnedLLA + PMA blackbox
        totalAverageMovements = 0
        perturbedClassicResults[percentage] = []
        for _ in range(numExperiments):
            perturbedPredictionsClassic = perturbPredictions(predictionsClassic, percentage)
            averageMovements = runSingleLLA(testData, perturbedPredictionsClassic, 'LearnedLLA+PMA')
            totalAverageMovements += averageMovements
            perturbedClassicResults[percentage].append(averageMovements)   
        LearnedLLAwithPMAAverageMovementsPerturbed.append(totalAverageMovements / numExperiments) 
        LearnedLLAwithPMAAverageMovementsPerturbedSD.append(statistics.pstdev(perturbedClassicResults[percentage]))

        #LearnedLLA + APMA blackbox
        totalAverageMovements = 0
        perturbedAdaptiveResults[percentage] = []
        for _ in range(numExperiments):
            perturbedPredictionsAdaptive = perturbPredictions(predictionsAdaptive, percentage)
            averageMovements = runSingleLLA(testData, perturbedPredictionsAdaptive, 'LearnedLLA+APMA')
            totalAverageMovements += averageMovements
            perturbedAdaptiveResults[percentage].append(averageMovements)   
        LearnedLLAwithAPMAAverageMovementsPerturbed.append(totalAverageMovements / numExperiments) 
        LearnedLLAwithAPMAAverageMovementsPerturbedSD.append(statistics.pstdev(perturbedAdaptiveResults[percentage]))

    #Classic
    averageMovements = runSingleLLA(testData, [0] * testDataSize, 'Classic') #does not use predictions
    PMAAverageMovements = [averageMovements] * len(badPredictionsPercentages)   
    #Adaptive
    averageMovements = runSingleLLA(testData, [0] * testDataSize, 'Adaptive') #does not use predictions
    APMAAverageMovements = [averageMovements] * len(badPredictionsPercentages)   

    print('PMA Amortized Cost =', PMAAverageMovements)
    print('APMA Amortized Cost =', APMAAverageMovements)
    print('LearnedLLA+PMA Amortized Cost (perturbed predictions, mean of experiments) = \n', LearnedLLAwithPMAAverageMovementsPerturbed)
    print('LearnedLLA+PMA Amortized Cost (perturbed predictions, standard deviation of experiments) = \n', LearnedLLAwithPMAAverageMovementsPerturbedSD)
    print('LearnedLLA+APMA Amortized Cost (perturbed predictions, mean of experiments) = \n', LearnedLLAwithAPMAAverageMovementsPerturbed)
    print('LearnedLLA+APMA Amortized Cost (perturbed predictions, standard deviation of experiments) = \n', LearnedLLAwithAPMAAverageMovementsPerturbedSD)
    print('Bad Predictions Percentages =', badPredictionsPercentages)
    print('Test Data Size=', testDataSize)
    
    xValues = badPredictionsPercentages
    plt.plot(xValues, PMAAverageMovements, color='blue', linestyle='-', label='PMA')
    plt.plot(xValues, APMAAverageMovements, color='blue', linestyle='--', label='APMA')
    plt.plot(xValues, LearnedLLAwithPMAAverageMovementsPerturbed, color='darkorange', linestyle='-', label='LearnedLLA+PMA')
    plt.plot(xValues, LearnedLLAwithAPMAAverageMovementsPerturbed, color='darkorange', linestyle='--', label='LearnedLLA+APMA')
    plt.fill_between(xValues, (np.array(LearnedLLAwithPMAAverageMovementsPerturbed) - np.array(LearnedLLAwithPMAAverageMovementsPerturbedSD)).tolist(), 
                 (np.array(LearnedLLAwithPMAAverageMovementsPerturbed) + np.array(LearnedLLAwithPMAAverageMovementsPerturbedSD)).tolist(), alpha=0.5,
                   edgecolor='darkorange', facecolor='pink')
    plt.fill_between(xValues, (np.array(LearnedLLAwithAPMAAverageMovementsPerturbed) - np.array(LearnedLLAwithAPMAAverageMovementsPerturbedSD)).tolist(), 
                 (np.array(LearnedLLAwithAPMAAverageMovementsPerturbed) + np.array(LearnedLLAwithAPMAAverageMovementsPerturbedSD)).tolist(), alpha=0.5,
                   edgecolor='darkorange', facecolor='gold', linestyle = '-')
    plt.xlabel('Percentage of Bad Predictions')
    plt.ylabel("Amortized Cost")
    plt.title(plotName + ': Robustness')
    plt.legend(loc='lower right')
    plt.show()

#generates the plots and prints numerical results for the dataset
#datasetName can be 'Gowalla Latitude', 'Gowalla LocationID', 'MOOC', 'AskUbuntu', or 'email-Eu-core'
def getResults(datasetName):
    match datasetName:
        case 'Gowalla Latitude':
            plotName = 'Gowalla Latitude'
            data = readGowalla('latitude')
        case 'Gowalla LocationID':
            plotName = 'Gowalla LocationID'
            data = readGowalla('locationID')
        case 'MOOC':
            plotName = 'MOOC User ID'  
            data = readMOOC()
        case 'AskUbuntu':
            plotName = 'AskUbuntu a2q Target ID'  
            data = readAskUbuntu()
        case 'email-Eu-core':
            plotName = 'email-Eu-core Target ID'  
            data = reademail_Eu_core()
    scaleTestDataSize(data, plotName)
    scaleTrainingDataSize(data, plotName)
    robustness(data, plotName) 

def main():
    getResults('Gowalla Latitude')
    #getResults('Gowalla LocationID')
    #getResults('MOOC')
    #getResults('AskUbuntu')
    #getResults('email-Eu-core')

if __name__ == "__main__":
    main()