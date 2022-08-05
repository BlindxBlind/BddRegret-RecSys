import numpy as np

def ArrivalGen(AgentNum, maxTime, type='Normal'):
    Process_types = ['Normal', 'Exp']
    if type not in Process_types:
        raise ValueError("Invalid sim type. Expected one of: %s" % Process_types)
    if type == 'Normal':
        arrivalList = []
        for j in range(AgentNum):
            tempList = []
            arrivalTime = 0
            meaninter = np.random.uniform(2, 20)
            stddev = meaninter/2
            interSampled = np.random.normal(meaninter, stddev)
            arrivalTime = np.maximum(interSampled, 0.2) + arrivalTime
            while arrivalTime < maxTime:
                tempList.append(np.array([int(j), arrivalTime]))
                interSampled = np.random.normal(meaninter, stddev)
                arrivalTime = np.maximum(interSampled, 0.2) + arrivalTime
            arrivalList = arrivalList + tempList
        sortedList = sorted(arrivalList, key=lambda x:x[1]) #sort all arrivals in the order of arrival time
        sortedArray = np.vstack(sortedList) # Make it array
    # Exponential arrival case
    if type == 'Exp':
        arrivalList = []
        for j in range(AgentNum):
            tempList = []
            arrivalTime = 0
            meaninter = np.random.uniform(2, 20)
            interSampled = np.random.exponential(1/meaninter, stddev)
            arrivalTime = interSampled + arrivalTime
            while arrivalTime < maxTime:
                tempList.append(np.array([j, arrivalTime]))
                interSampled = np.random.exponential(1/meaninter, stddev)
                arrivalTime = interSampled + arrivalTime
            arrivalList = arrivalList + tempList
        sortedList = sorted(arrivalList, key=lambda x:x[1]) #sort all arrivals in the order of arrival time
        sortedArray = np.vstack(sortedList) # Make it array    

    return sortedArray

def AgentGen(AgentNum, dim):
    listFeatureSet = []
    count = 0
    while count < AgentNum:
        count = count+1
        tempVec = np.random.normal(0,1,dim)
        tempSquared = np.square(tempVec)
        tempSum = np.sum(tempSquared)
        featureVec = tempVec/np.sqrt(tempSum)
        listFeatureSet = listFeatureSet + [featureVec] 
        featureSet= np.vstack(listFeatureSet)
    return featureSet

def ArmGen(ArmNum, dim):
    listFeatureSet = []
    for _ in range(ArmNum):
        tempVec = np.random.normal(0,1,dim)
        tempSum = np.sum(np.square(tempVec))
        featureVec = tempVec/np.sqrt(tempSum)
        listFeatureSet = listFeatureSet + [featureVec] 
        featureSet= np.vstack(listFeatureSet)
    return featureSet

def gapMatrixGen(agentFeatureSet, armFeatureSet): #The matrix of regret
    if np.shape(agentFeatureSet)[1] !=  np.shape(armFeatureSet)[1] :
        raise ValueError("Invalid feature set input") 
    rewardsMatrix = np.matmul(agentFeatureSet, armFeatureSet.T)
    bestarms = np.amax(rewardsMatrix, axis = 1) #it returns a tranposed array of max along axis 1
    temp = bestarms - rewardsMatrix.T
    gapMatrix = temp.T
    return gapMatrix

def Condition1Checker(agentFeatureSet, armFeatureSet, dim):
    gapMatrix = gapMatrixGen(agentFeatureSet, armFeatureSet)
    optTFMatrix = np.where(gapMatrix == 0, 1, 0)
    armOptAgentNum = np.sum(optTFMatrix, axis = 0) #get the number of agents which has each arm as the best
    conditionCheck = armOptAgentNum>=dim #get True or False vector of whether each arm has enough best number of agents
    print(armOptAgentNum)
    return np.all(conditionCheck) #check if all the elemets are nonzero
