import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("test")
    if __name__ == "__main__":
        main()
    print(ArmGen(3,4))    
        


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
    rewardsMatrix = np.inner(agentFeatureSet, armFeatureSet)
    bestarms = np.amax(rewardsMatrix, axis = 1) #it returns a tranposed array of max along axis 1
    temp = bestarms - rewardsMatrix.T
    gapMatrix = temp.T

    return gapMatrix
    #agentNum = np.shape(agentFeatureSet)[1]
    #armNum = np.shape(armFeatureSet)[1]



def Bandit(AgentNum, ArmNum, dim, trialNum, maxTime):
    noiseStddev = 0.3
    globalRegret1 = np.zeros(10000) # UCB's global mean of Regret at each time - anyarrival
    globalRegret2 = np.zeros(10000) # CFUCB's global mean of Regret at each time - anyarrival
    individualRegretSum1 = np.zeros(1000) # sum of UCB agents' sum of regrets until n for the trials until now
    individualRegretSum2 = np.zeros(1000) # sum of CFUCB agents' sum of regrets until n for the trials until now
    
    trialMeanRegret1 = np.zeros(10000) 

    maxLength =100000000
    for trial in np.arange(1,trialNum):
        storageProb1 = np.zeros((AgentNum, ArmNum, 2)) # the (agentNum x armNum) matrix of (#pullings, empirical mean) for UCB algorithm
        storageProb2 = np.zeros((AgentNum, ArmNum, 2)) # the (agentNum x armNum) matrix of (#pullings, empirical mean) for CFUCB algorithm
            #Arrival Generation
        arrivals = ArrivalGen(AgentNum, maxTime, type='Normal') # long array of [index, arrival time]
        maxLength = np.minimum(maxLength, len(arrivals))
            #Agent feature set gen
        agentFeatureSet = AgentGen(AgentNum, dim)
            #Arm feature set gen
        armFeatureSet = ArmGen(AgentNum, dim)
            #Gap matrix gen
        rewardsMatrix = np.inner(agentFeatureSet, armFeatureSet)
        gapMatrix = gapMatrixGen(agentFeatureSet, armFeatureSet)
            #Now the bandit starts
        
        ############################################
        # UCB algorithm part
        arrivalCount = 0
        
        for arrival in arrivals:
            arrivalCount = arrivalCount + 1
            agentIndex = int(arrival[0])   
            pullsVector = storageProb1[agentIndex, :, 0].astype(int)
            totalPull = np.sum(pullsVector)
            meansVector = storageProb1[agentIndex, :, 1]
            widthVector = pullsVector
            widthVector = np.where((widthVector == 0), np.ones(ArmNum)*10000, np.sqrt(np.log(totalPull)/pullsVector))
            ucbVector = meansVector + widthVector #ucb of all arms
            armIndex = np.argmax(ucbVector)
            rewardObserved = rewardsMatrix[agentIndex][armIndex]+np.random.normal(0,noiseStddev) #reward observed
            pullOfChosen = storageProb1[agentIndex, armIndex, 0]
            meanOfChosen = storageProb1[agentIndex, armIndex, 1]
            storageProb1[agentIndex, armIndex, 1] = (pullOfChosen*meanOfChosen+rewardObserved)/(pullOfChosen+1) # mean update
            storageProb1[agentIndex, armIndex, 0] = storageProb1[agentIndex, armIndex, 0]+1 # pull number update
            globalRegret1[arrivalCount] = (globalRegret1[arrivalCount]*(trial-1) + gapMatrix[agentIndex, armIndex])/trial
            
        #####Finish of all the arrivals of one trial####
  
        #CFUCB algorithm part  

    globalRegret1 = (globalRegret1[0:maxLength])

        
        #########Finish of one trial ###
    
    #After all trials

    Xglobal=np.arange(len(globalRegret1))
    globalCumRegret1 = np.cumsum(globalRegret1)
    globalCumRegret2 = np.cumsum(globalRegret2)
    Y1 = globalCumRegret1
    Y2 = globalCumRegret2

    plt.plot(Xglobal, Y1, color='r', label='UCB')
    #plt.plot(Xglobal, Y2, color='g', label='CFUCB')
        # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Total arrivals")
    plt.ylabel("Regret")
    plt.title("Sine and Cosine functions")
    plt.show()

Bandit(4, 4, 2, 100, 1000)