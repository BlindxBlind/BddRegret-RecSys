from calendar import c
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
    globalRegret1 = np.zeros(50000000) # UCB's global mean of Regret at each time - anyarrival
    globalRegret2 = np.zeros(50000000) # CFUCB's global mean of Regret at each time - anyarrival
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
        armFeatureSet = ArmGen(ArmNum, dim)
            #Gap matrix gen
        rewardsMatrix = np.matmul(agentFeatureSet, armFeatureSet.T)
        gapMatrix = gapMatrixGen(agentFeatureSet, armFeatureSet)
            #Now the bandit starts
        
        ############################################
        
        arrivalCount = 0
        for arrival in arrivals:
            arrivalCount = arrivalCount + 1
            agentIndex = int(arrival[0])   

            # UCB algorithm part

            pullsVector1 = storageProb1[agentIndex, :, 0].astype(int)
            totalPull1 = np.sum(pullsVector1)
            meansVector1 = storageProb1[agentIndex, :, 1]
            widthVector1 = pullsVector1
            widthVector1 = np.where((widthVector1 == 0), 100000, np.sqrt(np.log(totalPull1)/pullsVector1))
            ucbVector1 = meansVector1 + widthVector1 #ucb of all arms

            armIndex1 = np.argmax(ucbVector1)
            rewardObserved = rewardsMatrix[agentIndex][armIndex1]+np.random.normal(0,noiseStddev) #reward observed
            pullOfChosen = storageProb1[agentIndex, armIndex1, 0]
            meanOfChosen = storageProb1[agentIndex, armIndex1, 1]
            storageProb1[agentIndex, armIndex1, 1] = (pullOfChosen*meanOfChosen+rewardObserved)/(pullOfChosen+1) # mean update
            storageProb1[agentIndex, armIndex1, 0] = storageProb1[agentIndex, armIndex1, 0]+1 # pull number update
            globalRegret1[arrivalCount] = (globalRegret1[arrivalCount]*(trial-1) + gapMatrix[agentIndex, armIndex1])/trial
            

            ######CFUCB algorithm part######################

            ## oucb calculation

            pullsMatrix2 = storageProb2[:, :, 0].astype(int) ## in this case we consider full pulls of all agents and arms
            
            pullsVector2 = storageProb2[agentIndex, :, 0].astype(int)
            totalPull2 = np.sum(pullsVector2)
            meansVector2 = storageProb2[agentIndex, :, 1] ## arrived agent's means for each arms till now
            owidthVector2 = pullsVector2
            owidthVector2 = np.where((owidthVector2 == 0), 100000, np.sqrt(np.log(totalPull2)/pullsVector2))
            oucbVector2 = meansVector2 + owidthVector2
            
            ## conuterfactual ucb calculation
            
            topDindices = np.argpartition(pullsMatrix2.T, -(dim+1))[:,-(dim+1):].T # sorted top d+1 agent's indices for each arms
            componentIndices = (np.sort(np.where(topDindices==agentIndex, -1, topDindices).T).T)[1:,] # changed agentIndex to -1, sorted, and deleted - sorted Agent indices for each arm
            
            sanitychecks = np.partition(pullsMatrix2.T, -(dim+1))[:,-(dim+1):].T
            sanitychecks2 = (np.sort(np.where(topDindices==agentIndex, -1, sanitychecks).T).T)[1:,]
            #if trial ==1 and agentIndex==1:
            #    print(sanitychecks2)
            #    print(storageProb2[:,:,0])

            CFmeansVector2 = np.zeros(ArmNum)
            CFwidthVector2 = np.zeros(ArmNum)
           
            #Find CFmean, CFwidth for each arm for the arrived agent
            for armIndex in range(ArmNum):
                topDAgentIndices = componentIndices.T[armIndex] #best agent indices for the arm
                
                pullsofBestAgents = storageProb2[:, armIndex, 0][topDAgentIndices] #pulls of the best d agents for the arm for the agent
                

                Nmmin_dtj = np.amin(pullsofBestAgents)
                
                #if trial == 1 and armIndex == 1 and agentIndex ==1:
                #    print(pullsofBestAgents)
                #    print(Nmmin_dtj)
                
                topDAgents = agentFeatureSet[componentIndices.T[armIndex]] #collected top d sorted agent indices for the arm for the arrived agent
                coeffs = np.linalg.solve(topDAgents.T, agentFeatureSet[agentIndex])
                coeffAbsSum=np.sum(np.absolute(coeffs))
                match = np.vstack((componentIndices.T[armIndex], coeffs)).T # an array of (agent index, coefficient) 
                
                #check = np.matmul(topDAgents.T, coeffs)
                #check2 = agentFeatureSet[agentIndex]
                #print(check-check2)
                CFmeansVector2[armIndex]= 0
                sanity= 0
                for temp in match:
                    CFmeansVector2[armIndex] = CFmeansVector2[armIndex]+temp[1]*storageProb2[int(temp[0]), armIndex, 1]
                    sanity = sanity +temp[1]*rewardsMatrix[int(temp[0])][armIndex]
                
                
               
                if Nmmin_dtj == 0 or totalPull2 == 0:
                    CFwidthVector2[armIndex] = 10000
                else:
                    CFwidthVector2[armIndex] = np.sqrt(np.log(totalPull2/dim)/(Nmmin_dtj/coeffAbsSum**2))
                    #if CFwidthVector2[armIndex]>widthVector1[armIndex]:
                        #print(Nmmin_dtj,widthVector1[armIndex])
            
            CFucbVector2 = CFmeansVector2+CFwidthVector2 
            ucbVector2 = np.minimum(CFucbVector2, oucbVector2)
            
            # When arm is pulled 
            armIndex2 = np.argmax(ucbVector2) # arm chosen
            rewardObserved2 = rewardsMatrix[agentIndex][armIndex2]+np.random.normal(0,noiseStddev) #arm pulled and reward observed
            pullOfChosen2 = storageProb2[agentIndex, armIndex2, 0] # total pulls of agent, arm pair up to now
            meanOfChosen2 = storageProb2[agentIndex, armIndex2, 1] # mean of agent, arm pair up to now
            storageProb2[agentIndex, armIndex2, 1] = (pullOfChosen2*meanOfChosen2+rewardObserved2)/(pullOfChosen2+1) # mean update
            storageProb2[agentIndex, armIndex2, 0] = storageProb2[agentIndex, armIndex2, 0]+1 # pull number update
            globalRegret2[arrivalCount] = (globalRegret2[arrivalCount]*(trial-1) + gapMatrix[agentIndex, armIndex2])/trial
            #if trial == 1 and agentIndex == 1 and arrivalCount>30000:
                #print(CFwidthVector2, widthVector1)
                #print(rewardsMatrix[agentIndex],CFmeansVector2, meansVector1)
                #print(rewardsMatrix[agentIndex, armIndex] -sanity )
                #quit()
                #print(sanitychecks2)
                #print(arrivalCount, Nmmin_dtj, pullsVector2[armIndex])

        #after last arrival
        print(trial)        
        
        
        ######### Finish of one trial ###
    
    ##### After all trials

    globalRegret1 = (globalRegret1[0:maxLength])
    globalRegret2 = (globalRegret2[0:maxLength])
    Xglobal=np.arange(len(globalRegret1))
    globalCumRegret1 = np.cumsum(globalRegret1)
    globalCumRegret2 = np.cumsum(globalRegret2)
    Y1 = globalCumRegret1
    Y2 = globalCumRegret2

    plt.plot(Xglobal, Y1, color='r', label='UCB')
    plt.plot(Xglobal, Y2, color='g', label='CFUCB')
        # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Total arrivals")
    plt.ylabel("Regret")
    plt.title("Sine and Cosine functions")
    plt.show()

Bandit(32, 4, 2, 2, 2000000)