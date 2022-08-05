from calendar import c
from unittest.util import _MAX_LENGTH
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from Banditlib import *


def Bandit(AgentNum, ArmNum, dim, trialNum, maxTime):
    noiseStddev = 0.3
    
    maxLength =5000000

    trialIndividualMeanRegret = np.zeros(maxLength)

    for trial in np.arange(1,trialNum):
        
        individualRegret = np.zeros((AgentNum, 5000000))
        meanIndividualRegret = np.zeros(5000000)

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
            
            #sanitychecks = np.partition(pullsMatrix2.T, -(dim+1))[:,-(dim+1):].T
            #sanitychecks2 = (np.sort(np.where(topDindices==agentIndex, -1, sanitychecks).T).T)[1:,]
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
                  
            
            CFucbVector2 = CFmeansVector2+CFwidthVector2 
            ucbVector2 = np.minimum(CFucbVector2, oucbVector2)
            
            # When arm is pulled 
            armIndex2 = np.argmax(ucbVector2) # arm chosen
            rewardObserved2 = rewardsMatrix[agentIndex][armIndex2]+np.random.normal(0,noiseStddev) #arm pulled and reward observed
            pullOfChosen2 = storageProb2[agentIndex, armIndex2, 0] # total pulls of agent, arm pair up to now
            meanOfChosen2 = storageProb2[agentIndex, armIndex2, 1] # mean of agent, arm pair up to now
            storageProb2[agentIndex, armIndex2, 1] = (pullOfChosen2*meanOfChosen2+rewardObserved2)/(pullOfChosen2+1) # mean update
            storageProb2[agentIndex, armIndex2, 0] = storageProb2[agentIndex, armIndex2, 0]+1 # pull number update
            agentArrivalCount = np.sum(storageProb2[agentIndex,:,0])
            #globalRegret2[arrivalCount] = (globalRegret2[arrivalCount]*(trial-1) + gapMatrix[agentIndex, armIndex2])/trial
            ### mean over trials
            individualRegret[agentIndex, int(agentArrivalCount)]=(individualRegret[agentIndex, int(agentArrivalCount)]*(trial-1)+gapMatrix[agentIndex, armIndex2])/trial
    
    
        #Finish of all arrivals
        for AgId in range(AgentNum):
            indMaxlength = np.max(np.nonzero(individualRegret))
            length = min(len(meanIndividualRegret), indMaxlength)
            meanIndividualRegret = (meanIndividualRegret[:length]*AgId + individualRegret[AgId,:length])/(AgId+1) 
        #Finish of a iteration
        length2 = min(len(meanIndividualRegret), len(trialIndividualMeanRegret))
        trialIndividualMeanRegret = (trialIndividualMeanRegret[:length2] *(trial-1) + meanIndividualRegret[:length2])/(trial)
    #Finish of all iterations 
        
    return(trialIndividualMeanRegret)         
        
        
        ######### Finish of one trial ###
    
    ##### After all trials

Banditresult1 = np.cumsum(Bandit(16, 4, 2, 10, 500000))
Banditresult2 = np.cumsum(Bandit(32, 4, 2, 10, 500000))
Banditresult3 = np.cumsum(Bandit(64, 4, 2, 10, 500000))

maxLength = min(len(Banditresult1), len(Banditresult2), len(Banditresult3))

Xglobal=np.arange(maxLength)

Y1 = Banditresult1[0:maxLength]
Y2 = Banditresult2[0:maxLength]
Y3 = Banditresult3[0:maxLength]

plt.plot(Xglobal, Y1, 'o-', color='r', label='#Agent = 16')
plt.plot(Xglobal, Y2, '^-', color='g', label='#Agent = 32')
plt.plot(Xglobal, Y3, 's-', color='b', label='#Agent = 64')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Individual arrivals")
plt.ylabel("Individual regret")
plt.title("Individual regret")
plt.show()
