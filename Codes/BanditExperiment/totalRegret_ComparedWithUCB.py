from calendar import c
from Banditlib import *
import numpy as np
import matplotlib.pyplot as plt


def Bandit(AgentNum, ArmNum, dim, trialNum, maxTime):
    noiseStddev = 0.1
    globalRegret1 = np.zeros(50000000) # UCB's global mean of Regret at each time - anyarrival
    globalRegret2 = np.zeros(50000000) # CFUCB's global mean of Regret at each time - anyarrival


    maxLength =100000000
    for trial in np.arange(1,trialNum):
        storageProb1 = np.zeros((AgentNum, ArmNum, 2)) # the (agentNum x armNum) matrix of (#pullings, empirical mean) for UCB algorithm
        storageProb2 = np.zeros((AgentNum, ArmNum, 2)) # the (agentNum x armNum) matrix of (#pullings, empirical mean) for CFUCB algorithm
        NmminMatrix = np.zeros((AgentNum, ArmNum)) 
            #Arrival Generation
        arrivals = ArrivalGen(AgentNum, maxTime, type='Normal') # long array of [index, arrival time]
        maxLength = np.minimum(maxLength, len(arrivals))
            #Agent feature set gen
        agentFeatureSet = AgentGen(AgentNum, dim)
            #Arm feature set gen
        armFeatureSet = ArmGen(ArmNum, dim)

        while not Condition1Checker(agentFeatureSet, armFeatureSet, dim):
            agentFeatureSet = AgentGen(AgentNum, dim)
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

            pullsMatrix2 = storageProb2[:, :, 0] ## in this case we consider full pulls of all agents and arms
            
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
        
                

            CFmeansVector2 = np.zeros(ArmNum)
            CFwidthVector2 = np.zeros(ArmNum)
           
            #Find CFmean, CFwidth for each arm for the arrived agent
            for armIndex in range(ArmNum):
                topDAgentIndices = componentIndices.T[armIndex] #best agent indices for the arm
                
                pullsofBestAgents = storageProb2[:, armIndex, 0][topDAgentIndices] #pulls of the best d agents for the arm for the agent
                

                Nmmin_dtj = np.amin(pullsofBestAgents)
                NmminMatrix[agentIndex, armIndex] = np.amin(pullsofBestAgents)

                
                

                                
                topDAgents = agentFeatureSet[componentIndices.T[armIndex]] #collected top d sorted agent indices for the arm for the arrived agent
                coeffs = np.linalg.solve(topDAgents.T, agentFeatureSet[agentIndex])
                coeffAbsSum=np.sum(np.absolute(coeffs))
                match = np.vstack((componentIndices.T[armIndex], coeffs)).T # an array of (agent index, coefficient) 
                
              
                CFmeansVector2[armIndex]= 0
                sanity= 0
                for temp in match:
                    CFmeansVector2[armIndex] = CFmeansVector2[armIndex]+temp[1]*storageProb2[int(temp[0]), armIndex, 1]
                    sanity = sanity +temp[1]*rewardsMatrix[int(temp[0])][armIndex]
                
                
                if Nmmin_dtj == 0 or totalPull2 == 0:
                    CFwidthVector2[armIndex] = 10000
                else:
                    #CFwidthVector2[armIndex] = np.sqrt(np.log(totalPull2/dim)/(NmminMatrix[agentIndex, armIndex]/coeffAbsSum**2))
                    CFwidthVector2[armIndex] = np.sqrt(np.log(totalPull2)/(NmminMatrix[agentIndex, armIndex]))
                    #The above equation, commented out, is what the theory uses for bounding regret.
                    #The equation below, which is used in this code, is much more stable and used in this simulation.

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
    

        #after last arrival
        print(trial)        
        
        
        ######### Finish of one trial ###
    
    ##### After all trials
    plotPeriod = int(maxLength/50)
    globalRegret1 = (globalRegret1[0:maxLength])
    globalRegret2 = (globalRegret2[0:maxLength])
    Xglobal=np.arange(len(globalRegret1))
    globalCumRegret1 = np.cumsum(globalRegret1)
    globalCumRegret2 = np.cumsum(globalRegret2)
    Y1 = globalCumRegret1
    Y2 = globalCumRegret2

    plt.plot(Xglobal[::plotPeriod], Y1[::plotPeriod], 'o-', markersize=3.5, color='r', linewidth = 0.25, markerfacecolor = 'None', label='UCB')
    plt.plot(Xglobal[::plotPeriod], Y2[::plotPeriod], '^-',markersize=3.5 , color='g', linewidth = 0.25, markerfacecolor = 'None',label='CFUCB')
    plt.legend(loc="upper left")
        # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Total arrivals of agents")
    plt.ylabel("Total Regret")
    plt.grid(axis="x", linestyle = 'dashed')
    plt.title("Comparison with CFUCB and UCB")
    plt.savefig("RegretComparison.png")
    plt.show()

Bandit(32, 4, 2, 5, 30000)
# The probability that the Condition 1 of the paper does not hold for 32 agents, 4 arms, dimension=2 is quite high. 
# If Condition 1 does not hold, you won't observe flat regret of CFUCB. This is because some agent's regret will be O(logT).
# We excluded that possibility by using CondOneChecker function.