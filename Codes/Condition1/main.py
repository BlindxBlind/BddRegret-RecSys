from pickle import FALSE
import numpy as np

def main():
    print("test")
if __name__ == "__main__":
    main()

def Cond1Func():
    numTrials = 10000
    armNumber=0
    resultMatrix1 = np.zeros(4, 100)
    resultMatrix2 = np.zeros(4, 100)
    while armNumber<=1000:
        armNumber = armNumber + 10 # increase M from 10 to 1000
        agentNumber = 0
        FLAG = True # We want to stop increasing agent when already the probability is larger than 99.99
        for dim in [5, 10, 25, 50]:
            while (agentNumber <= 1000*dim*5) and FLAG==True:
                agentNumber = agentNumber + dim/5 # increase A                
                rng = np.ramdom.default.rng()
                expMatrix = rng.multinomial(agentNumber, [float(1/armNumber)]*armNumber, size=numTrials)
                comparedMatrix = expMatrix<dim+1 # Make it True/False array of whether if an element is bad
                comparedvector = np.sum(comparedMatrix, axis = 1) # Check each experiment violates condition 1 or not
                count = np.count_nonzero() # Count the numbr of violations among all expriments
                rate = count/numTrials
                if (dim == 100) and rate >= 99.99:
                    FLAG=False
