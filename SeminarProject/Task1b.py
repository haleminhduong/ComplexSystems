# -*- coding: utf-8 -*-
"""
Complex Systems in Bioinformatics
Seminar, Block 3, Project 2
Homework 1b)
@authors:   Duong Ha Le Minh, 5314209
            Kristian Reinhart, 4474140
"""


import numpy as np
import matplotlib.pyplot as plt

# Function to perform actual move while checking boundaries. Use clipped boundaries.
def boundMove(a,b):
    global PlaneSize # Access global variable
    if( a+b < PlaneSize[0]):
        return 0
    elif( a+b > PlaneSize[1]):
        return PlaneSize[1]
    else:
        return a+b

# Function to move agent to a new random position
def moveAgents(agentArr,sigma):
    if(len(agentArr)>0):
        for i in range(len(agentArr)):
            xMove = np.random.normal(0,sigma)
            yMove = np.random.normal(0,sigma)
            agentArr[i,0] = boundMove(agentArr[i,0],xMove) # Update X coordinate
            agentArr[i,1] = boundMove(agentArr[i,1],yMove) # Update Y coordinate
    return agentArr


# Function to check which prey gets eaten and if predators propagate
def eatPreyAndPropagate(prey,predators,r,p):
    global PlaneSize # Global variable for new predator locations
    # See if any predators left
    if(len(predators)>0):
        # Counter for how many new predators will be born this round
        predatorPropagationCounter = 0
        # Go through all predators
        for i in range(len(predators)):
            # Check if any prey left, if not, break loop
            if(len(prey)==0):
                break
            # If prey left, find all prey in range of current predator
            # Formula for range in task is strange, just use circular distance
            preyInRange = np.where(np.sqrt((predators[i,0]-prey[:,0])**2 + (predators[i,1]-prey[:,1])**2) < r) # Using radius as a distance measure
            #preyInRange = np.where( (np.abs(predators[i,0]-prey[:,0]) + np.abs(predators[i,1]-prey[:,1]))<r ) # Using euclidean distance
            # See if any prey was in range
            if(len(preyInRange)>0):
                # Eat the prey
                prey = np.delete(prey, preyInRange, axis=0)
                # Roll for new predator births
                predatorPropagationCounter += sum(np.random.uniform(low=0, high=1, size=len(preyInRange))>=p)
        # All predators checked, add newborn predators if any
        if (predatorPropagationCounter>0):
            newPred = np.empty((predatorPropagationCounter, 2)) # X, Y, Type, Color
            newPred[:, 0] = np.random.uniform(low=PlaneSize[0], high=PlaneSize[1], size=predatorPropagationCounter) # Random X
            newPred[:, 1] = np.random.uniform(low=PlaneSize[0], high=PlaneSize[1], size=predatorPropagationCounter) # Random Y

            # Add new predators to current list
            predators = np.concatenate((predators, newPred), axis=0)
    return prey,predators


# Initial model parameters
#np.random.seed(int(12345)) # Random seed if reproducibility is needed
PlaneSize = [0,100]
numPrey_z1 = 100
movePrey_z1_sigma = 10 # Standard deviation for the zero-centered gaussian distribution for prey movement

numPredator_z2 = 20
movePredator_z2_sigma = 10 # Standard deviation for the zero-centered gaussian distribution for predator movement
predatorRadius = 20  # Interaction radius (square, not round:  (z_2(i)_{x,∗} − z_1(j)_{x,∗})^2 + ((z_2(i)_{∗,y} − z_1(j)_{∗,y})^2 < 20 )
predatorPropagationProb = 0.5 # Probability to propagate
T_end = 100


# Initialize predator and prey: Random X, Y positions in plane
preyZ1 = np.empty((numPrey_z1, 2)) # X, Y, Type, Color
preyZ1[:, 0] = np.random.uniform(low=PlaneSize[0], high=PlaneSize[1], size=numPrey_z1) # Random X
preyZ1[:, 1] = np.random.uniform(low=PlaneSize[0], high=PlaneSize[1], size=numPrey_z1) # Random Y

predatorZ2 = np.empty((numPredator_z2, 2)) # X, Y, Type, Color
predatorZ2[:, 0] = np.random.uniform(low=PlaneSize[0], high=PlaneSize[1], size=numPredator_z2) # Random X
predatorZ2[:, 1] = np.random.uniform(low=PlaneSize[0], high=PlaneSize[1], size=numPredator_z2) # Random Y


# Simulation loop
for t in range(T_end):
    # Update positions of predator and prey
    preyZ1 = moveAgents(preyZ1,movePrey_z1_sigma)
    predatorZ2 = moveAgents(predatorZ2,movePredator_z2_sigma)

    # For each predator, check if prey is in range; eat and propagate
    preyZ1,predatorZ2 = eatPreyAndPropagate(preyZ1,predatorZ2,predatorRadius,predatorPropagationProb)



# TODO: Plot / animate this stuff
# Plot the model at the current time point
# plt.figure(figsize=(6, 6))
# for t in np.arange(len(Types)):
#     mask = (Agents[:,2] == t)
#     plt.scatter(Agents[mask][:, 0], Agents[mask][:, 1],
#                 c=Colors[t], label=Types[t], alpha=0.6)
# plt.title("Schelling's Segregation Model")
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.show()

