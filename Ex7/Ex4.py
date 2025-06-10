# Ex4.py
# Implementation of Schelling's Model

import numpy as np
import matplotlib.pyplot as plt


# Function given an agent ID checks which IDs are within range thus neighbors
# Returns list of neighbor IDs
def getNeighborIDs(agentID, allAgents):
    # Calculate regular euclidean distance between points
    dists = np.linalg.norm(allAgents[:,0:2] - allAgents[agentID,0:2], axis=1) # Slicing 0:2 gives columns 0 and 1
    return np.where((dists < NeighborhoodRadiusR2) & (dists > 0))[0]

# Function to determine if an agent will move or not because it is unhappy with its neighborhood
def willMove(agentID,allAgents):
    neighborIDs = getNeighborIDs(agentID, allAgents)

    # If no neighbors will move
    if len(neighborIDs) == 0:
        return True

    # Check if neighbor percentage of same type OK
    return (sum(allAgents[neighborIDs,2]==allAgents[agentID,2])/neighborIDs.size) <= NeighborhoodMoveProbability


# Function to move agent to a new random position
def moveAgent(agentID,allAgents):
    allAgents[agentID, 0] = np.random.uniform(low=PlaneSize[0], high=PlaneSize[1]) # Random X
    allAgents[agentID, 1] = np.random.uniform(low=PlaneSize[0], high=PlaneSize[1]) # Random Y
    return allAgents


# Function to get n colors for plotting
def getIndexableColors(n, cmap_name='tab10'):
    # Use a colormap and resample it to get n evenly spaced colors
    cmap = plt.colormaps.get_cmap(cmap_name).resampled(n)
    return [cmap(i) for i in range(n)]



# Model parameters
np.random.seed(int(12345)) # Random seed if reproducibility is needed
NumAgents = 300
UpdateAgentsPerTimestep = 10
T_end = 300 # Simulate 300 Steps
NeighborhoodRadiusR2 = np.sqrt(0.1)
NeighborhoodMoveProbability = 0.5 # Move if neighbors of same type <= 50%
PlaneSize = [0,1] # Size of the plane
Types = ['earthling', 'martian'] # Define two types for the agents
Colors = getIndexableColors(len(Types))


# Initialize agents: Random X, Y positions in plane, random type assignment and colors as well
Agents = np.empty((NumAgents, 3)) # X, Y, Type, Color
# Assign agent positions randomly
Agents[:, 0] = np.random.uniform(low=PlaneSize[0], high=PlaneSize[1], size=NumAgents) # Random X
Agents[:, 1] = np.random.uniform(low=PlaneSize[0], high=PlaneSize[1], size=NumAgents) # Random Y
Agents[:, 2] = np.random.choice(np.arange(len(Types)), NumAgents) # Random type, assigned as integers from [0,size(Types)) to easily be used as indexes



# Simulation loop
for t in range(T_end):
    # Get random agents to update
    candidates = np.random.choice(NumAgents, UpdateAgentsPerTimestep, replace=False)

    # For each candidate check if they will move or not
    willMoveArr = np.empty(UpdateAgentsPerTimestep, dtype=bool)
    for i in range(UpdateAgentsPerTimestep):
         willMoveArr[i] = willMove(candidates[i],Agents)

    # If any agents will move, update all of them in sequence
    if np.any(willMoveArr):
        for agentID in candidates[willMoveArr]:
            Agents = moveAgent(agentID,Agents)


# Plot the model at the current time point
plt.figure(figsize=(6, 6))
for t in np.arange(len(Types)):
    mask = (Agents[:,2] == t)
    plt.scatter(Agents[mask][:, 0], Agents[mask][:, 1],
                c=Colors[t], label=Types[t], alpha=0.6)
plt.title("Schelling's Segregation Model")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()




