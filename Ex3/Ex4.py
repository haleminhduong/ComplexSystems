# Complex Systems In Bioinformatics, SoSe2025
# Exercise 04, Homework 4a)
#
# Submitted by:
# Kristian Reinhart, 4474140
# Duong Ha Le Minh: 5314209
#
# Submitted to:
# provided to you by the best bioinformatics lecturer you'll ever have (starts with M, ends with t)

####

import numpy as np
# import matplotlib.pyplot as pl
# import os # For getting and setting working directory via os.getcwd() and os.chdir()

# import the input file
try:
    In = np.loadtxt('Input.txt', ndmin=2)  # initial state
except FileNotFoundError:
    print("Error: Input.txt not found.")


np.random.seed(seed=int(In[0]))
NrSimulations = int(In[1])

# -----Fixed Quantities-----
# Stoichiometric matrix

# <------- fill in the model specifics ---->

#              r0   r0
S = np.array([[-1,  0],  # X0
              [ 1, -1]], dtype=int)  # X1

# Initial system status at t(0)
X0 = np.array([[100.0] , [0.0]]) # TODO: Homework 2 and 4 only define x1(t0)=0.0, not x0(t0).

# Reaction parameters
k = [0.5, 0.3]  # ka, ke

# Final time for simulation
t_final = 24.0

# <------------------------------>

# <------- fill in the reaction rate functions ---->

# Get the upper bounds for reactions at timepoint t
# Upperbound calculation for R[1]
# max(sin(x)) = 1 for x = t * 180, t in [0, inf)
# 0.5(sin(t * 180) + 2) = 0.5(1 + 2) = 1.5
# So upper bound will be X[1] * k[1] * 1.5 (OR: x1(t) * ke * 1.5 )
def bounds(X, k):
    b = np.zeros((2, 1))
    b[0] = k[0] * X[0,0] # ka * x0(t)
    b[1] = k[1] * X[1,0] * 1.5 # ke * x1(t) *1.5
    return b

# reaction propensities / reaction rates
def propensities(X, k, t):
    R = np.zeros((2, 1))
    R[0] = X[0,0] * k[0] # ka * x0
    R[1] = X[1,0] * k[1] * 0.5 * (np.sin(np.radians(t*180)) + 2) # ke * x1 in Homework 2. Homework 4 modified to x1(t) * ke * 0.5 * (sin(t*180) + 2)
    return R
# <------------------------------>


# For Extrande use upper bound B instead of lambda, B being equal or larger than the sum of the reaction rates
def Time_To_Next_Reaction(B):
    """
    @brief The function samples from an exponential distribution with rate "B".
    @param B : real value positive.
    """

    # small hack as the numpy uniform random number includes 0
    r = np.random.rand()
    while r == 0:
        r = np.random.rand()

    return (1.0/B)*np.log(1.0/r)


def Find_Reaction_Index(a,b):
    """
    @brief The function takes in the reaction rate vector and returns
    the index of the reaction to be fired of a possible reaction candidate.
    @param a : Array (num_reaction,1)
    @param b : Sum of upper bounds B

    """

    # small hack as the numpy uniform random number includes 0
    r = np.random.rand()
    while r == 0:
        r = np.random.rand()

    #return np.sum(np.cumsum(a) < r*np.sum(a))
    # Interval to choose from must include probability for rejection
    return np.sum((np.cumsum(a)+b) < r*(np.sum(a)+b)) # TODO: Probably implemented this wrongly.


def Extrande(Stochiometry, X0, t_final, k):
    """
    @brief  Extrande Algorithm. Given the stochiometry,
    propensities, the initial state and bounds for the reactions; the algorithm
    gives a stochastic trajectory until $t_final.$

    @param Stochiometry : Numpy Array (Num_species,Num_reaction).
    @param X_0: Numpy Array (Num_species, 1).
    @param t_final : positive number.
    @param ka,ke: positive numbers  (reaction rate parameters)
    @param bound: Numpy Array (Num_reactions, 1)

    """

    # For storage
    X_store = []
    T_store = []

    # Initialize
    t = 0.0
    x = X0.copy()
    X_store.append(x[1,0])
    T_store.append(t)

    while t < t_final:
        # Compute upper bounds for all reactions
        upperbounds = bounds(x, k)
        B = np.sum(upperbounds) # Get sum of all upper bounds of all rections for boundary B

        # 1. When? Compute first Jump Time
        tau = Time_To_Next_Reaction(B)

        # compte reaction rate functions
        a = propensities(x, k, t)

        """ Stopping criterium: Test if we have jumped too far and if
		yes, return the stored variables (states, times)
		"""
        if (t + tau > t_final) or (np.sum(a) == 0):
            return np.array(X_store), np.array(T_store)
        else:
            # Since we have not, we need to find the next reaction
            t = t + tau  # update time
            # 2. What? find reaction to execute and execute the reaction
            j = Find_Reaction_Index(a,B)
            # Check if reaction was rejected
            if(j >= size(k) ):
                # Chosen reaction to fire is outside of number of reactions -> Rejected
                # Update our storage, keeping system as is
                X_store.append(x[1,0])
                T_store.append(t)
            else:
                # Actual reaction fires, update system
                x = x + Stochiometry[:, [j]]
                # Update our storage
                X_store.append(x[1,0])
                T_store.append(t)


# Run a number of simulations and save the respective trajectories
for i in range(NrSimulations):
    # get a single realisation
    states, times = Extrande(S, X0, t_final, k)
    # a) save trajectory
    Output = np.concatenate(
        (np.array(times, ndmin=2), np.array(states, ndmin=2)), axis=0)
    np.savetxt('Task4Traj'+str(i+1)+'_TEST.txt', Output, delimiter=',', fmt='%1.2f') # TODO: '_TEST' entfernen
    print('Task4Traj ' + str(i+1) + ' filled.')
