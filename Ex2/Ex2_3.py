# Complex Systems In Bioinformatics, SoSe2025
# Exercise 02, Task 3a)
#
# Submitted by:
# Kristian Reinhart, 4474140
# Duong Ha Le Minh: 5314209
#
# Submitted to:
# provided to you by the best bioinformatics lecturer you'll ever have (starts with M, ends with t)

####
# Trajectories for the SIR model in Exercise 3
####

import numpy as np
# import matplotlib.pyplot as pl

# import the input file
try:
    X0 = np.loadtxt('Input.txt', ndmin=2)  # initial state
except FileNotFoundError:
    print("Error: Input.txt not found.")
try:
    In = np.loadtxt('Input2.txt')  # seed and number of simulations
except FileNotFoundError:
    print("Error: Input.txt not found.")
    # set the seed and the number of simulations from the input file

np.random.seed(seed=int(In[0]))
NrSimulations = int(In[1])

# -----Fixed Quantities-----
# Stoichiometric matrix

# <------- fill in the model specifics ---->

#               r1   r2  r3
S = np.array([[-1,  0,  0],  # X1
              [-1,  1,  1],  # X2
              [0,  0, -1]], dtype=int)  # X3

# reaction parameters

k = [0.01, 0.1, 0.01]  # k1, k2, k3 from the first task

t_final = 10.0  # final time T = 10

# Upperbound calculation
# max(sin(x)) = 1 for x = t * 180, t in [0, inf)
# 0.5(sin(t * 180) + 2) = 0.5(1 + 2) = 1.5

bound = k[1] * 1.5

# <------- fill in the reaction rate functions ---->
# reaction propensities


def propensities(X, k, t):
    R = np.zeros((3, 1))
    R[0] = X[0]*X[1]*k[0]
    R[1] = k[1] * 0.5 * (np.sin(np.radians(t*180)) + 2)
    R[2] = X[1]*X[2]*k[2]
    return R


def propensities_upper(X, k, B):
    R = np.zeros((3, 1))
    R[0] = X[0]*X[1]*k[0]
    R[1] = B
    R[2] = X[1]*X[2]*k[2]
    return R
# <------------------------------>


def Time_To_Next_Reaction(lam):
    """
    @brief The function samples from an exponential distribution with rate "lam".
    @param lam : real value positive.
    """

    # small hack as the numpy uniform random number includes 0
    r = np.random.rand()
    while r == 0:
        r = np.random.rand()

    return (1.0/lam)*np.log(1.0/r)


def Find_Reaction_Index(a):
    """
    @brief The function takes in the reaction rate vector and returns
    the index of the reaction to be fired of a possible reaction candidate.
    @param a : Array (num_reaction,1)

    """
    # small hack as the numpy uniform random number includes 0
    r = np.random.rand()
    while r == 0:
        r = np.random.rand()

    return np.sum(np.cumsum(a) < r*np.sum(a))


def Extrande(Stochiometry, X0, t_final, k, bound):

    # for storage
    X_store = []
    T_store = []
    # initialize
    t = 0.0
    x = X0.copy()
    B = bound
    X_store.append(x[1, 0])
    T_store.append(t)

    while t < t_final:
        # compute upperbound
        a_upper = propensities_upper(x, k, B)
        a_upper_sum = np.sum(a_upper)

        # 1. When? Compute first Jump Time
        tau = Time_To_Next_Reaction(a_upper_sum)

        """ Stopping criterium: Test if we have jumped too far and if
		yes, return the stored variables (states, times)
		"""
        if (t + tau > t_final) or (a_upper_sum <= 0):
            return np.array(X_store), np.array(T_store)

        # Since we have not, we need to find the next reaction
        t = t + tau  # update time
        # 2. What? find reaction to execute and execute the reaction
        a_actual = propensities(x, k, t)
        a_actual_sum = np.sum(a_actual)

        u2 = np.random.rand()
        while u2 == 0:
            u2 = np.random.rand()

        accepted = False
        if u2 * a_upper_sum <= a_actual_sum:
            # Check if any actual reaction can occur
            if a_actual_sum > 0:
                accepted = True

        if accepted:
            j = Find_Reaction_Index(a_actual)
            x = x + Stochiometry[:, [j]]
            # Store results
            X_store.append(x[1, 0])
            T_store.append(t)


# Run a number of simulations and save the respective trajectories
for i in range(NrSimulations):
    # get a single realisation
    states, times = Extrande(S, X0, t_final, k, bound)
    # a) save trajectory
    Output = np.concatenate(
        (np.array(times, ndmin=2), np.array(states, ndmin=2)), axis=0)
    np.savetxt('Task3Traj'+str(i+1)+'.txt', Output, delimiter=',', fmt='%1.3f')
