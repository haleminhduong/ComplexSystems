# Complex Systems In Bioinformatics, SoSe2025
# Exercise 04, Homework 3a)
#
# Submitted by:
# Kristian Reinhart, 4474140
# Duong Ha Le Minh: 5314209


# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
# import os # For getting and setting working directory via os.getcwd() and os.chdir()


# import the input file
try:
    In = np.loadtxt('Input.txt', ndmin=2)  # initial state
except FileNotFoundError:
    print("Error: Input.txt not found.")

np.random.seed(seed=int(In[0]))
NrSimulations = int(In[1])  # N

# -----Fixed Quantities-----
X0 = np.array([[100], [0]], dtype=int)
# Stoichiometric matrix

# <------- fill in the model specifics ---->

S = np.array([[-1,  0],
              [1, -1]],
             dtype=int)

# reaction parameters

k = [0.5, 0.3]  # k1, k2, k3 from the first task

t_final = 24.0  # final time T = 24

dt_store = 1.0

# <------------------------------>

# <------- fill in the reaction rate functions ---->
# reaction propensities


def propensities(X, k):
    R = np.zeros((2, 1))
    R[0] = X[0]*k[0]
    R[1] = X[1]*k[1]
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


def SSA(Stochiometry, X0, t_final, k):
    """
    @brief  The Stochastic Simulation Algorithm. Given the stochiometry,
    propensities and the initial state; the algorithm
    gives a stochastic trajectory of the Kurtz process until $t_final.$

    @param Stochiometry : Numpy Array (Num_species,Num_reaction).
    @param X_0: Numpy Array (Num_species, 1).
    @param t_final : positive number.
    @param k1,k2,k3,k4: positive numbers  (reaction rate parameters)

    """

    # for storage
    X0_store = []
    X1_store = []
    T_store = []
    # initialize
    t = 0.0
    x = X0.copy()
    X0_store.append(x[0, 0])
    X1_store.append(x[1, 0])
    T_store.append(t)

    while t < t_final:
        # compte reaction rate functions
        a = propensities(x, k)
        # 1. When? Compute first Jump Time
        tau = Time_To_Next_Reaction(np.sum(a))

        """ Stopping criterium: Test if we have jumped too far and if
		yes, return the stored variables (states, times)
		"""
        if (t + tau > t_final) or (np.sum(a) == 0):
            return np.array(X0_store), np.array(X1_store), np.array(T_store)
        else:
            # Since we have not, we need to find the next reaction
            t = t + tau  # update time
            # 2. What? find reaction to execute and execute the reaction
            j = Find_Reaction_Index(a)
            x = x + Stochiometry[:, [j]]
            # Update Our Storage
            X0_store.append(x[0, 0])
            X1_store.append(x[1, 0])
            T_store.append(t)

# ----- Post-Processing for Fixed Time Points (Adapted Version) -----


def get_states_at_fixed_times(t_store, x0_store, x1_store, output_times):
    """
    @brief Interpolates (finds last state before) the state vector at specified output_times,
    using separate storage arrays for x0 and x1.


    @param t_store: array of reaction times from SSA.
    @param x0_store: array of x0 states corresponding to reaction times.
    @param x1_store: array of x1 states corresponding to reaction times.
    @param output_times: array of times at which to get the state.
    """

    output_states = np.zeros((len(output_times), 2))

    for i, t_out in enumerate(output_times):
        idx = np.searchsorted(t_store, t_out, side='right') - 1
        # idx >= 0
        idx = max(0, idx)

        output_states[i, 0] = x0_store[idx]
        output_states[i, 1] = x1_store[idx]

    return output_states


all_x1 = []
output_times = []

# Run a number of simulations and save the respective trajectories
for i in range(NrSimulations):
    # get a single realisation
    X0_states, X1_states, times = SSA(S, X0, t_final, k)

    output_times = np.arange(0, t_final + dt_store, dt_store)
    output_times = output_times[output_times <=
                                t_final]

    output_states_at_times = get_states_at_fixed_times(
        times, X0_states, X1_states, output_times)

    all_x1.append(output_states_at_times[:, 1])

    # a) save trajectory
    Output = np.concatenate(
        (np.array(output_times, ndmin=2), np.array(output_states_at_times[:, 0], ndmin=2), np.array(output_states_at_times[:, 1], ndmin=2)), axis=0)
    np.savetxt('Task3aTraj'+str(i+1)+'Timed.txt',
               Output, delimiter=',', fmt='%1.2f')
    print('Task3aTraj ' + str(i+1) + ' filled.')


## Plotting the sample means and saving the file as 'Exercise3Homework3b.png'
#
## Placeholder for plotting. Substitute with actual sample times
#timeT = output_times
## Placeholder for plotting. Substitute with actual sample means
#sampleMeanX1 = np.mean(all_x1, axis=0)
## Placeholder for plotting. Substitute with actual sample mean standard deviations
#sampleMeanX1Deviation = np.std(all_x1, axis=0)
#
## Create the plot
#plt.figure(figsize=(8, 4))  # figsize=(8, 4), 8 inches wide, 4 inches tall
## Solid black line
#plt.plot(timeT, sampleMeanX1, 'k-', label='sample mean')
#plt.plot(timeT, sampleMeanX1 + sampleMeanX1Deviation, 'r--',
#         label='sample mean \pm stdev')       # Dashed red line
#plt.plot(timeT, sampleMeanX1 - sampleMeanX1Deviation, 'r--')
#
#plt.xlabel('time')
#plt.ylabel('X1')
#plt.legend()
#plt.tight_layout()
#plt.savefig("Exercise3Homework3b.png")
#plt.show()
