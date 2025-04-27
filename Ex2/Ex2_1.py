# Complex Systems In Bioinformatics, SoSe2025
# Exercise 02, Task 1b)
#
# Submitted by:
# Kristian Reinhart, 4474140
# Duong Ha Le Minh: 5314209
#
# Submitted to:
# provided to you by the best bioinformatics lecturer you'll ever have (starts with M, ends with t)

import numpy as np
import os # For changing working directory path


# Change working directory to path where script and input file is
#os.chdir('ADD_YOUR_FILEPATH')

# A: -----Fixed Quantities-----
#0. initial state
X0 = np.loadtxt('Input.txt')

# ===> fill here, everywhere where a "..." is <===

#1. Stoichiometric matrix
# np.array is apparently filled out row-wise.
# Saving S gives the same result as referene SMatrix.txt, so should be fine
S = np.array([[-1,0,0],[-1,1,1],[0,0,-1]]);# !!check dimension of the array!!

#2. reaction parameters
k = [0.01, 0.1, 0.01];


# B: functions that depend on the state of the system X
def ReactionRates(k,X):
        R = np.zeros((3,1))
        R[0] = k[0] * X[0] * X[1]
        R[1] = k[1]
        R[2] = k[2] * X[2] * X[1]
        return R
# ===>       -----------------------     <===

# compute reaction propensities/rates
R = ReactionRates(k,X0)

#compute the value of the ode with time step delta_t = 1
dX = np.dot(S,R)

##a) save stoichiometric Matrix
np.savetxt('SMatrix.txt',S,delimiter = ',',fmt='%1.0f');
##b) save ODE value as float with 2 digits after the comma (determined with the c-style precision argument e.g. '%3.2f')
np.savetxt('ODEValue.txt',dX,delimiter=',',fmt='%1.2f');

