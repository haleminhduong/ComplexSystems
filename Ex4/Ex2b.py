# Complex Systems In Bioinformatics, SoSe2025
# Exercise 04, Homework 2b)
#
# Solving ODE with 4th order Runge-Kutta-scheme
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
    In = np.loadtxt('Input.txt', ndmin=2)  # Initial state of system, x0(t0), x1(t0)
except FileNotFoundError:
    print("Error: Input.txt not found.")

# -----Fixed Quantities-----
X0 = np.array([[In[0,0]], [In[1,0]]], dtype=int) # Store initial condition


# Reaction parameters

k = [0.5, 0.3]  # ka, ke
t_final = 24.0  # final time T = 24
dt_store = 0.5 # Step size


# Solved equation for x0(t) and plugged into x1(t) to solve it in Homework 3, Task 2b)
# x0(t) = dose * e^(-ka * t)
x0 = dose * np.exp(-k[0] * t)
# x1(t) = 2.5 * dose * ( e^(-ke*t) - e^(-ka*t) )
x1 = 2.5 * dose * (np.exp(-k[1]*t) - np.exp(-k[0]*t))


# Time steps:
t = np.arange(0,t_final+1e-6,dt_store) # Range is [0,t_final], so t_final is included.


# Lecture 7, slide 21  for 4th order Runge-Kutta-Scheme


# Save results. Time points t and x1
Output = np.concatenate(
    ( t, x1 ), axis=0) # TODO: Actual correct variables
np.savetxt('Task2bTraj.txt',
           Output, delimiter=',', fmt='%1.2f')







