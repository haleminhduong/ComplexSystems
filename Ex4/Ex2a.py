# Complex Systems In Bioinformatics, SoSe2025
# Exercise 04, Homework 2a)
#
# Implementation of implicit Euler Method ODE solving
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
    # Initial state of system, x0(t0), x1(t0)
    In = np.loadtxt('Input.txt', ndmin=2)
except FileNotFoundError:
    print("Error: Input.txt not found.")

# -----Fixed Quantities-----
X0 = np.array([[In[0, 0]], [In[1, 0]]], dtype=int)  # Store initial condition


# Reaction parameters

k = [0.5, 0.3]  # ka, ke
t_final = 24.0  # final time T = 24
dt_store = 0.5  # Step size

# Solving the equation for x0(t)
# f(t_{n+1}) = -r0 = -ka * x0 = -0.5 * x0
# x0(t_{n+1}) = x0(t_n) + step * f(t_{n+1})
# x0(t_{n+1}) = x0(t_n) + 0.5 * (-0.5 * x0(t_{n+1})
# x0(t_{n+1}) = x0(t_n) - 0.25 * x0(t_{n+1})
# 1.25 * x0(t_{n+1}) = x0(t_n)
# x0(t_{n+1}) = 0.8 * x0(t_n)

<<<<<<< Updated upstream
# --- Read Initial Conditions from Input.txt ---
=======
# TODO: Remove this comment:
    # Code based on:
        # https://www.youtube.com/watch?v=N7Oh0mk4YGc
        # https://jonshiach.github.io/ODEs-book/_pages/B_Python_code.html

# f: function of IVP tp solve
# tspan: Time span to solve for, here [0,t_final]
# X0: Initial value for x0 and x1, [x0(t0), x1(t0)]
# deltaT: Time step size
# solver: Generic name for solver function to use for solving the IVP. We'll use our own euler_implicit
def solveIVP(f, tspan, X0, deltaT, solver):

    # Initialise t and y arrays
    t = np.arange(tspan[0], tspan[1] + 1e-6, deltaT) # Range is [0,t_final], so t_final is included.
    x0 = np.zeros(len(t)) # Preallocate space for solution
    x1 = np.zeros(len(t)) # Preallocate space for solution
    t[0] = tspan[0]
    x0[0,:] = X0[0] # Set the initial value for x0(t0)
    x1[0,:] = X0[1] # Set the initial value for x1(t0)

    # Go through all time points and calculate single step solver solution
    for n in range(len(t) - 1):
        x1[n+1,:] = solver(f, t[n], x1[n,:], deltaT)

    return t, x1
>>>>>>> Stashed changes

# Initialize simulation variables
t_current = 0.0
x = X0.copy()
x0_current = x[0, 0]
x1_current = x[1, 0]

<<<<<<< Updated upstream
T_Store = [t_current]
X1_Store = [x1_current]
=======
# Single step solution for the implicit euler method
def euler(f, t, y, h):
    return y + h * f(t, y)
>>>>>>> Stashed changes

num_steps = int(round(t_final / dt_store))

<<<<<<< Updated upstream
for _ in range(num_steps):
    x0_next = 0.8 * x0_current
=======
# Define function of IVP to solve
def f(t, y):
    return t * y # TODO: Proper function.

    # Solved equation for x0(t) and plugged into x1(t) to solve it in Homework 3, Task 2b)
    # x0 = dose * np.exp(-k[0] * t)                             # x0(t) = dose * e^(-ka * t)
    # x1 = 2.5 * dose * (np.exp(-k[1]*t) - np.exp(-k[0]*t))     # x1(t) = 2.5 * dose * ( e^(-ke*t) - e^(-ka*t) )



# Define IVP parameters
tspan = [0, 1]  # boundaries of the t domain
y0 = [1]        # solution at the lower boundary
h = 0.2         # step length

# Calculate the solution to the IVP
t, y = solveIVP(f, tspan, y0, h, euler)
>>>>>>> Stashed changes

    x1_next = (x1_current + 0.2 * x0_current)/1.15

    x0_current = x0_next
    x1_current = x1_next
    t_current += dt_store

    T_Store.append(t_current)
    X1_Store.append(x1_current)

Output = np.concatenate(
    (np.array(T_Store, ndmin=2), np.array(X1_Store, ndmin=2)), axis=0)
np.savetxt('Task2aTraj.txt',
           Output, delimiter=',', fmt='%1.2f')
