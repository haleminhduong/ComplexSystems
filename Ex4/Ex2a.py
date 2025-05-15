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

# --- Read Initial Conditions from Input.txt ---

# Initialize simulation variables
t_current = 0.0
x = X0.copy()
x0_current = x[0, 0]
x1_current = x[1, 0]

T_Store = [t_current]
X1_Store = [x1_current]

num_steps = int(round(t_final / dt_store))

for _ in range(num_steps):
    x0_next = 0.8 * x0_current

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
