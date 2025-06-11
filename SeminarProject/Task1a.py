# -*- coding: utf-8 -*-
"""
Complex Systems in Bioinformatics
Seminar, Block 3, Project 2
Homework 1a)
@authors:   Duong Ha Le Minh, 5314209
            Kristian Reinhart, 4474140
"""

# Inport required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define parameters
k1 = 0.3
k2 = 0.01
k3 = 0.01
k4 = 0.3

T_end = 100

# Define initial conditions at t0
z0 = [28, 10]


# Define ODE system for (slightly modified) Lotka-Volterra predator-prey model as given in task and as needed by solve_ivp()
def model(t, z):
    z1, z2 = z
    dz1dt = k1 * z1 - k2 * z1 * z2 - k3 * z1 * z2
    dz2dt = k3 * z1 * z2 - k4 * z2
    return [dz1dt, dz2dt]


# Define time span for solve_ivp
t_span = (0, T_end)
t_eval = np.linspace(*t_span, 100)

# Solve the system
solution = solve_ivp(model, t_span, z0, t_eval=t_eval, method='RK45')

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(solution.t, solution.y[0], label='$z_1(t)$', color='blue')
plt.plot(solution.t, solution.y[1], label='$z_2(t)$', color='red')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Dynamical behavior of $z_1$ (prey) and $z_2$ (predator) over time')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig('Task1a_PredatorPreyOscillation') # Save the plot


################################################################################################
#### Redo it with same parameters as wikipedia: https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations#A_simple_example


# Define parameters
k1 = 1.1
k2 = 0.4
k3 = 0.1
k4 = 0.4

T_end = 100

# Define initial conditions at t0
z0 = [10, 10]


# Define ODE system for (slightly modified) Lotka-Volterra predator-prey model as given in task and as needed by solve_ivp()
def model(t, z):
    z1, z2 = z
    dz1dt = k1 * z1 - k2 * z1 * z2 - k3 * z1 * z2
    #dz1dt = k1 * z1 - k2 * z1 * z2 # As in wikipedia, regular Lotka-Volterra equation
    dz2dt = k3 * z1 * z2 - k4 * z2
    return [dz1dt, dz2dt]


# Define time span for solve_ivp
t_span = (0, T_end)
t_eval = np.linspace(*t_span, 100)

# Solve the system
solution = solve_ivp(model, t_span, z0, t_eval=t_eval, method='RK45')

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(solution.t, solution.y[0], label='$z_1(t)$', color='blue')
plt.plot(solution.t, solution.y[1], label='$z_2(t)$', color='red')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Dynamical behavior of $z_1$ (prey) and $z_2$ (predator) over time')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig('Task1a_PredatorPreyOscillationWikipediaParameters') # Save the plot
