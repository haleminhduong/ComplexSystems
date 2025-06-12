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

# Inport required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

wikipedia = False # If model should be exactly as in wikipedia (True) or ours (False)

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
    if(wikipedia):
        dz1dt = k1 * z1 - k2 * z1 * z2 # As in wikipedia, regular Lotka-Volterra equation
    else:
        dz1dt = k1 * z1 - k2 * z1 * z2 - k3 * z1 * z2
    dz2dt = k3 * z1 * z2 - k4 * z2
    return [dz1dt, dz2dt]


# Define time span for solve_ivp
t_span = (0, T_end)
t_eval = np.linspace(*t_span, 1000)

# Solve the system
solution = solve_ivp(model, t_span, z0, t_eval=t_eval, method='RK45')

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(solution.t, solution.y[0], label='$z_1(t)$', color='blue')
plt.plot(solution.t, solution.y[1], label='$z_2(t)$', color='red')
plt.xlabel('Time')
plt.ylabel('Population')
if(wikipedia):
    plt.title('Lotka-Volterra Model (Wikipedia)')
else:
    plt.title('Dynamical behavior of $z_1$ (prey) and $z_2$ (predator) over time')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
#plt.show()
if(wikipedia):
    plt.savefig('Task1a_PredatorPreyOscillationWikipediaParametersAndModel') # Save the plot
else:
    plt.savefig('Task1a_PredatorPreyOscillationWikipediaParameters') # Save the plot



################################################################################################
#### Plot phase portrait

# Inport required libraries
import numpy as np
import matplotlib.pyplot as plt

def plot_phase_portrait():
    # Parameters of the model
    k1 = 0.3
    k2 = 0.01
    k3 = 0.01
    k4 = 0.3

    # ODE system
    def dzdt(z, t=0):
        z1, z2 = z
        dz1dt = k1 * z1 - k2 * z1 * z2 - k3 * z1 * z2
        dz2dt = k3 * z1 * z2 - k4 * z2
        return np.array([dz1dt, dz2dt])

    # Create grid for phase portrait
    z1_vals = np.linspace(0, 100, 50)
    z2_vals = np.linspace(0, 100, 50)
    Z1, Z2 = np.meshgrid(z1_vals, z2_vals)

    # Compute vector field
    dZ1dt, dZ2dt = dzdt([Z1, Z2])
    norm = np.sqrt(dZ1dt**2 + dZ2dt**2)
    dZ1dt /= norm
    dZ2dt /= norm

    # Plot vector field
    plt.figure(figsize=(8, 6))
    plt.streamplot(z1_vals, z2_vals, dZ1dt, dZ2dt, color='gray', density=1)

    # Plot nullclines
    plt.axhline(y= k1 / (k2 + k3), color='blue', linestyle='--', label='$dz_1/dt = 0$')
    plt.axvline(x= k4 / k3 , color='red', linestyle='--', label='$dz_2/dt = 0$')

    # Add point for initial condition
    plt.plot(28, 10, 'ko', label='Initial condition', color='black')

    plt.xlim(0, 100)
    plt.ylim(0, 50)
    plt.xlabel('$z_1$ (prey)')
    plt.ylabel('$z_2$ (predator)')
    plt.title('Phase Portrait: $\dot{z_1} = k_1 * z_1 - (k_2+k_3) * z_1 * z_2$, $\dot{z_2} = k_3 * z_1 * z_2 - k_4 * z_2$')
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig('Task1a_PhasePortrait.png') # Save the plot

plot_phase_portrait()

