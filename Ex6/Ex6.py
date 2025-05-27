# Complex Systems In Bioinformatics, SoSe2025
# Exercise 06, Task f)
#
# Submitted by:
# Kristian Reinhart, 4474140
# Duong Ha Le Minh: 5314209

import numpy as np
import matplotlib.pyplot as plt
# import os # For getting and setting working directory via os.getcwd() and os.chdir()

# --- System 1 ---
def system1_ode(x, y, v0, k1, k2, t):
    dxdt = v0 - k1 * x * y**2
    dydt = k1 * x * y**2 - k2 * y
    return [dxdt, dydt]

def plot_phase_portrait_1(V0,K1,K2,savePlotName):
    x_range = np.linspace(0.25, 5, 20)
    y_range = np.linspace(0.25, 5, 20)
    X, Y = np.meshgrid(x_range, y_range)

    DX, DY = system1_ode(X,Y,V0,K1,K2,0)

    M = np.sqrt(DX**2 + DY**2)
    DX_norm = DX / M
    DY_norm = DY / M

    plt.figure(figsize=(10, 8))

    # Plot vector field
    plt.quiver(X, Y, DX, DY, color='gray', alpha=0.7, headwidth=3, headlength=5, scale=None, angles='xy') # No normalization

    # Nullclines
    x_null_1 = np.linspace(0, 5, 500)
    y_null_1_x_is_0 = np.sqrt( V0 / (K1 * x_null_1)) # y = sqrt(v0 / (k1 * x))
    y_null_1_y_is_0 = K2 / ( K1 * x_null_1)   # y = k2 / (k1 * x)

    plt.plot(x_null_1, y_null_1_x_is_0, 'green', label=r'$y = sqrt(v_0/(k_1 x))$ ($\dot{x}=0$)')
    plt.plot(x_null_1, y_null_1_y_is_0, 'blue', label=r'$y = k_2 / (k_1 x)$ ($\dot{y}=0$)')

    # Steady states
    ss1_x, ss1_y = (K2**2/(K1*V0)), (V0/K2) # Steady state depends on parameters
    plt.plot(ss1_x, ss1_y, 'ro', markersize=8, label='Steady States')
    plt.text(ss1_x + 0.1, ss1_y, 'Steady State')
   # plt.text(ss2_x + 0.1, ss2_y, '(-1,1) Saddle')

    plt.streamplot(X, Y, DX_norm, DY_norm, color='cornflowerblue', linewidth=0.8, density=1.5, arrowstyle='->', arrowsize=1)

    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Phase Portrait: $\dot{x} = v_0 - k_1 x y^2$, $\dot{y} = k_1 x y^2 - k_2 y$, v0='+str(V0)+', k1='+str(K1)+', k2='+str(K2))
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.savefig(savePlotName) # Save the plot



plot_phase_portrait_1(2,1,1,'PhasePortrait_1_StableFocus.png') # v0, k1, k2, PlotSaveName
print("Saved PhasePortrait_1_StableFocus.png")

plot_phase_portrait_1(3,1,1,'PhasePortrait_2_StableNode.png') # v0, k1, k2, PlotSaveName
print("Saved PhasePortrait_2_StableNode.png")

plot_phase_portrait_1(5,1,3,'PhasePortrait_3_UnstableFocus.png') # v0, k1, k2, PlotSaveName
print("Saved PhasePortrait_3_UnstableFocus.png")

plot_phase_portrait_1(1,1,2,'PhasePortrait_4_UnstableNode.png') # v0, k1, k2, PlotSaveName
print("Saved PhasePortrait_4_UnstableNode.png")




# ############################################################################
# # ChatGPT:

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp

# # === Parameters ===
# v0 = 1.0     # You can try different values
# k2 = 1.0     # You can try different values

# # === ODE system ===
# def system(t, z):
#     x, y = z
#     dxdt = v0 - x * y**2
#     dydt = x * y**2 - k2 * y
#     return [dxdt, dydt]

# # === Time span and initial conditions ===
# t_span = (0, 50)
# t_eval = np.linspace(*t_span, 1000)
# z0 = [0.5, 0.5]  # Initial condition

# # === Solve ODE ===
# sol = solve_ivp(system, t_span, z0, t_eval=t_eval)

# # === Plot time course ===
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.plot(sol.t, sol.y[0], label='x(t)')
# plt.plot(sol.t, sol.y[1], label='y(t)')
# plt.xlabel('Time')
# plt.ylabel('Concentration')
# plt.title('Time Course')
# plt.legend()
# plt.grid()

# # === Phase plane plot with vector field ===
# plt.subplot(1, 2, 2)

# # Create a grid
# x_vals = np.linspace(0, 3, 20)
# y_vals = np.linspace(0, 3, 20)
# X, Y = np.meshgrid(x_vals, y_vals)

# # Compute derivatives at each point
# U = v0 - X * Y**2
# V = X * Y**2 - k2 * Y

# # Normalize vectors for quiver plot
# speed = np.sqrt(U**2 + V**2)
# U /= speed
# V /= speed

# plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=10, color='gray', alpha=0.5)
# plt.plot(sol.y[0], sol.y[1], 'b', label='Trajectory')
# plt.plot(z0[0], z0[1], 'ro', label='Initial Condition')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Phase Plane')
# plt.legend()
# plt.grid()

# plt.tight_layout()
# plt.show()



# ############################################################################
# # Perplexity.AI:

# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt

# # Parameters
# v0 = 2.0
# k2 = 1.0

# # ODE system
# def odes(t, z):
#     x, y = z
#     dxdt = v0 - x * y**2
#     dydt = x * y**2 - k2 * y
#     return [dxdt, dydt]

# # Initial conditions and time span
# z0 = [1.0, 0.5]
# t_span = (0, 20)
# t_eval = np.linspace(*t_span, 400)

# # Integrate ODE
# sol = solve_ivp(odes, t_span, z0, t_eval=t_eval)

# # Time course plots
# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1)
# plt.plot(sol.t, sol.y[0], label='x(t)')
# plt.plot(sol.t, sol.y[1], label='y(t)')
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.legend()
# plt.title('Time Course')

# # Phase plane plot
# plt.subplot(1,2,2)
# plt.plot(sol.y[0], sol.y[1])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Phase Plane')
# plt.tight_layout()
# plt.show()
