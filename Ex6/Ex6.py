# Complex Systems In Bioinformatics, SoSe2025
# Exercise 06, Task f)
#
# Submitted by:
# Kristian Reinhart, 4474140
# Duong Ha Le Minh: 5314209

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# import os # For getting and setting working directory via os.getcwd() and os.chdir()

# --- ODE System, for plotting phase plane with streamplot ---
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



# y=[x0,y0]: Initial condition
# v0, k1, k2: Parameters, >0
# savePlotName: Name fro saving plot
def doTimeCourseAndPhasePlanePlot(y0,v0, k1, k2,savePlotName):
    # --- ODE System, for plotting phase plane with quiver and solving with np.solve ---
    def system(t, z):
        x, y = z
        dxdt = v0 - k1 * x * y**2
        dydt = k1 * x * y**2 - k2 * y
        return [dxdt, dydt]

    # === Time span and initial conditions ===
    t_span = (0.25, 5)
    t_eval = np.linspace(*t_span, 100)

    # === Solve ODE ===
    sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

    # === Plot time course ===
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sol.t, sol.y[0], label='x(t)')
    plt.plot(sol.t, sol.y[1], label='y(t)')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('Time Course. Initial condition: ['+str(y0[0])+','+str(y0[1])+']. v0='+str(v0)+', k1='+str(k1)+', k2='+str(k2) )
    plt.legend()
    plt.grid()

    # === Phase plane plot with vector field ===
    plt.subplot(1, 2, 2)

    # Create a grid
    x_vals = np.linspace(0.25, 10, 40)
    y_vals = np.linspace(0.25, 10, 40)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compute derivatives at each point
    DX, DY = system(0,[X,Y])

    # Normalize vectors for quiver plot
    speed = np.sqrt(DX**2 + DY**2)
    DX /= speed
    DY /= speed

    plt.quiver(X, Y, DX, DY, angles='xy', scale_units='xy', scale=5, color='gray', alpha=0.5)
    plt.plot(sol.y[0], sol.y[1], 'b', label='Trajectory')
    plt.plot(y0[0], y0[1], 'ro', label='Initial Condition')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Phase Plane. Initial condition: ['+str(y0[0])+','+str(y0[1])+']. v0='+str(v0)+', k1='+str(k1)+', k2='+str(k2) )
    plt.legend()
    plt.grid()

    plt.tight_layout()
    #plt.show()
    plt.savefig(savePlotName) # Save the plot


# y=[x0,y0]: Initial condition
# v0, k1, k2: Parameters, >0
# savePlotName: Name fro saving plot
doTimeCourseAndPhasePlanePlot([2, 1], 2.0, 1.0, 1.0, 'TimeCourse_1_StableFocus.png')
doTimeCourseAndPhasePlanePlot([2, 1], 3.0, 1.0, 1.0, 'TimeCourse_2_StableNode.png')
doTimeCourseAndPhasePlanePlot([2, 1], 5.0, 1.0, 3.0, 'TimeCourse_3_UnstableFocus.png')
doTimeCourseAndPhasePlanePlot([2, 1], 1.0, 1.0, 2.0, 'TimeCourse_4_UnstableNode.png')





