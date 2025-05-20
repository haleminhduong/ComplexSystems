import numpy as np
import matplotlib.pyplot as plt

# --- System 1 ---
def system1_ode(Y, t):
    x, y = Y
    dxdt = x**2 - y
    dydt = x + y
    return [dxdt, dydt]

def plot_phase_portrait_1():
    x_range = np.linspace(-2.5, 2.5, 20)
    y_range = np.linspace(-1.5, 3.5, 20)
    X, Y = np.meshgrid(x_range, y_range)

    DX, DY = system1_ode([X, Y], 0)

    M = np.sqrt(DX**2 + DY**2)
    DX_norm = DX / M
    DY_norm = DY / M

    plt.figure(figsize=(10, 8))
    
    # Plot vector field
    plt.quiver(X, Y, DX, DY, color='gray', alpha=0.7, headwidth=3, headlength=5, scale=None, angles='xy') # No normalization

    # Nullclines
    x_null_1 = np.linspace(-2.5, 2.5, 400)
    y_null_1_x_is_0 = x_null_1**2 # y = x^2
    y_null_1_y_is_0 = -x_null_1   # y = -x

    plt.plot(x_null_1, y_null_1_x_is_0, 'green', label=r'$y = x^2$ ($\dot{x}=0$)')
    plt.plot(x_null_1, y_null_1_y_is_0, 'blue', label=r'$y = -x$ ($\dot{y}=0$)')

    # Steady states
    ss1_x, ss1_y = 0, 0
    ss2_x, ss2_y = -1, 1
    plt.plot([ss1_x, ss2_x], [ss1_y, ss2_y], 'ro', markersize=8, label='Steady States')
    plt.text(ss1_x + 0.1, ss1_y, '(0,0) Unstable Spiral')
    plt.text(ss2_x + 0.1, ss2_y, '(-1,1) Saddle')

    plt.streamplot(X, Y, DX_norm, DY_norm, color='cornflowerblue', linewidth=0.8, density=1.5, arrowstyle='->', arrowsize=1)

    plt.xlim([-2.5, 2.5])
    plt.ylim([-1.5, 3.5])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Phase Portrait: $\dot{x} = x^2 - y$, $\dot{y} = x + y$')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.savefig('phase_portrait_1.png') # Save the plot

# --- System 2 ---
def system2_ode(Y, t):
    x, y = Y
    dxdt = x**2 - 1
    dydt = 2*y
    return [dxdt, dydt]

def plot_phase_portrait_2():
    x_range = np.linspace(-2.5, 2.5, 20)
    y_range = np.linspace(-2.0, 2.0, 20) # Adjusted y-range for better view of y=0 nullcline
    X, Y = np.meshgrid(x_range, y_range)

    DX, DY = system2_ode([X, Y], 0)

    plt.figure(figsize=(10, 8))
    
    plt.quiver(X, Y, DX, DY, color='gray', alpha=0.7, headwidth=3, headlength=5, angles='xy')

    # Nullclines
    # x_dot = 0 => x^2 - 1 = 0 => x = 1 or x = -1
    plt.axvline(1, color='green', label=r'$x = 1$ ($\dot{x}=0$)')
    plt.axvline(-1, color='green', label=r'$x = -1$ ($\dot{x}=0$)')
    # y_dot = 0 => 2y = 0 => y = 0
    plt.axhline(0, color='blue', label=r'$y = 0$ ($\dot{y}=0$)')


    # Steady states
    ss1_x, ss1_y = 1, 0
    ss2_x, ss2_y = -1, 0
    plt.plot([ss1_x, ss2_x], [ss1_y, ss2_y], 'ro', markersize=8, label='Steady States')
    plt.text(ss1_x + 0.1, ss1_y, '(1,0) Unstable Node')
    plt.text(ss2_x + 0.1, ss2_y, '(-1,0) Saddle')

    Y_stream = Y + 1e-9 # Add a tiny perturbation to avoid division by zero if M becomes zero on y=0
    DX_stream, DY_stream = system2_ode([X, Y_stream],0)

    plt.streamplot(X, Y, DX_stream, DY_stream, color='cornflowerblue', linewidth=0.8, density=1.5, arrowstyle='->', arrowsize=1)


    plt.xlim([-2.5, 2.5])
    plt.ylim([-2.0, 2.0])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Phase Portrait: $\dot{x} = x^2 - 1$, $\dot{y} = 2y$')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axvline(0, color='black', lw=0.5) # Vertical axis x=0
    plt.savefig('phase_portrait_2.png') # Save the plot

plot_phase_portrait_1()
print("Saved phase_portrait_1.png")

plot_phase_portrait_2()
print("Saved phase_portrait_2.png")
