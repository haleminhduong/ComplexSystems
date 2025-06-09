# Ex2b.py
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


def boundary(z, zmin, zmax):
    if (z[0] <= zmin):
        z[0] = zmin
    if (z[1] <= zmin):
        z[1] = zmin
    if (z[0] >= zmax):
        z[0] = zmax
    if (z[1] >= zmax):
        z[1] = zmax
    return z


def simulate_random_walk():

    zmin, zmax = -2, 2
    sigma = 0.3
    tau = 1.0
    T_end = 100

    try:
        seed = np.loadtxt('Input.txt')
    except FileNotFoundError:
        print("Error: Input.txt not found.")

    np.random.seed(int(seed))

    particle_z = np.random.uniform(zmin, zmax, size=2)

    trajectory = []

    for t in range(int(T_end / tau)):
        trajectory.append(particle_z.copy())

        # Euler-Maruyama step for random walk (no drift)
        random_step = np.random.normal(0, 1, size=2)
        particle_z += sigma * np.sqrt(tau) * random_step

        particle_z = boundary(particle_z, zmin, zmax)

    trajectory = np.array(trajectory)

    # Animation setup
    fig, ax = plt.subplots()
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(zmin, zmax)
    ax.set_title("Task 2b: 2D Random Walk")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_aspect('equal')

    particle_plot, = ax.plot([], [], 'r.', ms=5)
    traj_plot, = ax.plot([], [], 'g-', lw=1, alpha=0.5)

    def update(frame):

        particle_plot.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])

        traj_plot.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])

        return particle_plot, traj_plot

    ani = FuncAnimation(fig, update, frames=len(
        trajectory), blit=True, interval=50)

    ani.save('Task2bTraj.mp4', writer='ffmpeg', fps=15)
    print("Animation complete. Movie saved to 'Task2bTraj.mp4'.")


simulate_random_walk()
