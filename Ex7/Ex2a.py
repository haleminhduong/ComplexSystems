# Ex2a.py
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

    np.savetxt('Task2Traj.txt', np.array(
        trajectory), fmt='%.2f', delimiter=',')
    print("Simulation complete. Trajectory saved to 'Task2Traj.txt'.")


simulate_random_walk()
