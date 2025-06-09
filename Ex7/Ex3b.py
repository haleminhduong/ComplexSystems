import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter


def boundary(z, zmin, zmax):
    return np.clip(z, zmin, zmax)


def drift_term(z):
    return 2 * (np.sign(z) - z)


def get_quadrant_colour(particles):
    z1, z2 = particles[:, 0], particles[:, 1]
    # these are the quadrants we used at the AG Baumgrass at DRFZ
    colours = {
        'q1': (z1 >= 0) & (z2 >= 0),  # Red
        'q2': (z1 >= 0) & (z2 < 0),  # Blue
        'q3': (z1 < 0) & (z2 < 0),   # Green
        'q4': (z1 < 0) & (z2 >= 0)    # Black
    }
    return colours


def calculate_shannon_entropy(types_in_quadrant):
    if not types_in_quadrant:
        return 0.0

    count = Counter(types_in_quadrant)
    # print("count ", count)
    total = len(types_in_quadrant)
    # print("total ", total)
    entropy = 0.0
    for c in count.values():
        # print("c ", c)
        p = c / total
        entropy -= p * np.log2(p)
    return entropy


def simulate_drift_walk():

    zmin, zmax = -2, 2
    sigma_random = 0.3
    sigma_drift = 0.1
    tau = 1.0
    T_end = 100

    try:
        seed = np.loadtxt('Input.txt')
    except FileNotFoundError:
        print("Error: Input.txt not found.")

    np.random.seed(int(seed))

    particles = np.random.uniform(zmin, zmax, size=(100, 2))

    initial_quadrants = get_quadrant_colour(particles)
    # print(initial_quadrants)
    colours = np.empty(100, dtype='U10')

    colours[initial_quadrants['q1']] = 'red'
    colours[initial_quadrants['q2']] = 'blue'
    colours[initial_quadrants['q3']] = 'green'
    colours[initial_quadrants['q4']] = 'black'
    # print(colours)

    trajectory = []
    entropy_history = {'q1': [], 'q2': [], 'q3': [], 'q4': []}

    for t in range(int(T_end / tau)):
        trajectory.append(particles.copy())

        # get the mask for all particles, shows where they are right now
        current_quadrants = get_quadrant_colour(particles)
        # print("current_quadrants ", current_quadrants)

        for q_name, q_colour in current_quadrants.items():
            # print("qname ", q_name)
            # true for all the particles in the this quadrant at the current time
            # print("qcolour ", q_colour)
            # take the list of the original colours of the particles inside the quadrant
            colours_in_q = colours[q_colour]
            # print("colours_in_q ", colours_in_q)
            entropy = calculate_shannon_entropy(list(colours_in_q))
            entropy_history[q_name].append(entropy)

        # Update particle positions
        drift = drift_term(particles)
        random_steps = np.random.normal(0, 1, size=(100, 2))
        particles += sigma_drift * drift * tau + \
            sigma_random * np.sqrt(tau) * random_steps
        particles = boundary(particles, zmin, zmax)

    trajectory = np.array(trajectory)
    return trajectory, colours, entropy_history


def create_animation(trajectory, colours):
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title("Task 3b: Quadwell Walk with Drift")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    scatter = ax.scatter([], [], s=15)

    def update(frame):
        scatter.set_offsets(trajectory[frame])
        scatter.set_color(colours)
        return scatter,

    ani = FuncAnimation(fig, update, frames=len(
        trajectory), blit=True, interval=50)
    ani.save('Task3bTraj.mp4', writer='ffmpeg')
    print("Animation complete. Movie saved to 'Task3bTraj.mp4'.")


def plot_entropy(entropy_history, title):
    fig, ax = plt.subplots()
    # we are confident that the entropy_history is equal length for all quadrants
    time = np.arange(len(entropy_history['q1']))

    ax.plot(time, entropy_history['q1'], label='Quadrant 1 (Red)', color='red')
    ax.plot(time, entropy_history['q2'],
            label='Quadrant 2 (Blue)', color='blue')
    ax.plot(time, entropy_history['q3'],
            label='Quadrant 3 (Green)', color='green')
    ax.plot(time, entropy_history['q4'],
            label='Quadrant 4 (Black)', color='black')

    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Shannon Entropy")
    ax.set_ylim(0, 2.1)  # entropy_max for 4 types is log2(4) = 2
    ax.legend()
    ax.grid(True, linestyle='--')
    plt.savefig(f"{title.replace(' ', '_')}.png")
    print(f"Entropy plot saved as {title.replace(' ', '_')}.png")


traj, colours, entropy = simulate_drift_walk()
create_animation(traj, colours)
# plot_entropy(entropy, title="Shannon Entropy (sigma_random = 0.1)")
