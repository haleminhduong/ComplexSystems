from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

size_min = 0
# with this size, all the inviduals are captured
size_max = 20
c = np.random.uniform(size_min, size_max, size=(30, 3))
v = np.random.uniform(size_min, size_max, size=(30, 3))
for i in range(v.shape[0]):
    v[i] = v[i] / np.linalg.norm(v[i])

params = {
    'T_end': 100.0,
    'tau': 0.1,
    'r_r': 1.0,
    'r_o': 2.5,
    'r_a': 17.5,
    'alpha': 270,
    's': 3.0,
    'theta': 40.0
}


def calculate_direction(c, v, params):
    r_r = params['r_r']
    r_o = params['r_o']
    r_a = params['r_a']
    alpha = np.deg2rad(params['alpha'])
    num_individuals = c.shape[0]
    desired_directions = np.zeros_like(v)

    for i in range(num_individuals):
        neighbours_in_zor = []
        neighbours_in_zoo = []
        neighbours_in_zoa = []
        looking_at = v[i]

        for j in range(num_individuals):
            if i == j:
                continue
            vector_to_other = c[j] - c[i]
            distance = np.linalg.norm(vector_to_other)

            if distance > r_a:
                continue

            # np.dot gives the scalar product
            # divide distance for normalisation (not a unit vector)
            # arccos gives us the angle
            angle_to_other = np.arccos(
                np.dot(looking_at, vector_to_other) / distance)

            if angle_to_other > (alpha / 2):
                continue

            if distance <= r_r:
                neighbours_in_zor.append(j)
            elif distance <= r_o:
                neighbours_in_zoo.append(j)
            else:
                neighbours_in_zoa.append(j)

        if neighbours_in_zor:
            direction = repulsion(i, neighbours_in_zor, c)
        else:
            direction = orientation_attraction(
                i, neighbours_in_zoo, neighbours_in_zoa, c, v)

        norm = np.linalg.norm(direction)
        if norm > 0:
            desired_directions[i] = direction / norm
        else:
            desired_directions[i] = v[i]

    return desired_directions


def repulsion(this, neighbours_in_zor, c):
    repulsion_vector = np.zeros(3)
    for other in neighbours_in_zor:
        vector_from_other = c[this] - c[other]
        distance = np.linalg.norm(vector_from_other)
        if distance > 0:
            repulsion_vector += vector_from_other / distance
    return repulsion_vector


def orientation_attraction(this, neighbours_in_zoo, neighbours_in_zoa, c, v):
    orientation_vector = np.zeros(3)
    if neighbours_in_zoo:
        for other in neighbours_in_zoo:
            orientation_vector += v[other]

    attraction_vector = np.zeros(3)
    if neighbours_in_zoa:
        for other in neighbours_in_zoa:
            vector_to_other = c[other] - c[this]
            distance = np.linalg.norm(vector_to_other)
            if distance > 0:
                attraction_vector += vector_to_other / distance

    if neighbours_in_zoo and neighbours_in_zoa:
        return (orientation_vector + attraction_vector) / 2
    elif neighbours_in_zoo:
        return orientation_vector
    elif neighbours_in_zoa:
        return attraction_vector
    else:
        return v[this]


def update_positions(c, v, desired_directions, params):
    tau = params['tau']
    s = params['s']
    theta_max = np.deg2rad(params['theta']) * tau

    for i in range(c.shape[0]):
        angle_between = np.arccos(
            np.clip(np.dot(v[i], desired_directions[i]), -1.0, 1.0))

        if angle_between <= theta_max:
            v[i] = desired_directions[i]
        else:
            # using Rodrigues' rotation formula
            # https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
            axis = np.cross(v[i], desired_directions[i])
            if np.linalg.norm(axis) > 0:
                axis = axis / np.linalg.norm(axis)
                v[i] = v[i] * np.cos(theta_max) + np.cross(axis, v[i]) * np.sin(
                    theta_max) + axis * np.dot(axis, v[i]) * (1 - np.cos(theta_max))

        c[i] += v[i] * s * tau
    return c, v


# def calculate_polarization(v):
#     return np.linalg.norm(np.sum(v, axis=0)) / v.shape[0]
#
#
# def calculate_angular_momentum(c, v):
#     c_group = np.mean(c, axis=0)
#     angular_momentum = 0
#     for i in range(c.shape[0]):
#         r_ic = c[i] - c_group
#         angular_momentum += np.linalg.norm(np.cross(r_ic, v[i]))
#     return angular_momentum / c.shape[0]


history_c = []
history_v = []
# polarization_history = []
# angular_momentum_history = []

for t in np.arange(0, params['T_end'], params['tau']):
    desired_directions = calculate_direction(c, v, params)
    c, v = update_positions(c, v, desired_directions, params)
    history_c.append(c.copy())
    history_v.append(v.copy())
    # polarization_history.append(calculate_polarization(v))
    # angular_momentum_history.append(calculate_angular_momentum(c, v))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


def update(i):
    ax.clear()
    ax.set_xlim(np.min(history_c[i][:, 0]) - 5, np.max(history_c[i][:, 0]) + 5)
    ax.set_ylim(np.min(history_c[i][:, 1]) - 5, np.max(history_c[i][:, 1]) + 5)
    ax.set_zlim(np.min(history_c[i][:, 2]) - 5, np.max(history_c[i][:, 2]) + 5)
    ax.set_title("Torus Formation")
    ax.quiver(history_c[i][:, 0], history_c[i][:, 1], history_c[i][:, 2],
              history_v[i][:, 0], history_v[i][:, 1], history_v[i][:, 2], length=2, normalize=True)
    return fig,


ani = FuncAnimation(fig, update, frames=len(history_c), interval=17)
ani.save('FishSwarm.mp4', writer='ffmpeg')
print("Animation complete. Movie saved to 'FishSwarm.mp4'.")
# plt.show()

# plt.figure(figsize=(12, 5))
# plt.plot(np.arange(0, params['T_end'], params['tau']), polarization_history)
# plt.title("P_group")
# plt.xlabel("t")
# plt.ylabel("P_group")
# plt.ylim(0, 1)
# plt.show()
#
# plt.plot(np.arange(0, params['T_end'],
#          params['tau']), angular_momentum_history)
# plt.title("M_group")
# plt.xlabel("t")
# plt.ylabel("M_group")
# plt.show()
