# -*- coding: utf-8 -*-
"""
Complex Systems in Bioinformatics
Seminar, Block 3, Project 2
Homework 1b)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

np.random.seed(int(3))

params = {
    "T_end": 1000,
    "PlaneSize": [0, 100],
    "numPrey_z1": 100,
    "numPredator_z2": 20,
    "movePrey_sigma": 10,
    "k1_base": 0.1,
    "movePredator_sigma": 10,
    "k4": 0.1,
    "collision_distance_sq": 20.0,
    "predator_propagation_prob": 0.5,
}


# Function to perform actual move while checking boundaries. Use clipped boundaries.
def boundMove(a, b, plane_size):
    if (a+b < plane_size[0]):
        return 0
    elif (a+b > plane_size[1]):
        return plane_size[1]
    else:
        return a+b


# Function to move ALL agents to a new random position
def moveAgents(agentArr, sigma, plane_size):
    if (len(agentArr) > 0):
        for i in range(len(agentArr)):
            xMove = np.random.normal(0, sigma)
            yMove = np.random.normal(0, sigma)
            agentArr[i, 0] = boundMove(
                agentArr[i, 0], xMove, plane_size)  # Update X coordinate
            agentArr[i, 1] = boundMove(
                agentArr[i, 1], yMove, plane_size)  # Update Y coordinate
    return agentArr


# R1
def preyReproduction(prey_arr, k1_base):
    if prey_arr.shape[0] == 0 or prey_arr.shape[0] >= 10000:
        return prey_arr

    k1 = k1_base * (1 - prey_arr.shape[0] / 10000.0)

    # using binomial for compactness
    num_births = np.random.binomial(prey_arr.shape[0], k1)
    print("Born ", num_births)

    if num_births > 0:
        parent_indices = np.random.choice(
            prey_arr.shape[0], size=num_births, replace=True)
        new_prey = prey_arr[parent_indices]
        prey_arr = np.vstack([prey_arr, new_prey])
    return prey_arr


# R4
def predatorDeath(predator_arr, k4):
    if predator_arr.shape[0] == 0:
        return predator_arr

    # instead of dying, find who didn't
    predator_survived = np.random.rand(predator_arr.shape[0]) > k4
    return predator_arr[predator_survived]


# R2 and R3
def interactions(prey_arr, predator_arr, collision_dist_sq, prop_prob, eat_multiple):
    if prey_arr.shape[0] == 0 or predator_arr.shape[0] == 0:
        return prey_arr, predator_arr

    eaten_prey_indices = []

    for i in range(predator_arr.shape[0]):
        distances_sq = np.sum((prey_arr - predator_arr[i])**2, axis=1)
        # if i == 0:
        #     print("distances_sq ", distances_sq)

        # indices of all the preys in range
        in_range_indices = np.where(distances_sq < collision_dist_sq)[0]
        print("in_range_indices ", in_range_indices)

        if (not eat_multiple):

            ate = False

            while (not ate) and (len(in_range_indices) > 0):
                # if (not ate):
                # print("Not eaten")
                # print("in_range_indices ", in_range_indices)
                # print("Prey in sight: ", len(in_range_indices))

                # choose one in range
                prey_to_eat = np.random.choice(in_range_indices)
                in_range_indices = in_range_indices[~np.isin(
                    in_range_indices, prey_to_eat)]
                # print("prey_to_eat ", prey_to_eat)
                # print("eaten_prey_indices ", eaten_prey_indices)
                # print("in_range_indices ", in_range_indices)

                if prey_to_eat not in eaten_prey_indices:
                    # print("prey_to_eat not in eaten prey")
                    eaten_prey_indices.append(prey_to_eat)
                    # R3
                    if np.random.rand() < prop_prob:
                        new_predator = predator_arr[i]
                        predator_arr = np.vstack([predator_arr, new_predator])
                    ate = True
                    # print("Ate")

        else:

            # print("eaten_prey_indices ", eaten_prey_indices)

            if len(in_range_indices) != 0:

                # choose one in range
                for j in in_range_indices:

                    if j not in eaten_prey_indices:
                        # print("prey_to_eat not in eaten prey")
                        eaten_prey_indices.append(j)
                        # R3
                        if np.random.rand() < prop_prob:
                            new_predator = predator_arr[i]
                            predator_arr = np.vstack(
                                [predator_arr, new_predator])

    if eaten_prey_indices:
        print("eaten_prey_indices ", eaten_prey_indices)
        print("Prey count: ", prey_arr.shape[0])
        prey_arr = np.delete(prey_arr, eaten_prey_indices, axis=0)
        print("Prey count: ", prey_arr.shape[0])
    else:
        print("None eaten")

    return prey_arr, predator_arr


def create_animation(prey_history, predator_history,  eat_multiple, params_changed):

    title = "Predator-Prey Simulation"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('black')  # Set a background color for contrast
    ax.set_xlim(params["PlaneSize"])
    ax.set_ylim(params["PlaneSize"])
    ax.set_xticks([])
    ax.set_yticks([])

    # Initialize scatter plots. We start with the data from the first frame.
    # The 'set_offsets' method will update their positions in the animation loop.
    prey_scatter = ax.scatter([], [], c='cyan', label='Prey', s=10)
    predator_scatter = ax.scatter([], [], c='red', label='Predators', s=25)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')
    ax.legend(loc='upper right')
    ax.set_title(title, color='white')

    # The update function is the core of the animation.
    # It's called for each 'frame' of the animation.
    def update(frame, prey_hist, pred_hist):
        # --- KEY CHANGE ---
        # Get the agent positions for the current frame from the history lists.
        prey_positions = prey_hist[frame]
        predator_positions = pred_hist[frame]

        # Update the data for the scatter plots.
        # 'set_offsets' is the correct and efficient way to update scatter plot data.
        if prey_positions.shape[0] > 0:
            prey_scatter.set_offsets(prey_positions)
        else:
            # Clear plot if no agents
            prey_scatter.set_offsets(np.empty((0, 2)))

        if predator_positions.shape[0] > 0:
            predator_scatter.set_offsets(predator_positions)
        else:
            predator_scatter.set_offsets(np.empty((0, 2)))

        # Update the text elements
        time_text.set_text(
            f'Time: {
                frame+1} | Prey: {len(prey_positions)} | Predators: {len(predator_positions)}'
        )

        # The function must return a tuple of the artists that were modified.
        return prey_scatter, predator_scatter, time_text

    # Create the animation object.
    # 'fargs' is used to pass the history lists to the update function.
    ani = FuncAnimation(
        fig,
        update,
        # Number of frames is the length of our simulation
        frames=len(prey_history),
        fargs=(prey_history, predator_history),
        blit=True,
        interval=17 # milliseconds between frames (e.g., 50ms = 20fps)
    )

    # Save the animation as an MP4 file. You'll need ffmpeg installed.
    print("Saving animation... This may take a moment.")
    name = "ani"
    if eat_multiple:
        name += "_multiple"
    else:
        name += "_one"
    if params_changed:
        name += "_changed"
    else:
        name += "_standard"
    name += "_beautiful.mp4"
    ani.save(name, writer='ffmpeg', dpi=150)
    print("Animation saved as 'predator_prey_animation.mp4'")


def run_simulation(params, eat_multiple):

    # init
    prey_z1 = np.random.uniform(
        params["PlaneSize"][0], params["PlaneSize"][1], size=(params["numPrey_z1"], 2))
    # print("prey_z1 ", prey_z1)
    predator_z2 = np.random.uniform(
        params["PlaneSize"][0], params["PlaneSize"][1], size=(params["numPredator_z2"], 2))
    # print("predator_z2 ", predator_z2)

    # history
    prey_count = [prey_z1.shape[0]]
    # print("prey_count ", prey_count)
    predator_count = [predator_z2.shape[0]]
    # print("predator_count ", predator_count)
    prey_history = [prey_z1]
    predator_history = [predator_z2]

    for t in range(params["T_end"]-1):
        # R1
        prey_z1 = preyReproduction(prey_z1, params["k1_base"])

        # R4
        predator_z2 = predatorDeath(predator_z2, params["k4"])

        prey_z1 = moveAgents(
            prey_z1, params["movePrey_sigma"], params["PlaneSize"])
        predator_z2 = moveAgents(
            predator_z2, params["movePredator_sigma"], params["PlaneSize"])

        # R2 and R3
        print("Before: ", prey_z1.shape[0])
        prey_z1, predator_z2 = interactions(
            prey_z1, predator_z2, params["collision_distance_sq"], params["predator_propagation_prob"], eat_multiple)
        print("After: ", prey_z1.shape[0])

        prey_history.append(prey_z1)
        predator_history.append(predator_z2)
        prey_count.append(prey_z1.shape[0])
        predator_count.append(predator_z2.shape[0])

    print("prey_count ", prey_count)
    print("predator_count", predator_count)
    return prey_history, predator_history, prey_count, predator_count


def plot_simulation_results(prey_count, predator_count, eat_multiple, params_changed):
    time_steps = range(len(prey_count))

    plt.figure(figsize=(10, 6))

    # Plot prey count over time
    plt.plot(time_steps, prey_count, label='$z_1(t)$ (Prey)', color='blue')

    # Plot predator count over time
    plt.plot(time_steps, predator_count,
             label='$z_2(t)$ (Predator)', color='red')

    plt.xlabel('Time Steps')
    plt.ylabel('Population')

    # Set title based on the 'wikipedia' flag
    plt.title(
        'Dynamical behavior of $z_1$ (prey) and $z_2$ (predator) over time')

    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    # Set filename based on the 'wikipedia' flag and save the plot
    name = 'plot'
    if eat_multiple:
        name += "_multiple"
    else:
        name += "_one"
    if params_changed:
        name += "_changed"
    else:
        name += "_standard"
    name += "_beautiful.png"

    plt.savefig(name)
    print(f"Plot saved to {name}")
    plt.show()


eat_multiple = False 
params_changed_bool = True

params_changed = params.copy()
params_changed.update({
    "movePrey_sigma": 2.0,
    "movePredator_sigma": 5.0,
    "k4": 0.2,
    "k1_base": 0.1,
    "collision_distance_sq": 10.0
})
params_changed_beautiful = params.copy()
params_changed_beautiful.update({
    "movePrey_sigma": 1.0,
    "movePredator_sigma": 1.0,
    "k4": 0.06,
    "k1_base": 0.06,
    "collision_distance_sq": 20.0
})
# prey_history, predator_history, prey_count, predator_count = run_simulation(
#    params_changed)
prey_history, predator_history, prey_count, predator_count = run_simulation(
    params_changed, eat_multiple)
# create_animation(prey_history, predator_history,
#                  eat_multiple, params_changed_bool)
plot_simulation_results(prey_count, predator_count,
                        eat_multiple, params_changed_bool)

# run_simulation(params_changed)
