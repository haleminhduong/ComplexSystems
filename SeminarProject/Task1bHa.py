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
# import imageio

params = {
    "T_end": 5000,
    "PlaneSize": [0, 100],
    "numPrey_z1": 100,
    "numPredator_z2": 20,
    "movePrey_sigma": 0.2,
    "k1_base": 0.002,
    "movePredator_sigma": 0.2,
    "k4": 0.002,
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
    if prey_arr.shape[0] == 0 or prey_arr.shape[0] >= 1000:
        return prey_arr

    k1 = k1_base * (1 - prey_arr.shape[0] / 10000.0)

    # using binomial for compactness
    num_births = np.random.binomial(prey_arr.shape[0], k1)

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

        # indices of all the preys in range
        in_range_indices = np.where(distances_sq < collision_dist_sq)[0]

        if (not eaten_prey_indices):

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

                while (len(in_range_indices) > 0):

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
                            predator_arr = np.vstack(
                                [predator_arr, new_predator])

    if eaten_prey_indices:
        print("eaten_prey_indices ", eaten_prey_indices)
        prey_arr = np.delete(prey_arr, eaten_prey_indices, axis=0)

    return prey_arr, predator_arr


def run_simulation(params, title="Predator-Prey Simulation"):

    # init
    prey_z1 = np.random.uniform(
        params["PlaneSize"][0], params["PlaneSize"][1], size=(params["numPrey_z1"], 2))
    # print("prey_z1 ", prey_z1)
    predator_z2 = np.random.uniform(
        params["PlaneSize"][0], params["PlaneSize"][1], size=(params["numPredator_z2"], 2))
    # print("predator_z2 ", predator_z2)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(params["PlaneSize"])
    ax.set_ylim(params["PlaneSize"])

    prey_scatter = ax.scatter(
        prey_z1[:, 0], prey_z1[:, 1], c='blue', label='Prey', s=10)
    predator_scatter = ax.scatter(
        predator_z2[:, 0], predator_z2[:, 1], c='red', label='Predators', s=20)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend(loc='upper right')
    ax.set_title(title)

    # history
    prey_count = [prey_z1.shape[0]]
    # print("prey_count ", prey_count)
    predator_count = [predator_z2.shape[0]]
    # print("predator_count ", predator_count)

    # This function is called every frame
    def update(frame, prey_z1, predator_z2):

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
            prey_z1, predator_z2, params["collision_distance_sq"], params["predator_propagation_prob"], eat_multiple=True)
        print("After: ", prey_z1.shape[0])

        prey_count.append(prey_z1.shape[0])
        predator_count.append(predator_z2.shape[0])

        if prey_z1.shape[0] > 0:
            prey_scatter.set_offsets(prey_z1)
        else:
            prey_scatter.set_offsets(np.empty((0, 2)))

        # Update predator positions
        if predator_z2.shape[0] > 0:
            predator_scatter.set_offsets(predator_z2)
        else:
            predator_scatter.set_offsets(np.empty((0, 2)))

        time_text.set_text(
            f'Time: {frame+1}, Prey: {prey_z1.shape[0]}, Predators: {predator_z2.shape[0]}')

        # Return a tuple of the modified artists
        return prey_scatter, predator_scatter, time_text,

    ani = FuncAnimation(
        fig,
        lambda frame: update(frame, prey_z1, predator_z2),
        frames=params["T_end"],
        blit=True,
        interval=200# milliseconds between frames
    )
    ani.save('Animation.mp4', writer='ffmpeg')
    print("prey_count ", prey_count)
    print("predator_count", predator_count)


# run_simulation(params)

params_changed = params.copy()
params_changed.update({
    "movePrey_sigma": 2.0,
    "movePredator_sigma": 5.0,
    "k4": 0.2,
    "collision_distance_sq": 10.0
})
run_simulation(params_changed)
