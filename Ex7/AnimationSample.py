# -*- coding: utf-8 -*-
"""
@author: Kristian Reinhart
"""

# Import packages
import os # For getting and setting working directory via os.getcwd() and os.chdir()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Example from matplotlib documentation:
# https://matplotlib.org/stable/users/explain/animations/animations.html


# Define an empty plot
fig, ax = plt.subplots()


# Define 2 functions for plotting:
t = np.linspace(0, 3, 40) # Define x axis points

g = -9.81
v0 = 12
z = g * t**2 / 2 + v0 * t # Function 1 to plot

v02 = 5
z2 = g * t**2 / 2 + v02 * t  # Function 2 to plot


# Add first point of each function to plot as an initialization:
scat = ax.scatter(t[0], z[0], c="b", s=5, label=f'v0 = {v0} m/s')
line2 = ax.plot(t[0], z2[0], label=f'v0 = {v02} m/s')[0]

# Define x and y limits and axis labels / legends for the plot
ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
ax.legend()

# Update function for the plot animation.
def update(frame):
    # for each frame, update the data stored on each artist.
    x = t[:frame] # x coordinate
    y = z[:frame] # y coodrinate (for function 1)
    # update the scatter plot:
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    # update the line plot:
    line2.set_xdata(t[:frame])
    line2.set_ydata(z2[:frame])
    return (scat, line2)

# frames: Number of frames to plot / animate. Here could also be 'size(z)' or 'size(z2)'
# interval: Milliseconds between frames, or how long each frame is shown in milliseconds. 100ms/f => 10fps
ani = FuncAnimation(fig=fig, func=update, frames=40, interval=100)

# Save as MP4. Can set fps, e.g. 'fps=30', but overrides interval of FuncAnimation
ani.save('aniExample.mp4', writer='ffmpeg')
# Save as GIF
# ani.save('sine_wave.gif', writer='pillow')

plt.show()
