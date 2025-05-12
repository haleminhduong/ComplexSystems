# Complex Systems In Bioinformatics, SoSe2025
# Exercise 04, Homework 2a)
#
# Implementation of implicit Euler Method ODE solving
#
# Submitted by:
# Kristian Reinhart, 4474140
# Duong Ha Le Minh: 5314209


# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
# import os # For getting and setting working directory via os.getcwd() and os.chdir()


# import the input file
try:
    In = np.loadtxt('Input.txt', ndmin=2)  # initial state
except FileNotFoundError:
    print("Error: Input.txt not found.")


NrSimulations = 24 # N






