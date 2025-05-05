# Complex Systems In Bioinformatics, SoSe2025
# Exercise 04, Homework 3a)
#
# Submitted by:
# Kristian Reinhart, 4474140
# Duong Ha Le Minh: 5314209


# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
#import os # For getting and setting working directory via os.getcwd() and os.chdir()


# import the input file
try:
    In = np.loadtxt('Input.txt', ndmin=2)  # initial state
except FileNotFoundError:
    print("Error: Input.txt not found.")

np.random.seed(seed=int(In[0]))
NrSimulations = int(In[1]) # N





### Plotting the sample means and saving the file as 'Exercise3Homework3b.png'

timeT = np.arange(start=0,stop=100,step=1) # Placeholder for plotting. Substitute with actual sample times
sampleMeanX1 = 10+(np.random.rand(100)*5) # Placeholder for plotting. Substitute with actual sample means
sampleMeanX1Deviation = 2+np.random.rand(100) # Placeholder for plotting. Substitute with actual sample mean standard deviations

# Create the plot
plt.figure(figsize=(8, 4)) # figsize=(8, 4), 8 inches wide, 4 inches tall
plt.plot(timeT, sampleMeanX1, 'k-', label='sample mean')      # Solid black line
plt.plot(timeT, sampleMeanX1 + sampleMeanX1Deviation, 'r--', label='sample mean \pm stdev')       # Dashed red line
plt.plot(timeT, sampleMeanX1 - sampleMeanX1Deviation, 'r--')

plt.xlabel('time')
plt.ylabel('X1')
plt.legend()
plt.tight_layout()
plt.savefig("Exercise3Homework3b.png")
plt.show()
