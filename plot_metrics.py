# Jack DeLano

import numpy as np
import matplotlib.pyplot as plt
import math

# Load metrics
classicTimes = np.load('metrics/classic_times.npy')
classicTrainAccs = np.load('metrics/classic_trainacc.npy')
classicTestAccs = np.load('metrics/classic_testacc.npy')

lalTimes = np.load('metrics/lal_times.npy')
lalTrainAccs = np.load('metrics/lal_trainacc.npy')
lalTestAccs = np.load('metrics/lal_testacc.npy')

# Plot metrics
#plt.plot(classicTimes, classicTrainAccs)
plt.plot(classicTimes, classicTestAccs)

#plt.plot(lalTimes, lalTrainAccs)
plt.plot(lalTimes, lalTestAccs)

plt.xlabel('Time (s)')
plt.ylabel('Test Accuracy (%)')

plt.show()

