import numpy as np
import matplotlib.pyplot as plt
import pylab

n=400
t = np.array(range(n))
mean = 0
std = 0.2
num_samples = n
samples = np.random.normal(mean, std, size=num_samples)
sino = np.sin(0.05*t)
cosino= np.sin(0.1*t)
for i in range(120):
    sino[i]=0
#
for j in range(200,300):
     sino[j]=0
#
for z in range(390,400):
    sino[z]=0

plt.plot(0.25*(samples+1.5*sino+ 0.3*cosino)+0.5)
pylab.xlim([0, n])
pylab.ylim([-0.5, 1.5])
plt.xlabel("Frames")
# plt.ylabel("Confidence")
# plt.title("Online graph")
plt.grid()
plt.show()