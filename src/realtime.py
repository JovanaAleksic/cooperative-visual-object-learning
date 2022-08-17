import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 10, 0, 1])
plt.ion()

y1=[0, 0]

for i in range(100):
    y= np.random.random()
    y1 = [y1[1], y]
    i1 = [i-1, i]
    plt.plot(i1, y1, '-bo')
    plt.pause(0.05)
    if i == 5:
        answer = input("Something")
        if answer == 1:
            break

plt.show(block=False)



