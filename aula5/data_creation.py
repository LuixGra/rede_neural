import sys
import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data




def create_data(points, classes):
    x = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points) #area
        t = np.linspace(class_number*4,(class_number+1)*4, points) + np.random.randn(points)*0.2
        x[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return x,y

import matplotlib.pyplot as plt

x, y = create_data(100, 3)
plt.scatter(x[:,0],x[:,1])
plt.show()

plt.scatter(x[:,0], x[:,1], c=y, cmap="brg")
plt.show()

