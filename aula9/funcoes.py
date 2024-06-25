
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x**2;

p2_delta = 0.0001

x = np.arange(0,50,0.001)
y = f(x)

plt.plot(x,y)


plt.show()


