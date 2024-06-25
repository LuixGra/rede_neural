import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)

norm_values = exp_values /np.sum(layer_outputs, axis=1, keepdims=True) #fazendo a soma dos valores em linhas, não em colunas

print(norm_values)