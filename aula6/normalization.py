

#como normalizar os resultados sem perder o significado dos outputs negativos com RELU
#ponteica
import math

layer_outputs = [4.8, 1.21, 2.385]
E = math.e
exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)

print(exp_values)

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value/norm_base)

print(norm_values)
print(sum(norm_values))

#forma alternativa e eficiente

import numpy as np

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values) #valores normalizados

print(sum(norm_values))
