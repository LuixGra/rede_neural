import sys
import numpy as np
import matplotlib


inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]
'''
output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
print(output)

'''

layer_outputs = [] #output da camada atual
for neuron_weights, neuron_bias in zip(weights, biases): #atribui neuron weights a weights, neuron bias à biases
    neuron_output = 0 #inicio do somatorio
    for n_input, weight in zip(inputs, neuron_weights):#numero de inputs, weights
        neuron_output += n_input*weight #weight variavel atual somantorio 
    neuron_output += neuron_bias #biases
    layer_outputs.append(neuron_output)

print(layer_outputs)