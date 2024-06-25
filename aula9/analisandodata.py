
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

X,y = vertical_data(samples=100, classes=3)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):

        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

#categorizando com log a porcentagem de erro
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_Clipped = np.clip(y_pred, 1e-7, 1-1e-7)#clipando a memoria caso exista 0

        if len(y_true.shape)==1:
            correct_confidences = y_pred_Clipped[range(samples), y_true]
        elif len(y_true.shape)==2:
            correct_confidences = np.sum(y_pred_Clipped*y_true, axis=1)

        negative_log_likehoods = -np.log(correct_confidences)
        return negative_log_likehoods



dense1 = Layer_Dense(2, 3) #2 dimensoes x e y
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
dense2 = Layer_Dense(3, 3)

loss_function = Loss_CategoricalCrossentropy()
lowest_loss = 9999999
d1w = dense1.weights.copy()
d1b = dense1.biases.copy()
d2w = dense2.weights.copy()
d2b = dense2.biases.copy()


for iteration in range(100000):

    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3,3)
    dense2.biases+= 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions==y)

    if loss < lowest_loss:
        print('Novo set de weights encontrado, iteracao:', iteration, 'perda:', loss, 'precisao:',accuracy)
        d1w = dense1.weights.copy()
        d1b = dense1.biases.copy()
        d2w = dense2.weights.copy()
        d2b = dense2.biases.copy()

        lowest_loss = loss
    else:
        dense1.weights = d1w.copy()
        dense1.biases = d1b.copy()
        dense2.weights = d2w.copy()
        dense2.biases = d2b.copy()

plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap='brg')
plt.show()