
import sys
import numpy as np
import matplotlib
from nnfs.datasets import spiral_data




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

