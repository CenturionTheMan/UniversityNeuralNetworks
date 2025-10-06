import math
import random
from typing import List
import numpy as np


class NeuralNetwork:
    def __init__(self, nn_structure: List[int], learning_rate: float, epochs_num: int, dataset: np.array):

        if len(nn_structure) < 2:
            raise Exception("There must be no less than 2 layers!")

        self.nn_structure = nn_structure
        self.epochs_num = epochs_num
        self.dataset = dataset
        
        self.__learning_rate = learning_rate

        self.__biases = []
        self.__weights = []

        self.__layers_before_activation = [np.zeros((x, 1)) for x in nn_structure]


        for i in range(len(nn_structure) - 1):
            current_size = nn_structure[i]  # from layer (index)
            next_size = nn_structure[i + 1]  # to layer (index)

            # setup weights matrix between from/to (index) layers
            self.__weights.append(np.array([(x - 0.5) / 2 for x in np.random.rand(next_size, current_size)]))

            # setup biases matrix for (next index - skipping input layer) layers
            self.__biases.append(np.array([(x - 0.5) / 5 for x in np.random.rand(next_size, 1)]))
            
    def predict(self, inputs: list) -> list:
        """
        This function will give prediction about outputs for given input.
        IMPORTANT: inputs should be normalized (between 0 and 1).
        Note: network should be first trained.
        :param inputs: Data point to predict results for
        :return: Neural network predictions
        """
        predictions = self.__feedforward(inputs)
        raw = predictions.transpose()[0]

        return list(raw)
            
    def train(self):
        for epoch in range(self.epochs_num):
            error_sum = 0
            random.shuffle(self.dataset)
            
            for idx, sample in enumerate(self.dataset):
                input_point = sample[0]
                expected = sample[1]
                
                # forward
                prediction = self.__feedforward(input_point)

                # backward
                self.__backpropagation(expected, prediction)
                
                error_sum += calculate_cross_entropy_cost(expected, prediction)

            print(f"Epoch {epoch + 1}/{self.epochs_num} - Error: {error_sum/len(self.dataset)}")

                
                 
                 
    def __feedforward(self, inputs: List) -> np.array:
        self.__layers_before_activation[0] = np.array(inputs).reshape(len(inputs), 1)
        current_layer_value = self.__layers_before_activation[0]
        
        
        # TODO try to put this loop to sleep after each iteration and update gui, maybe some event setup? Prob will need to rebuild that loop into step/save action with state machine 
        for index in range(len(self.__layers_before_activation) - 1):
            # Multiply layer weights with its values
            multiplied_by_weights_layer = np.matmul(self.__weights[index], current_layer_value)

            # Add biases
            layer_with_added_biases = np.add(multiplied_by_weights_layer, self.__biases[index])
            
            # Apply activation function
            if index == len(self.__layers_before_activation) - 2:
                activated_layer = softmax(layer_with_added_biases)
            else:
                activated_layer = ReLU(layer_with_added_biases)
            
            self.__layers_before_activation[index + 1] = layer_with_added_biases
            current_layer_value = activated_layer
            
        return current_layer_value 
    
    
    def __backpropagation(self, expected_results: list, predictions: list):
        # Check if expected results array size matches the output layer size
        if np.shape(expected_results) != (self.nn_structure[-1],):
            raise Exception(f'Wrong result array size! Should be {(self.nn_structure[-1],)} and was '
                            f'{np.shape(expected_results)}')

        # Prepare expected results list
        expected_results_transposed = np.array(expected_results).reshape(len(expected_results), 1)

        # Initialize error matrix with output layer error
        errors_matrix = expected_results_transposed - predictions

        # Iterate over each weight / bias matrix in reverse order
        # TODO might be also needed to swap into state machine
        for index in reversed(range(len(self.__weights))):
            
            # Get the derivative of activation function for each layer weighted input
            if index == len(self.__weights) - 1:
                activation_derivative_layer = softmax_derivative(self.__layers_before_activation[index + 1])
            else:
                activation_derivative_layer = ReLU_derivative(self.__layers_before_activation[index + 1])

            # Calculate the gradient
            gradient_matrix = activation_derivative_layer * errors_matrix * self.__learning_rate

            # Calculate matrix with delta weights (values to change weights in given layer)
            delta_weights_matrix = np.matmul(gradient_matrix, self.__layers_before_activation[index].transpose())

            # Calculate error for next layer with respect to its weights
            errors_matrix = np.matmul(self.__weights[index].transpose(), errors_matrix)


            # Adjust weights and biases
            self.__weights[index] = self.__weights[index] + delta_weights_matrix
            self.__biases[index] = self.__biases[index] + gradient_matrix
                        
            
def calculate_cross_entropy_cost(expected_values, real_values):
    """
    Calculate the cross-entropy cost between expected and real values.

    :param expected_values: The expected values.
    :param real_values: The real values.
    :return: The cross-entropy cost.
    """
    val_sum = 0
    for expected, real in zip(expected_values, real_values):
        val_sum += expected * math.log(real)
    return -val_sum
            
def softmax(x):
    """
    Compute the softmax activation function.

    :param x: Input values.
    :return: Softmax output.
    """
    tmp = np.exp(x)
    return tmp / np.sum(tmp)


def ReLU(x):
    """
    Compute the ReLU (Rectified Linear Unit) activation function.

    :param x: Input values.
    :return: ReLU output.
    """
    return x * (x > 0)


def softmax_derivative(x):
    """
    Compute the derivative of the softmax activation function.

    :param x: Input values.
    :return: Derivative of softmax.
    """
    tmp = softmax(x)
    return tmp * (1 - tmp)


def ReLU_derivative(x):
    """
    Compute the derivative of the ReLU activation function.

    :param x: Input values.
    :return: Derivative of ReLU.
    """
    return 1. * (x >= 0)