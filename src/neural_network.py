import math
import random
from typing import List
import numpy as np
from enum import Enum

class NNState(Enum):
    UNTRAINED = 1
    FORWARD = 2
    ERROR = 3
    BACKWARD = 4
    NEW_SAMPLE = 5
    TRAINED = 7
    PREDICT_FORWARD = 8
    

class NeuralNetwork:
    def __init__(self, nn_structure: List[int], learning_rate: float, epochs_num: int, dataset: np.array, test_percent:float=.2):

        if len(nn_structure) < 2:
            raise Exception("There must be no less than 2 layers!")

        self.__nn_structure = nn_structure
        self.__epochs_num = epochs_num
        self.__train_set, self.__test_set = dataset[:int(len(dataset)*(1-test_percent))], dataset[int(len(dataset)*(1-test_percent)):] 
        
        self.__learning_rate = learning_rate

        self.__biases = []
        self.__weights = []

        self.__state = NNState.UNTRAINED
        self.__state_pre = None

        self.__layers_before_activation = [np.zeros((x, 1)) for x in self.__nn_structure]

        self.__epoch_counter = 0
        self.__train_sample_counter = 0
        self.__test_sample_counter = 0
        self.__counter = None
        self.current_layer_value = None
        self.__layers_values = [[] for x in range(len(self.__nn_structure))]
        self.__error_sum = 0.0
        self.__errors_matrix = None
        
        for i in range(len(self.__nn_structure) - 1):
            current_size = self.__nn_structure[i]  # from layer (index)
            next_size = self.__nn_structure[i + 1]  # to layer (index)

            # setup weights matrix between from/to (index) layers
            self.__weights.append(np.array([(x - 0.5) / 2 for x in np.random.rand(next_size, current_size)]))

            # setup biases matrix for (next index - skipping input layer) layers
            self.__biases.append(np.array([(x - 0.5) / 5 for x in np.random.rand(next_size, 1)]))
     
    def get_errors_matrix(self):
        return self.__errors_matrix
     
    def get_train_set(self):
        return self.__train_set
    def get_test_set(self):
        return self.__test_set
     
    def get_current_epoch(self) -> int:
        return self.__epoch_counter
    
    def get_epoch_number(self) -> int:
        return self.__epochs_num
    
    def get_current_test_sample_index(self) -> int:
        return self.__test_sample_counter

    def get_current_train_sample_index(self) -> int:
        return self.__train_sample_counter
    
    def get_learning_rate(self) -> float:
        return self.__learning_rate 
    
    def get_state(self):
        return self.__state.name, self.__counter
     
    def get_layers_with_values(self) -> List[np.array]:
        return self.__layers_values
    
    def get_structure(self) -> List[int]:
        return self.__nn_structure
            
    def get_weights(self) -> List[np.array]:
        return self.__weights
    
    def get_biases(self) -> List[np.array]:
        return self.__biases
            
    def predict_step(self, inputs: list) -> bool:
        if not self.__state in [NNState.TRAINED, NNState.NEW_SAMPLE, NNState.PREDICT_FORWARD, NNState.UNTRAINED]:
            raise Exception("Prediction can be made only on states TRAINED or NEW_SAMPLE!")
        
        if self.__state in [NNState.TRAINED, NNState.NEW_SAMPLE, NNState.UNTRAINED]:
            self.__state_pre = self.__state
            self.__counter = None
            self.current_layer_value = None
            self.__layers_values = [[] for x in range(len(self.__nn_structure))]
            
        self.__state = NNState.PREDICT_FORWARD
        self.__feedforward(inputs)
        
        return self.__state == NNState.PREDICT_FORWARD
    
    def get_prediction(self) -> list:
        if self.__state == NNState.TRAINED:
            return self.current_layer_value.transpose()[0]
        return None
        
    def train_step(self) -> bool:
        if self.__state == NNState.PREDICT_FORWARD:
            raise Exception("Cannot train while predicting!")
        
        sample = self.__train_set[self.__train_sample_counter]
        
        if self.__state == NNState.TRAINED or self.__state == NNState.PREDICT_FORWARD:
            print("Network already trained!")
            return False
        
        elif self.__state == NNState.UNTRAINED:
            random.shuffle(self.__train_set)
            self.__epoch_counter = 0
            self.__train_sample_counter = 0
            self.current_layer_value = None
            self.__layers_values = [[] for x in range(len(self.__nn_structure))]
            self.__feedforward(sample[0])
            self.__state = NNState.FORWARD
        
        elif self.__state == NNState.FORWARD:
            self.__feedforward(sample[0])
            
        elif self.__state == NNState.ERROR:
            predictions = self.current_layer_value
            expected_results = sample[1]
            
            #expected_results_transposed = np.array(expected_results).reshape(len(expected_results), 1)
            #self.__errors_matrix = expected_results_transposed - predictions
            
            self.__error_sum += calculate_cross_entropy_cost(expected_results, predictions)

            self.__state = NNState.BACKWARD
            self.__backpropagation()
            
        
        elif self.__state == NNState.BACKWARD:
            self.__backpropagation()
            
        elif self.__state == NNState.NEW_SAMPLE:
            self.__layers_values = [[] for x in range(len(self.__nn_structure))]
            self.__counter = None
            self.current_layer_value = None
            self.__errors_matrix = None
            
            if self.__epoch_counter >= self.__epochs_num:
                self.__train_sample_counter = len(self.__train_set) - 1
                self.__state = NNState.TRAINED
                print("Network trained!")
                return False
            
            if self.__train_sample_counter >= len(self.__train_set) - 1:
                print(f"Epoch {self.__epoch_counter} completed. Error: {self.__error_sum/len(self.__train_set):.3f}")
                self.__epoch_counter +=1
                self.__train_sample_counter = 0                
                self.__error_sum = 0.0
            
            self.__feedforward_init(sample[0])
            self.__state = NNState.FORWARD
            
        return True
                 
    def __feedforward_init(self, inputs: List):
        self.__layers_before_activation[0] = np.array(inputs).reshape(len(inputs), 1)
        self.current_layer_value = self.__layers_before_activation[0]
        self.__counter = 0
        self.__layers_values[self.__counter] = self.__layers_before_activation[0].copy()
    
    def __feedforward(self, inputs: List):
        if self.__counter == None: 
            self.__feedforward_init(inputs)
            return
        else:
            self.__counter += 1

        if self.__counter >= len(self.__layers_before_activation):         
            self.__counter = None
            
            if self.__state == NNState.PREDICT_FORWARD:
                self.__state = self.__state_pre
                self.__test_sample_counter += 1
                if self.__test_sample_counter >= len(self.__test_set):
                    self.__test_sample_counter = 0
            else:
                sample = self.__train_set[self.__train_sample_counter]
                predictions = self.current_layer_value
                expected_results = sample[1]
                expected_results_transposed = np.array(expected_results).reshape(len(expected_results), 1)
                self.__errors_matrix = expected_results_transposed - predictions
                self.__state = NNState.ERROR
            return
            

        index = self.__counter - 1
        
        # Multiply layer weights with its values
        multiplied_by_weights_layer = np.matmul(self.__weights[index], self.current_layer_value)

        # Add biases
        layer_with_added_biases = np.add(multiplied_by_weights_layer, self.__biases[index])
        
        # Apply activation function
        if index == len(self.__layers_before_activation) - 2:
            activated_layer = softmax(layer_with_added_biases)
        else:
            activated_layer = ReLU(layer_with_added_biases)
        
        self.__layers_before_activation[index + 1] = layer_with_added_biases
        self.current_layer_value = activated_layer
        
        self.__layers_values[self.__counter] = activated_layer
    
    
    def __backpropagation(self):
        if self.__counter == None:
            self.__counter = len(self.__weights) - 1
        else:
            self.__counter -= 1
            
        if self.__counter < 0:
            self.__counter = None
            self.__layers_values = [[] for x in range(len(self.__nn_structure))]
            self.__train_sample_counter += 1
            self.__state = NNState.NEW_SAMPLE
            return

        index = self.__counter

        # Get the derivative of activation function for each layer weighted input
        if index == len(self.__weights) - 1:
            activation_derivative_layer = softmax_derivative(self.__layers_before_activation[index + 1])
        else:
            activation_derivative_layer = ReLU_derivative(self.__layers_before_activation[index + 1])

        # Calculate the gradient
        gradient_matrix = activation_derivative_layer * self.__errors_matrix * self.__learning_rate

        # Calculate matrix with delta weights (values to change weights in given layer)
        delta_weights_matrix = np.matmul(gradient_matrix, self.__layers_before_activation[index].transpose())

        # Calculate error for next layer with respect to its weights
        self.__errors_matrix = np.matmul(self.__weights[index].transpose(), self.__errors_matrix)


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