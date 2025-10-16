import math
import random
from typing import List
import numpy as np
from enum import Enum


# ===========================================
# Neural Network State Machine
# ===========================================

class NNState(Enum):
    UNTRAINED = 1
    FORWARD = 2
    ERROR = 3
    BACKWARD = 4
    NEW_SAMPLE = 5
    TRAINED = 7
    PREDICT_FORWARD = 8


# ===========================================
# CLASS: Neural Network
# ===========================================

class NeuralNetwork:
    def __init__(self, nn_structure: List[int], learning_rate: float, epochs_num: int,
                 dataset: np.array, test_percent: float = 0.2):

        if len(nn_structure) < 2:
            raise Exception("There must be at least 2 layers!")

        # Core parameters
        self.__nn_structure = nn_structure
        self.__epochs_num = epochs_num
        self.__learning_rate = learning_rate

        # Split dataset into training and testing
        split_index = int(len(dataset) * (1 - test_percent))
        self.__train_set = dataset[:split_index]
        self.__test_set = dataset[split_index:]

        # Initialize weights, biases, and layer containers
        self.__weights = []
        self.__biases = []
        self.__layers_before_activation = [np.zeros((n, 1)) for n in nn_structure]
        self.__layers_values = [[] for _ in range(len(nn_structure))]

        # State management
        self.__state = NNState.UNTRAINED
        self.__state_pre = None
        self.__counter = None

        # Progress tracking
        self.__epoch_counter = 0
        self.__train_sample_counter = 0
        self.__test_sample_counter = 0

        # Training stats
        self.__error_sum = 0.0
        self.__errors_matrix = None
        self.current_layer_value = None

        # Initialize weights and biases
        for i in range(len(nn_structure) - 1):
            in_size = nn_structure[i]
            out_size = nn_structure[i + 1]
            self.__weights.append(np.array([(x - 0.5) / 2 for x in np.random.rand(out_size, in_size)]))
            self.__biases.append(np.array([(x - 0.5) / 5 for x in np.random.rand(out_size, 1)]))

    # ===========================================
    # GETTERS
    # ===========================================

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

    # ===========================================
    # PREDICTION STEP
    # ===========================================

    def predict_step(self, inputs: list) -> bool:
        """Performs one step of forward propagation for visualization."""
        if self.__state not in [NNState.TRAINED, NNState.NEW_SAMPLE,
                                NNState.PREDICT_FORWARD, NNState.UNTRAINED]:
            raise Exception("Prediction can only occur in TRAINED, NEW_SAMPLE, or UNTRAINED states!")

        if self.__state in [NNState.TRAINED, NNState.NEW_SAMPLE, NNState.UNTRAINED]:
            self.__state_pre = self.__state
            self.__counter = None
            self.current_layer_value = None
            self.__layers_values = [[] for _ in range(len(self.__nn_structure))]

        self.__state = NNState.PREDICT_FORWARD
        self.__feedforward(inputs)

        return self.__state == NNState.PREDICT_FORWARD


    # ===========================================
    # TRAINING STEP
    # ===========================================

    def train_step(self) -> bool:
        """Executes a single step of training visualization."""
        if self.__state == NNState.PREDICT_FORWARD:
            raise Exception("Cannot train while predicting!")

        sample = self.__train_set[self.__train_sample_counter]

        # Handle each network state
        if self.__state in [NNState.TRAINED, NNState.PREDICT_FORWARD]:
            print("Network already trained!")
            return False

        elif self.__state == NNState.UNTRAINED:
            random.shuffle(self.__train_set)
            self.__epoch_counter = 0
            self.__train_sample_counter = 0
            self.current_layer_value = None
            self.__layers_values = [[] for _ in range(len(self.__nn_structure))]
            self.__feedforward(sample[0])
            self.__state = NNState.FORWARD

        elif self.__state == NNState.FORWARD:
            self.__feedforward(sample[0])

        elif self.__state == NNState.ERROR:
            predictions = self.current_layer_value
            expected_results = sample[1]
            self.__error_sum += calculate_cross_entropy_cost(expected_results, predictions)
            self.__state = NNState.BACKWARD
            self.__backpropagation()

        elif self.__state == NNState.BACKWARD:
            self.__backpropagation()

        elif self.__state == NNState.NEW_SAMPLE:
            self.__reset_for_new_sample()

            if self.__epoch_counter >= self.__epochs_num:
                self.__state = NNState.TRAINED
                print("Network trained!")
                return False

            self.__feedforward_init(sample[0])
            self.__state = NNState.FORWARD

        return True

    # ===========================================
    # INTERNAL METHODS
    # ===========================================

    def __reset_for_new_sample(self):
        """Resets buffers when moving to a new sample."""
        self.__layers_values = [[] for _ in range(len(self.__nn_structure))]
        self.__counter = None
        self.current_layer_value = None
        self.__errors_matrix = None

        if self.__train_sample_counter >= len(self.__train_set) - 1:
            print(f"Epoch {self.__epoch_counter} completed. "
                  f"Error: {self.__error_sum / len(self.__train_set):.3f}")
            self.__epoch_counter += 1
            self.__train_sample_counter = 0
            self.__error_sum = 0.0

    def __feedforward_init(self, inputs: List):
        """Initializes the first forward pass."""
        self.__layers_before_activation[0] = np.array(inputs).reshape(len(inputs), 1)
        self.current_layer_value = self.__layers_before_activation[0]
        self.__counter = 0
        self.__layers_values[self.__counter] = self.__layers_before_activation[0].copy()

    def __feedforward(self, inputs: List):
        """Executes one feedforward step."""
        if self.__counter is None:
            self.__feedforward_init(inputs)
            return
        else:
            self.__counter += 1

        if self.__counter >= len(self.__layers_before_activation):
            self.__finish_feedforward()
            return

        index = self.__counter - 1

        # Weighted sum + bias
        z = np.matmul(self.__weights[index], self.current_layer_value)
        z = np.add(z, self.__biases[index])

        # Activation
        if index == len(self.__layers_before_activation) - 2:
            activated = softmax(z)
        else:
            activated = ReLU(z)

        self.__layers_before_activation[index + 1] = z
        self.current_layer_value = activated
        self.__layers_values[self.__counter] = activated

    def __finish_feedforward(self):
        """Finalizes feedforward and transitions state."""
        self.__counter = None
        if self.__state == NNState.PREDICT_FORWARD:
            self.__state = self.__state_pre
            self.__test_sample_counter = (self.__test_sample_counter + 1) % len(self.__test_set)
        else:
            sample = self.__train_set[self.__train_sample_counter]
            predictions = self.current_layer_value
            expected = np.array(sample[1]).reshape(len(sample[1]), 1)
            self.__errors_matrix = expected - predictions
            self.__state = NNState.ERROR

    def __backpropagation(self):
        """Executes one backpropagation step."""
        if self.__counter is None:
            self.__counter = len(self.__weights) - 1
        else:
            self.__counter -= 1

        if self.__counter < 0:
            self.__finalize_backprop()
            return

        index = self.__counter
        activation_derivative = (
            softmax_derivative(self.__layers_before_activation[index + 1])
            if index == len(self.__weights) - 1
            else ReLU_derivative(self.__layers_before_activation[index + 1])
        )

        gradient = activation_derivative * self.__errors_matrix * self.__learning_rate
        delta_weights = np.matmul(gradient, self.__layers_before_activation[index].transpose())
        self.__errors_matrix = np.matmul(self.__weights[index].transpose(), self.__errors_matrix)

        self.__weights[index] += delta_weights
        self.__biases[index] += gradient

    def __finalize_backprop(self):
        """Handles completion of backpropagation for one sample."""
        self.__counter = None
        self.__layers_values = [[] for _ in range(len(self.__nn_structure))]
        self.__train_sample_counter += 1
        self.__state = NNState.NEW_SAMPLE


# ===========================================
# HELPER FUNCTIONS
# ===========================================

def calculate_cross_entropy_cost(expected_values, real_values):
    """Compute cross-entropy cost between expected and real outputs."""
    return -sum(e * math.log(r) for e, r in zip(expected_values, real_values))


# ===========================================
# ACTIVATION FUNCTIONS
# ===========================================

def softmax(x):
    tmp = np.exp(x)
    return tmp / np.sum(tmp)


def ReLU(x):
    return x * (x > 0)


def softmax_derivative(x):
    tmp = softmax(x)
    return tmp * (1 - tmp)


def ReLU_derivative(x):
    return 1. * (x >= 0)
