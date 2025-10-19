import math
import random
from typing import List
import numpy as np
from enum import Enum


# -------------------------------------- Neural Network State Machine -------------------------------------- #

class NNState(Enum):
    UNTRAINED = 1          # Network has been initialized but not yet trained
    FORWARD = 2            # Currently performing forward propagation on a training sample
    ERROR = 3              # Error computed after forward propagation (before backpropagation)
    BACKWARD = 4           # Currently performing backpropagation to update weights
    NEW_SAMPLE = 5         # Moving to the next training sample (preparing buffers)
    TRAINED = 7            # Training is complete; network is fully trained
    PREDICT_FORWARD = 8    # Performing forward propagation for testing/prediction



# -------------------------------------- Neural Network Class -------------------------------------- #

class NeuralNetwork:
    def __init__(self, nn_structure: List[int], learning_rate: float, epochs_num: int,
                 dataset: np.array, test_percent: float = 0.2):
        if len(nn_structure) < 2:
            raise Exception("There must be at least 2 layers!")  # Validate network structure

        # Core hyperparameters
        self.__nn_structure = nn_structure          # Number of neurons per layer
        self.__epochs_num = epochs_num              # Total epochs for training
        self.__learning_rate = learning_rate        # Learning rate for weight updates

        # Split dataset into training and testing subsets
        split_index = int(len(dataset) * (1 - test_percent))
        self.__train_set = dataset[:split_index]
        self.__test_set = dataset[split_index:]

        # Initialize network storage
        self.__weights = []                          # List of weight matrices per layer
        self.__biases = []                           # List of bias vectors per layer
        self.__layers_before_activation = [np.zeros((n, 1)) for n in nn_structure]  # Linear activations
        self.__layers_values = [[] for _ in range(len(nn_structure))]               # Post-activation values

        # State management
        self.__state = NNState.UNTRAINED            # Current state of the network
        self.__state_pre = None                     # Previous state (used in prediction)
        self.__counter = None                       # Step counter for forward/backward

        # Training progress tracking
        self.__epoch_counter = 0
        self.__train_sample_counter = 0
        self.__test_sample_counter = 0

        # Training statistics
        self.__error_sum = 0.0
        self.__errors_matrix = None
        self.current_layer_value = None

        # Randomly initialize weights and biases for each layer
        for i in range(len(nn_structure) - 1):
            in_size = nn_structure[i]
            out_size = nn_structure[i + 1]
            self.__weights.append(np.array([(x - 0.5) / 2 for x in np.random.rand(out_size, in_size)]))
            self.__biases.append(np.array([(x - 0.5) / 5 for x in np.random.rand(out_size, 1)]))


# -------------------------------------- GETTERS -------------------------------------- #

    def get_errors_matrix(self):
        """Return the current errors matrix (difference between expected and predicted outputs)."""
        return self.__errors_matrix

    def get_train_set(self):
        """Return the training dataset."""
        return self.__train_set

    def get_test_set(self):
        """Return the testing dataset."""
        return self.__test_set

    def get_current_epoch(self) -> int:
        """Return the current epoch number (0-based)."""
        return self.__epoch_counter

    def get_epoch_number(self) -> int:
        """Return the total number of epochs configured for training."""
        return self.__epochs_num

    def get_current_test_sample_index(self) -> int:
        """Return the index of the current test sample being processed."""
        return self.__test_sample_counter

    def get_current_train_sample_index(self) -> int:
        """Return the index of the current training sample being processed."""
        return self.__train_sample_counter

    def get_learning_rate(self) -> float:
        """Return the learning rate used for training."""
        return self.__learning_rate

    def get_state(self):
        """Return a tuple of the current network state name and the step counter."""
        return self.__state.name, self.__counter

    def get_layers_with_values(self) -> List[np.array]:
        """Return a list of arrays representing each layer's post-activation values."""
        return self.__layers_values

    def get_structure(self) -> List[int]:
        """Return the neural network structure as a list of neuron counts per layer."""
        return self.__nn_structure

    def get_weights(self) -> List[np.array]:
        """Return the list of weight matrices for each layer."""
        return self.__weights

    def get_biases(self) -> List[np.array]:
        """Return the list of bias vectors for each layer."""
        return self.__biases

# -------------------------------------- PREDICTION / TRAINING -------------------------------------- #

    def predict_step(self, inputs: list) -> bool:
        """
        Perform one forward propagation step for the given input sample.

        This method is mainly used for visualization, stepping through the network layer by layer.
        It supports states: UNTRAINED, TRAINED, NEW_SAMPLE, or PREDICT_FORWARD.

        Parameters:
            inputs (list): Input vector for the network.

        Returns:
            bool: True if the network is in PREDICT_FORWARD state after this step.
        """
        # Ensure prediction is allowed only in specific states
        if self.__state not in [NNState.TRAINED, NNState.NEW_SAMPLE,
                                NNState.PREDICT_FORWARD, NNState.UNTRAINED]:
            raise Exception("Prediction can only occur in TRAINED, NEW_SAMPLE, or UNTRAINED states!")

        # Reset temporary buffers if starting a new prediction sequence
        if self.__state in [NNState.TRAINED, NNState.NEW_SAMPLE, NNState.UNTRAINED]:
            self.__state_pre = self.__state     # store previous state
            self.__counter = None               # reset layer counter
            self.current_layer_value = None     # clear current layer value
            self.__layers_values = [[] for _ in range(len(self.__nn_structure))]  # clear layers values

        # Set state to prediction mode and perform feedforward
        self.__state = NNState.PREDICT_FORWARD
        self.__feedforward(inputs)

        return self.__state == NNState.PREDICT_FORWARD


    def train_step(self) -> bool:
        """
        Execute a single step of training visualization.

        This method handles the training of one sample per call and updates the network state
        to reflect forward propagation, error computation, backpropagation, or moving to the next sample.
        
        Returns:
            bool: True if training can continue, False if network is already trained.
        """
        # Training is not allowed during prediction
        if self.__state == NNState.PREDICT_FORWARD:
            raise Exception("Cannot train while predicting!")

        # Get the current sample for training
        sample = self.__train_set[self.__train_sample_counter]

        # ==============================
        # Handle states individually
        # ==============================

        # Network already trained: nothing to do
        if self.__state in [NNState.TRAINED, NNState.PREDICT_FORWARD]:
            print("Network already trained!")
            return False

        # UNTRAINED state: initialize network for first sample
        elif self.__state == NNState.UNTRAINED:
            random.shuffle(self.__train_set)    # shuffle training set for randomness
            self.__epoch_counter = 0            # reset epoch counter
            self.__train_sample_counter = 0     # reset sample counter
            self.current_layer_value = None     # clear current layer activations
            self.__layers_values = [[] for _ in range(len(self.__nn_structure))]  # clear stored layer outputs
            self.__feedforward(sample[0])       # compute forward pass for first sample
            self.__state = NNState.FORWARD

        # FORWARD state: continue forward propagation
        elif self.__state == NNState.FORWARD:
            self.__feedforward(sample[0])

        # ERROR state: compute error and prepare for backpropagation
        elif self.__state == NNState.ERROR:
            predictions = self.current_layer_value
            expected_results = sample[1]
            self.__error_sum += calculate_cross_entropy_cost(expected_results, predictions)
            self.__state = NNState.BACKWARD
            self.__backpropagation()

        # BACKWARD state: execute one backpropagation step
        elif self.__state == NNState.BACKWARD:
            self.__backpropagation()

        # NEW_SAMPLE state: reset buffers and start next sample or epoch
        elif self.__state == NNState.NEW_SAMPLE:
            self.__reset_for_new_sample()

            # If all epochs completed, mark network as trained
            if self.__epoch_counter >= self.__epochs_num:
                self.__state = NNState.TRAINED
                self.__train_sample_counter = len(self.__train_set)-1
                print("Network trained!")
                return False

            # Start forward pass for the new sample
            self.__feedforward_init(sample[0])
            self.__state = NNState.FORWARD

        return True

# -------------------------------------- INTERNAL METHODS-------------------------------------- #

    def __reset_for_new_sample(self):
        """
        Reset temporary buffers and counters when transitioning to a new training sample.

        This prepares the network for the next forward pass:
        - Clears stored layer outputs.
        - Resets the forward/backward counter.
        - Clears the current activations.
        - Clears the error matrix from the previous sample.

        Additionally, if the current sample was the last in the training set:
        - Prints the average error for the epoch.
        - Increments the epoch counter.
        - Resets the sample counter for the new epoch.
        - Resets the cumulative error sum.
        """
        self.__layers_values = [[] for _ in range(len(self.__nn_structure))]
        self.__counter = None
        self.current_layer_value = None
        self.__errors_matrix = None

        # Check if the current sample was the last in the training set
        if self.__train_sample_counter >= len(self.__train_set) - 1:
            avg_error = self.__error_sum / len(self.__train_set)
            print(f"Epoch {self.__epoch_counter} completed. Error: {avg_error:.3f}")
            self.__epoch_counter += 1
            self.__train_sample_counter = 0
            self.__error_sum = 0.0


    def __feedforward_init(self, inputs: List):
        """
        Initializes the first forward pass for a new sample.

        - Converts input list to a column vector and stores it in the first layer.
        - Sets current_layer_value to the input layer for use in the next step.
        - Initializes the counter to track which layer is being processed.
        - Copies the input values into layers_values for visualization.
        """
        self.__layers_before_activation[0] = np.array(inputs).reshape(len(inputs), 1)
        self.current_layer_value = self.__layers_before_activation[0]
        self.__counter = 0
        self.__layers_values[self.__counter] = self.__layers_before_activation[0].copy()


    def __feedforward(self, inputs: List):
        """
        Executes one step of forward propagation through the network.
        - If counter is None, initialize feedforward with input.
        - Otherwise, increment the counter to move to the next layer.
        - Computes weighted sum plus bias (z = W·x + b).
        - Applies activation function: ReLU for hidden layers, softmax for output layer.
        - Updates the layer buffers and current_layer_value.
        - If the final layer is reached, calls __finish_feedforward to complete the pass.
        """
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
        if index == len(self.__layers_before_activation) - 2:  # output layer
            activated = softmax(z)
        else:  # hidden layers
            activated = ReLU(z)

        self.__layers_before_activation[index + 1] = z
        self.current_layer_value = activated
        self.__layers_values[self.__counter] = activated


    def __finish_feedforward(self):
        """
        Completes the feedforward step and transitions the network state.

        - Resets the counter for the next pass.
        - If predicting, restores previous state and advances the test sample counter.
        - If training, calculates the error matrix (expected - predicted) and sets state to ERROR for the backpropagation step.
        """
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
        """
        Executes one step of backpropagation for a single training sample.

        1. If counter is None, initialize it to the last weight layer (output layer). Otherwise, move one layer backward.
        2. If counter < 0, backpropagation is complete for this sample, call __finalize_backprop.
        3. Compute the derivative of the activation function:
            - Softmax derivative for output layer
            - ReLU derivative for hidden layers
        4. Calculate the gradient: element-wise multiplication of activation derivative, errors, and learning rate.
        5. Compute delta_weights using outer product of gradient and previous layer activations.
        6. Propagate errors backward through the network for the next layer.
        7. Update the weights and biases with the calculated gradients.
        """
        if self.__counter is None:
            self.__counter = len(self.__weights) - 1  # start from output layer
        else:
            self.__counter -= 1  # move to previous layer

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

        # Propagate errors to the previous layer
        self.__errors_matrix = np.matmul(self.__weights[index].transpose(), self.__errors_matrix)

        # Update weights and biases
        self.__weights[index] += delta_weights
        self.__biases[index] += gradient


    def __finalize_backprop(self):
        """
        Finalizes backpropagation for a single training sample.

        - Resets the counter to None to prepare for the next sample.
        - Clears layer activations for the next sample.
        - Increments the training sample counter.
        - Sets network state to NEW_SAMPLE to trigger initialization for the next sample.
        """
        self.__counter = None
        self.__layers_values = [[] for _ in range(len(self.__nn_structure))]
        self.__train_sample_counter += 1
        self.__state = NNState.NEW_SAMPLE



# -------------------------------------- HELPER FUNCTIONS-------------------------------------- #

def calculate_cross_entropy_cost(expected_values, real_values):
    """
    Computes the cross-entropy loss between expected (true) values and predicted outputs.

    Parameters:
        expected_values (list or array): True one-hot encoded target values.
        real_values (list or array): Predicted probabilities from the network (output of softmax).

    Returns:
        float: The cross-entropy cost for the sample.
    
    Notes:
        - Assumes expected_values and real_values have the same length.
        - Does not include batch averaging; this computes cost for a single sample.
    """
    epsilon = 1e-12
    return -sum(e * math.log(max(min(r, 1-epsilon), epsilon)) for e, r in zip(expected_values, real_values))


def softmax(x):
    """
    Softmax activation function.
    
    Args:
        x (np.array): Input vector (logits).

    Returns:
        np.array: Probability distribution over classes.
    """
    tmp = np.exp(x)
    return tmp / np.sum(tmp)


def ReLU(x):
    """
    Rectified Linear Unit (ReLU) activation function.
    
    Outputs x if x > 0, otherwise 0.
    
    Args:
        x (np.array): Input array.

    Returns:
        np.array: Element-wise ReLU output.
    """
    return x * (x > 0)


def softmax_derivative(x):
    """
    Simplified derivative of the softmax function.
    
    Used during backpropagation. This simplified version assumes
    that softmax is paired with cross-entropy loss, which allows
    the gradient to be computed element-wise as s * (1 - s).

    Args:
        x (np.array): Input vector (logits).

    Returns:
        np.array: Element-wise derivative approximation.
    """
    tmp = softmax(x)
    return tmp * (1 - tmp)


def ReLU_derivative(x):
    """
    Derivative of the ReLU function.
    
    Used during backpropagation. Returns 1 where x >= 0, 0 otherwise.

    Args:
        x (np.array): Input array.

    Returns:
        np.array: Element-wise derivative of ReLU.
    """
    return 1. * (x >= 0)