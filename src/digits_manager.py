from sklearn.datasets import load_digits
import numpy as np


class DigitsDataset(object):
    def __init__(self):
        # Load the built-in digits dataset from scikit-learn
        digits = load_digits()
        X, y_org = digits.data, digits.target

        # Normalize input features to the range [0, 1]
        X = X / 16.0

        # Convert integer labels (0–9) into one-hot encoded vectors
        y = []
        for label in y_org:
            one_hot = np.zeros(10)
            one_hot[label] = 1
            y.append(one_hot)
        
        # Combine normalized inputs and one-hot encoded labels into a single dataset
        self.__dataset = list(zip(X, y))
    
    def get_dataset(self) -> np.array:
        """
        Return the dataset as a list of (input, target) tuples.
        Each input is a normalized 64-dimensional vector,
        and each target is a one-hot encoded vector of length 10.
        """
        return self.__dataset
    
    def get_label(self, encoded_target: np.array) -> np.int64:
        """
        Convert a one-hot encoded target vector back to its numeric label.
        Example: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] -> 2
        """
        return np.argmax(encoded_target)
