from sklearn.datasets import load_digits
import numpy as np


class DigitsDataset(object):
    def __init__(self):
        digits = load_digits()
        X, y_org = digits.data, digits.target

        X = X / 16.0
        y = []
        for label in y_org:
            one_hot = np.zeros(10)
            one_hot[label] = 1
            y.append(one_hot)
        
        self.__dataset = list(zip(X, y))
    
    def get_dataset(self) -> np.array:
        return np.array(self.__dataset)
    
    def get_label(self, encoded_target: np.array) -> np.int64:
        return np.argmax(encoded_target)
    