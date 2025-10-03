import math
import random
from typing import List
import numpy as np


class NeuralNetwork:
    def __init__(self, nn_structure: List[int], learning_rate, epochs_num, data_set):

        if len(nn_structure) < 2:
            raise Exception("There must be no less than 2 layers!")

        self.nn_structure = nn_structure
        self.epochs_num = epochs_num
        self.data_set = data_set
        
        self.learning_rate = learning_rate

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
            
            
            
    def train(self):
        for epoch in self.epochs_num:
            for idx, sample in self.data_set:
                # forward
                #error calc
                # backward
                pass
                 
                 
        