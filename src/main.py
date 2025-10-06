from mainWindow import MainWindow
from neural_network import NeuralNetwork
import tkinter as tk
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    digits = load_digits()
    X, y_org = digits.data, digits.target
    print(f"Dataset shape: {X.shape}")
    
    # Normalize data
    X = X / 16.0
    y = []
    for label in y_org:
        one_hot = np.zeros(10)
        one_hot[label] = 1
        y.append(one_hot)
    dataset = list(zip(X, y))
    
    
    nn = NeuralNetwork(nn_structure=[64, 32, 10], learning_rate=0.01, epochs_num=10, dataset=dataset)
    nn.train()
    
    accuracy = 0.0
    for d, t in dataset:
        prediction = nn.predict(d)
        
        target_label = np.argmax(t)
        prediction_label = np.argmax(prediction)
        
        if target_label == prediction_label:
            accuracy += 1
    print(f"Accuracy: {accuracy/len(dataset)*100}%")
    
    
    #root = tk.Tk()
    #gui = MainWindow(root)
    #root.mainloop() 