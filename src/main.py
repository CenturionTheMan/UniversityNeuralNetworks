from configuratorWindow import ConfiguratorWindow
from neural_network import NeuralNetwork
import tkinter as tk
from digits_manager import DigitsDataset
import numpy as np

if __name__ == "__main__":
    root = tk.Tk()
    gui = ConfiguratorWindow(root)
    root.mainloop() 