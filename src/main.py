from mainWindow import MainWindow
from neural_network import NeuralNetwork
import tkinter as tk
from digits_manager import DigitsDataset

if __name__ == "__main__":
    digitsCls = DigitsDataset()
    dataset = digitsCls.get_dataset()
    

    
    
    #root = tk.Tk()
    #gui = MainWindow(root)
    #root.mainloop() 