from configuratorWindow import ConfiguratorWindow
from neural_network import NeuralNetwork
import tkinter as tk
from digits_manager import DigitsDataset
import numpy as np

def test_state_nn():
    digitsCls = DigitsDataset()
    dataset = digitsCls.get_dataset()
    
    nn = NeuralNetwork(nn_structure=[64, 16, 8, 10], learning_rate=0.01, epochs_num=15, dataset=dataset)
    
    con_training = True
    while con_training:
        con_training = nn.train_step()


    accuracy_sum = 0
    for idx, (d, t) in enumerate(dataset):
        
        pred_con = True
        while pred_con:
            pred_con = nn.predict_step(d)
            print(f"[idx={idx}] Predicting...")
        
        prediction = nn.get_prediction()
        
        target_label = np.argmax(t)
        pred_label = np.argmax(prediction)
        
        if target_label == pred_label:
            accuracy_sum +=1
            
    print(f"Accuracy: {accuracy_sum}/{len(dataset)} = {accuracy_sum/len(dataset)*100:.2f}%")

if __name__ == "__main__":
    # test_state_nn()
    root = tk.Tk()
    gui = ConfiguratorWindow(root)
    root.mainloop() 