from configuratorWindow import ConfiguratorWindow
from neural_network import NeuralNetwork
import tkinter as tk
from digits_manager import DigitsDataset
import numpy as np



def test_state_nn():
    digitsCls = DigitsDataset()
    dataset = digitsCls.get_dataset()
    
    test_percent = 0.2
    train, test = dataset[:int(len(dataset)*(1-test_percent))], dataset[int(len(dataset)*(1-test_percent)):]    
    
    nn = NeuralNetwork(nn_structure=[64, 16, 8, 10], learning_rate=0.01, epochs_num=15, dataset=train)
    
    con_training = True
    while con_training:
        con_training = nn.train_step()


    accuracy_sum = 0
    for idx, (d, t) in enumerate(test):
        
        pred_con = True
        while pred_con:
            pred_con = nn.predict_step(d)
            print(f"[idx={idx}] Predicting...")
        
        prediction = nn.get_prediction()
        
        target_label = np.argmax(t)
        pred_label = np.argmax(prediction)
        
        if target_label == pred_label:
            accuracy_sum +=1
            
    print(f"Accuracy: {accuracy_sum}/{len(test)} = {accuracy_sum/len(test)*100:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    gui = ConfiguratorWindow(root)
    root.mainloop() 
    # test_state_nn()