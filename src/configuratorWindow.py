import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from animationWindow import AnimationWindow
from neural_network import NeuralNetwork
from digits_manager import DigitsDataset
from style import *

class ConfiguratorWindow:
    def __init__(self, root):
        digitsCls = DigitsDataset()
        self.dataset = digitsCls.get_dataset()
        self.layers = []
        
        self.root = root
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.title("Neural Network Configurator")
        self.root.minsize(600, 200)
        self.root.resizable(True, False)
        self.root.configure(bg=COL_BACKGROUND)
        
        self.style = ttk.Style()
        configure_styles(self.style)
        self.root.style = self.style
        
        self.main_frame = ttk.Frame(root, padding=10, style="BG.TFrame")
        self.main_frame.pack(fill="both", expand=True)
        self.main_frame.grid_columnconfigure(0, weight=2)
        self.main_frame.grid_columnconfigure(1, weight=1)
        
        self.create_left_row()
        self.create_right_row()
        
        self.add_layer()
    
    def create_left_row(self):
        self.left_frame = ttk.Frame(self.main_frame, borderwidth=2, relief="groove", padding=10, style="BG.TFrame")
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Label(self.left_frame, text="Neural Network structure", style="Title.TLabel" ,
                  font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # Input 
        self.input_scale = self.create_fixed_layer(self.left_frame, row=1, label="Input layer:", neurons=64)

        # Frame for hidden
        self.hidden_layers_frame = ttk.Frame(self.left_frame, style="BG.TFrame")
        self.hidden_layers_frame.grid(row=2, column=0, columnspan=3, pady=5, sticky="ew")

        # Output 
        self.output_scale = self.create_fixed_layer(self.left_frame, row=3, label="Output layer:", neurons=10)

        # Add layer button
        self.add_layer_button = ttk.Button(self.left_frame, text="Add Hidden Layer", command=self.add_layer)
        self.add_layer_button.grid(row=4, column=0, columnspan=3, pady=10)

    def create_right_row(self):
        self.right_frame = ttk.Frame(self.main_frame, borderwidth=2, relief="groove", padding=10, style="BG.TFrame")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        ttk.Label(self.right_frame, text="Controls", style="Title.TLabel", font=("Arial", 12, "bold")).pack(pady=5)

        self.run_button = ttk.Button(self.right_frame, text="Run animation", command=self.run_animation)
        self.run_button.pack(pady=10)
        
    

    def create_fixed_layer(self, parent, row, label, neurons=8):
        ttk.Label(parent, text=label, width=12).grid(row=row, column=0, sticky="w", padx=5, pady=3)

        scale = ttk.Scale(parent, from_=1, to=64, orient="horizontal", style="TScale")
        scale.set(neurons)
        scale.state(["disabled"])
        scale.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        parent.grid_columnconfigure(1, weight=1)

        value_label = ttk.Label(parent, text=str(neurons), width=4, relief="sunken", anchor="center")
        value_label.grid(row=row, column=2, padx=5)

        return scale

    def add_layer(self):
        """Add a hidden layer dynamically"""
        idx = len(self.layers) + 1
        frame = ttk.Frame(self.hidden_layers_frame)
        frame.pack(fill="x", pady=3)

        label = ttk.Label(frame, text=f"Hidden {idx}:", width=12)
        label.pack(side="left", padx=5)

        neuron_var = tk.IntVar(value=8)
        scale = ttk.Scale(frame, from_=1, to=32, orient="horizontal", variable=neuron_var, style="TScale")
        scale.pack(side="left", fill="x", expand=True, padx=5)

        value_label = ttk.Label(frame, textvariable=neuron_var, width=4, relief="sunken", anchor="center")
        value_label.pack(side="left", padx=5)

        remove_btn = ttk.Button(frame, text="Remove", command=lambda f=frame: self.remove_layer(f))
        remove_btn.pack(side="left", padx=5)

        # scale fixed on ints
        scale.config(command=lambda v, var=neuron_var: var.set(round(float(v))))

        self.layers.append((frame, scale, neuron_var))
        self.update_labels()

    def remove_layer(self, frame):
        if len(self.layers) == 1:
            messagebox.showwarning("Warning", "At least one hidden layer is necessary!")
            return  
        for i, (f, s, v) in enumerate(self.layers):
            if f == frame:
                f.destroy()
                self.layers.pop(i)
                break
        self.update_labels()

    def update_labels(self):
        for idx, (f, s, v) in enumerate(self.layers, start=1):
            label = f.winfo_children()[0]
            label.config(text=f"Hidden {idx}:")

    def run_animation(self):
        input_size = int(self.input_scale.get())
        output_size = int(self.output_scale.get())
        hidden_structure = [v.get() for (_, _, v) in self.layers]
        structure = [input_size] + hidden_structure + [output_size]
        
        nn = NeuralNetwork(nn_structure=structure, learning_rate=0.01, epochs_num=10, dataset=self.dataset)
        animWin = AnimationWindow(self.root, nn)
        self.root.wait_window(animWin)
