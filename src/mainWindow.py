import tkinter as tk


class MainWindow:
    def __init__(self, root):
        self.root = root
        
       
        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # left row
        self.left_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        tk.Label(self.left_frame, text="Neural Network structure", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.nn_frame = tk.Frame(self.left_frame)
        self.nn_frame.pack(pady=5)
        
        self.add_layer_button = tk.Button(self.left_frame, text="Add Layer", command=self.add_layer)
        self.add_layer_button.pack(pady=5)
        
        # right row
        self.right_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
    
        tk.Label(self.right_frame, text="Controls", font=("Arial", 12, "bold")).pack(pady=5)

        self.run_button = tk.Button(self.right_frame, text="Run", command=self.run_animation)
        self.run_button.pack(pady=10)

        # Flex
        main_frame.grid_columnconfigure(0, weight=2)  # left column bigger
        main_frame.grid_columnconfigure(1, weight=1)
        
        self.layers = []
        
        
    def add_layer(self):
        layer_index = len(self.layers) + 1
        frame = tk.Frame(self.nn_frame)
        frame.pack(pady=3)

        label = tk.Label(frame, text=f"Layer {layer_index}:")
        label.pack(side=tk.LEFT, padx=5)

        scale = tk.Scale(frame, from_=1, to=32, orient=tk.HORIZONTAL)
        scale.set(8)
        scale.pack(side=tk.LEFT, padx=5)

        remove_button = tk.Button(frame, text="Remove", command=lambda f=frame: self.remove_layer(f))
        remove_button.pack(side=tk.LEFT, padx=5)

        self.layers.append((frame, scale))
        
    def remove_layer(self, frame):
        for i, (f, s) in enumerate(self.layers):
            if f == frame:
                f.destroy()
                self.layers.pop(i)
                break
        self.update_labels()
        
    def update_labels(self):
        for idx, (f, s) in enumerate(self.layers, start=1):
            label = f.winfo_children()[0]
            label.config(text=f"Layer {idx}:")
            
    def run_animation(self):
        structure = [s.get() for (f, s) in self.layers]
        #TODO


    


    