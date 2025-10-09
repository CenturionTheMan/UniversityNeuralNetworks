import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from neural_network import NeuralNetwork


class AnimationWindow(tk.Toplevel):
    def __init__(self, root, nn: NeuralNetwork):

        self.neuron_positions = None

        super().__init__(root)
        self.title("Animation")
        self.attributes("-fullscreen", True)
        self.transient(root)
        self.grab_set()
        self.nn = nn
        
        self.bind("<Escape>", lambda e: self.destroy())
        
        self.create_layout()
        self.after(100, self.calculate_cords)
        
    def create_layout(self):
        #! left
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", self.zoom)
        self.canvas.bind("<Button-5>", self.zoom)
        
        #! right
        ctrl_frame = tk.Frame(self, width=300, bg="#f0f0f0")
        ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=(20, 20))
        
        for text, cmd in [("Step →", self.next_step),
                          ("Auto Run", self.auto_run),
                          ("Reset", self.reset)]:
            tk.Button(ctrl_frame, text=text, command=cmd).pack(pady=10, fill=tk.X)

        exit_btn = tk.Button(ctrl_frame, text="Exit", command=self.exit_window, bg="red")
        exit_btn.pack(side=tk.BOTTOM, pady=5, fill=tk.X)
    
    def zoom(self, event):
        if event.num == 5 or event.delta == -120:
            scale = 0.9  # zoom out
        elif event.num == 4 or event.delta == 120:
            scale = 1.1  # zoom in
        else:
            return

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        self.canvas.scale("all", x, y, scale, scale)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    
    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def do_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
    
    def calculate_cords(self):
        structure = self.nn.get_structure()
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width < 100 or height < 100:
            width, height = 800, 600
            
        x_padding = 200
        y_padding = 200
        
        inner_width = width - 2 * x_padding
        inner_height = height - 2 * y_padding
            
        layer_spacing = inner_width / (len(structure) + 1)
        vertical_spacing = 50
        
        self.neuron_positions = []
        
        for i, num_neurons in enumerate(structure):
            layer_x = (i+1) * layer_spacing
            total_height = (num_neurons - 1) * vertical_spacing
            top_y = (inner_height - total_height) / 2
            
            layer_positions = []
            for j in range(num_neurons):
                y = top_y + j * vertical_spacing
                layer_positions.append((layer_x, y))
            self.neuron_positions.append(layer_positions)
        
        self.draw_network()
            
   
        
    def draw_network(self):
        self.canvas.delete("all")
        for i in range(len(self.neuron_positions) - 1):
            for src_x, src_y in self.neuron_positions[i]:
                for dest_x, dest_y in self.neuron_positions[i + 1]:
                    self.canvas.create_line(src_x, src_y, dest_x, dest_y, fill="gray", width=1)
        
        neuron_radius = 20
        
        layers_values = self.nn.get_layers_with_values()
        for i, layer in enumerate(self.neuron_positions):
            for j, (layer_x, y) in enumerate(layer):
                nn_graph = self.canvas.create_oval(
                    layer_x - neuron_radius, y - neuron_radius,
                    layer_x + neuron_radius, y + neuron_radius,
                    fill="lightblue", outline="black", width=1.5)
                
                
                text = f"{layers_values[i][j][0]:.2f}" if len(layers_values[i]) > 0 else "??"
                nn_text = self.canvas.create_text(
                    layer_x, y,
                    text=text,
                    font=("Arial", 10, "bold"),
                    fill="black"
                )
                
                
                
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        
        
    def next_step(self):
        self.nn.train_step()
        self.draw_network()
        print("Next step")
     
    def auto_run(self):
        pass
    
    def reset(self):
        pass
    
    def exit_window(self):
        self.destroy()