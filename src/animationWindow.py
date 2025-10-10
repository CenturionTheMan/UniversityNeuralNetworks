import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from neural_network import NeuralNetwork
from colors import *


class AnimationWindow(tk.Toplevel):
    def __init__(self, root, nn: NeuralNetwork):
        super().__init__(root)
        self.title("Neural Network Animation")
        self.attributes("-fullscreen", True)
        self.transient(root)
        self.grab_set()
        self.nn = nn

        # internal state
        self.neuron_positions = None
        self.neurons_dic = {}
        self.hovered_neuron = None
        self.clicked_neuron = None
        self.active_connections = []
        self.clicked_connections = []

        # Bind keys
        self.bind("<KeyPress-Escape>", lambda e: self.destroy())
        self.bind("<Right>", lambda e: self.next_step())

        # Colors
        self.configure(bg=COL_CONNECTIONS)

        
        self.create_layout()
        self.after(100, self.calculate_cords)
        self.focus_set()
        
    
        
    def create_layout(self):
        #! Left canvas area
        self.canvas = tk.Canvas(self, bg=COL_BACKGROUND, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Button-1>", self.on_click)

        #! Right control panel
        ctrl_frame = ttk.Frame(self, width=300)
        ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=15, pady=20)

        ttk.Label(ctrl_frame, text="Training Controls", style="Title.TLabel").pack(pady=(0, 10))

        self.create_right_train_section(ctrl_frame)

        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill=tk.X, pady=15)

        exit_btn = ttk.Button(ctrl_frame, text="Exit", command=self.exit_window)
        exit_btn.pack(side=tk.BOTTOM, pady=10, fill=tk.X)

    def create_right_train_section(self, ctrl_frame):
        train_frame = ttk.Frame(ctrl_frame)
        train_frame.pack(pady=10, fill=tk.X)

        ttk.Button(train_frame, text="▶ Training Step", command=self.next_step).pack(pady=10, fill=tk.X)
        ttk.Button(train_frame, text="⟳ Reset Network", command=self.reset).pack(pady=5, fill=tk.X)
        ttk.Button(train_frame, text="⚙ Auto Run", command=self.auto_run).pack(pady=5, fill=tk.X)

    def on_click(self, event):
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        items = self.canvas.find_overlapping(cx, cy, cx, cy)
        clicked = [i for i in items if "neurons" in self.canvas.gettags(i)]
        
        if not clicked or clicked[0] not in self.neurons_dic:
            return
        
        if self.clicked_neuron == clicked[0]:
            self.clicked_neuron = None
        else:        
            self.clicked_neuron = clicked[0]
        self.draw_network_connections()
        
    
    def on_motion(self, event):
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        items = self.canvas.find_overlapping(cx, cy, cx, cy)
        hovered = [i for i in items if "neurons" in self.canvas.gettags(i)]

        if not hovered or hovered[0] not in self.neurons_dic:
            if self.hovered_neuron is not None:
                self.hovered_neuron = None
                self.draw_network_connections()
            return

        if self.hovered_neuron == hovered[0]:
            return

        self.hovered_neuron = hovered[0]
        self.draw_network_connections()

    
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
            
        x_padding, y_padding = 200, 200
        inner_width, inner_height = width - 2 * x_padding, height - 2 * y_padding
            
        layer_spacing = inner_width / (len(structure) + 1)
        vertical_spacing = 80
        
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
        
        self.draw_network_all()
   
        
    def draw_network_all(self):
        self.canvas.delete("all")

        self.neuron_radius = 20
        self.bias_size = 30

        #self._draw_background_grid(spacing=60, color="#333333")

        self.neurons_dic = {}
        for i, layer in enumerate(self.neuron_positions):
            for j, (layer_x, y) in enumerate(layer):
                nn_neuron = self.canvas.create_oval(
                    layer_x - self.neuron_radius,
                    y - self.neuron_radius,
                    layer_x + self.neuron_radius,
                    y + self.neuron_radius,
                    fill=COL_NEURONS, outline=COL_TEXT, width=1.5, tags="neurons"
                )

                self.neurons_dic[nn_neuron] = {"index": (i, j), "cords": (layer_x, y)}

                if i > 0:
                    nn_bias = self.canvas.create_rectangle(layer_x + self.neuron_radius, y - self.neuron_radius, 
                                                           layer_x + self.neuron_radius + self.bias_size, 
                                                           y - self.neuron_radius - self.bias_size, 
                                                           fill=COL_BACKGROUND, outline=COL_TEXT, width=1, tags="biases")

        self.draw_network_text()
    
    def draw_network_connections(self, draw_all=False):
        self.canvas.delete("connections")
        self.active_connections = []
        
        neurons = []
        
        if self.clicked_neuron != None:
            neuron = self.neurons_dic[self.clicked_neuron]
            neurons.append(neuron)
        if self.hovered_neuron != None and self.hovered_neuron != self.clicked_neuron:
            neuron = self.neurons_dic[self.hovered_neuron]
            neurons.append(neuron)
                
        
        for neuron in neurons:
            i,j = neuron["index"][0],neuron["index"][1]
            src_x, src_y = neuron["cords"]
            
            if i + 1 < len(self.neuron_positions):
                for o, (dest_x, dest_y) in enumerate(self.neuron_positions[i + 1]):
                    nn_line = self.canvas.create_line(src_x, src_y, dest_x, dest_y, fill=COL_CONNECTIONS, width=1, tags="connections")
                    self.canvas.tag_lower(nn_line) 
                    self.active_connections.append([i,j, i+1, o, 
                                                    (src_x + dest_x)/2,
                                                    (src_y + dest_y)/2
                                                    ])
                    
            if i - 1 >= 0:
                for o, (dest_x, dest_y) in enumerate(self.neuron_positions[i - 1]):
                    nn_line = self.canvas.create_line(src_x, src_y, dest_x, dest_y, fill=COL_CONNECTIONS, width=1, tags="connections")
                    self.canvas.tag_lower(nn_line) 
                    self.active_connections.append([i-1, o, i, j, 
                                                    (src_x + dest_x)/2,
                                                    (src_y + dest_y)/2
                                                    ])
                    
        if draw_all:
            for i in range(len(self.neuron_positions) - 1):
                for src_x, src_y in self.neuron_positions[i]:
                    for o, (dest_x, dest_y) in enumerate(self.neuron_positions[i + 1]):
                        nn_line = self.canvas.create_line(src_x, src_y, dest_x, dest_y, fill=COL_CONNECTIONS, width=1, tags="connections")
                        self.canvas.tag_lower(nn_line) 
                        self.active_connections.append([i,j, i+1, o, 
                                                        (src_x + dest_x)/2,
                                                        (src_y + dest_y)/2
                                                        ])
        self.draw_network_text_connections()
        

    def draw_network_text(self):
        self.canvas.delete("text")

        layers_values = self.nn.get_layers_with_values()
        biases_values = self.nn.get_biases()
        for i, layer in enumerate(self.neuron_positions):
            for j, (layer_x, y) in enumerate(layer):
                nn_text_neuron = self.canvas.create_text(
                    layer_x, y,
                    text=f"{layers_values[i][j][0]:.2f}" if len(layers_values[i]) > 0 else "??",
                    font=("Arial", 10, "bold"),
                    fill=COL_TEXT,
                    tags="text"
                )
                
                if i > 0:
                    nn_text_bias = self.canvas.create_text(
                        layer_x + self.neuron_radius + self.bias_size/2, y - self.neuron_radius - self.bias_size/2,
                        text=f"{biases_values[i-1][j][0]:.2f}",
                        font=("Arial", 10, "bold"),
                        fill=COL_TEXT,
                        tags="text"
                    ) 
        self.draw_network_text_connections()
    
    def draw_network_text_connections(self):
        self.canvas.delete("text_connections")
        weights = self.nn.get_weights()
        for active_connection in self.active_connections + self.clicked_connections:
            offset = 15
            i,j,o,p, mid_x, mid_y = active_connection
            
            nn_text_weight_bg = self.canvas.create_rectangle(
                mid_x - offset, mid_y + offset,
                mid_x + offset, mid_y - offset,
                fill=COL_BACKGROUND,
                outline=COL_TEXT,
                width=1, tags="text_connections"
            ) 
            
            weight_value = weights[i][p][j]
            nn_text_weight = self.canvas.create_text(
                mid_x, mid_y,
                text=f"{weight_value:.2f}",
                font=("Arial", 8),
                fill=COL_TEXT,
                tags="text_connections"
            )    
                     
        
        
    def next_step(self):
        self.nn.train_step()
        self.draw_network_text()
        print("Next step")
     
    def auto_run(self):
        pass
    
    def reset(self):
        self.neuron_positions = None
        self.neurons_dic = {}
        self.hovered_neuron = None
        self.active_connections = []
        self.nn = NeuralNetwork(nn_structure=self.nn.get_structure(), 
                                learning_rate=self.nn.__learning_rate, 
                                epochs_num=self.nn.epochs_num, 
                                dataset=self.nn.dataset)
        self.draw_network_all()
    
    def exit_window(self):
        self.destroy()