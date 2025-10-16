import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import numpy as np
from neural_network import NeuralNetwork
from style import *
from PIL import Image, ImageTk

class AnimationWindow(tk.Toplevel):
    """
    A visualization and control window for a Neural Network (NN) model.
    It allows step-by-step observation of NN training, prediction, and
    weight updates, as well as direct user interaction with the network’s structure.
    """

    def __init__(self, root, nn: NeuralNetwork):
        """
        Initialize the animation window.

        Parameters:
            root (tk.Tk): The root application window.
            nn (NeuralNetwork): Instance of the neural network to visualize.
        """
        super().__init__(root)
        self.title("Neural Network Animation")
        self.attributes("-fullscreen", True)
        self.transient(root)
        self.grab_set()
        self.nn = nn

        # Internal visualization and interaction state
        self.neuron_positions = None        # List of neuron coordinates per layer
        self.neurons_dic = {}               # Dictionary mapping canvas items to neuron info
        self.hovered_neuron = None          # Currently hovered neuron ID
        self.clicked_neuron = None          # Currently clicked neuron ID
        self.active_connections = []        # List of visible connections between neurons
        self.clicked_connections = []       # Connections highlighted upon click

        # Keyboard bindings
        self.bind("<KeyPress-Escape>", lambda e: self.destroy())  # Exit fullscreen
        self.bind("<Right>", lambda e: self.__on_right_arrow())   # Step forward (training/prediction)

        # Window background color
        self.configure(bg=COL_CONNECTIONS)

        # Build interface layout
        self.create_layout()

        # Compute neuron coordinates after layout is drawn
        self.after(100, self.calculate_cords)
        self.focus_set()
        
        
    # ============================================================================================
    #                                      LAYOUT SETUP
    # ============================================================================================

    def create_layout(self):
        """Constructs the main layout: left visualization canvas and right control panel."""
        
        # Left canvas — used for drawing neurons, connections, and labels
        self.canvas = tk.Canvas(self, bg=COL_BACKGROUND, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mouse bindings for interaction
        self.canvas.bind("<ButtonPress-1>", self.__on_button_press)
        self.canvas.bind("<B1-Motion>", self.__on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.__on_button_release)
        self.canvas.bind("<Motion>", self.__on_motion)

        # Right-side control panel for NN control buttons and status
        ctrl_frame = ttk.Frame(self, style="ControlPanel.TFrame", width=250)
        ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=15, pady=20)
        ctrl_frame.pack_propagate(False)
        
        # --- Current Sample Display ---
        ttk.Label(ctrl_frame, text="Current sample", style="H1.TLabel").pack(pady=(10, 0))
        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill=tk.X, pady=(5,5))
        self.sample_label = ttk.Label(ctrl_frame)
        self.sample_label.pack(pady=(0,0))

        # --- Training Section ---
        ttk.Label(ctrl_frame, text="Training", style="H1.TLabel").pack(pady=(40, 0))
        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill=tk.X, pady=(5,5))
        self.create_right_train_section(ctrl_frame)

        # --- Testing Section ---
        ttk.Label(ctrl_frame, text="Testing", style="H1.TLabel").pack(pady=(40, 0))
        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill=tk.X, pady=(5,5))
        self.create_right_test_section(ctrl_frame)

        # Exit button
        ttk.Button(ctrl_frame, text="Exit", command=self.__exit_window).pack(side=tk.BOTTOM, pady=10, fill=tk.X)

    def create_right_test_section(self, ctrl_frame):
        """Create the right-side controls related to NN testing/prediction."""
        test_frame = ttk.Frame(ctrl_frame, style="ControlPanel.TFrame")
        test_frame.pack(pady=0, fill=tk.X)
        
        # Display current testing sample index
        self.nn_samples_testing_var = tk.StringVar(
            value=f"Sample (testing): {self.nn.get_current_test_sample_index() + 1} / {len(self.nn.get_test_set())}"
        )
        samples_label = ttk.Label(test_frame, textvariable=self.nn_samples_testing_var, style="H3.TLabel")
        samples_label.pack(pady=(0, 5))
        
        # Prediction step button
        ttk.Button(test_frame, text="▶ Predict Step", command=self.__predict_step).pack(pady=5, fill=tk.X)

    def create_right_train_section(self, ctrl_frame):
        """Create the right-side controls for NN training operations."""
        train_frame = ttk.Frame(ctrl_frame, style="ControlPanel.TFrame")
        train_frame.pack(pady=0, fill=tk.X)
        
        # Display NN current state and progress
        state, layer_index = self.nn.get_state()
        self.nn_state_var = tk.StringVar(value=f"NN State: {state}")
        ttk.Label(train_frame, textvariable=self.nn_state_var, style="H2.TLabel").pack(pady=(0, 5))
        
        self.nn_epoch_var = tk.StringVar(value=f"Epoch (training): 0 / {self.nn.get_epoch_number()}")
        ttk.Label(train_frame, textvariable=self.nn_epoch_var, style="H3.TLabel").pack(pady=(0, 5))
        
        self.nn_samples_var = tk.StringVar(
            value=f"Sample (training): {self.nn.get_current_train_sample_index() + 1} / {len(self.nn.get_train_set())}"
        )
        ttk.Label(train_frame, textvariable=self.nn_samples_var, style="H3.TLabel").pack(pady=(0, 5))
        
        # Control buttons
        ttk.Button(train_frame, text="▶ Training Step", command=self.__training_step).pack(pady=5, fill=tk.X)
        ttk.Button(train_frame, text="▶ Train epoch", command=self.__train_epoch).pack(pady=5, fill=tk.X)
        ttk.Button(train_frame, text="⚙ Train full", command=self.__train_full).pack(pady=5, fill=tk.X)
        ttk.Button(train_frame, text="⟳ Reset Network", command=self.__reset).pack(pady=5, fill=tk.X)
    
    
    # TODO FROM HERE
    def update_sample_photo(self, sample):
        if sample is None:
            sample = np.zeros((64,))
        sample = 1 - sample
        new_img = (sample.reshape(8, 8) * 255).astype("uint8")
        new_img = Image.fromarray(new_img, mode='L').resize((128, 128))
        sample_photo = ImageTk.PhotoImage(new_img)
        self.sample_label.configure(image=sample_photo)
        self.sample_label.image = sample_photo
    

    # -------------------------------------- BUTTONS EVENTS -------------------------------------- #
    def __on_right_arrow(self):
        state, layer_index = self.nn.get_state()
        if state =="TRAINED" or state=="PREDICT_FORWARD":
            self.__predict_step()
        else:
            self.__training_step()


    def __predict_step(self):
        index = self.nn.get_current_test_sample_index()
        sample, target = self.nn.get_test_set()[index - 1]
        
        try:
            self.nn.predict_step(sample)
        except Exception as e:
            messagebox.showinfo("Error", str(e))
            return
        
        self.draw_network_text()
        self.draw_active_layer_mark()
        state, layer_index = self.nn.get_state()
        
        self.nn_samples_testing_var.set(f"Sample (testing): {self.nn.get_current_test_sample_index() + 1} / {len(self.nn.get_test_set())}")
        
        if state == "PREDICT_FORWARD" and layer_index == 0:
            print(f"Predicting sample index: {index}")
            self.update_sample_photo(sample)
        elif state == "TRAINED":
            self.update_sample_photo(None)

    def __training_step(self):
        try:
            self.nn.train_step()
        except Exception as e:
            messagebox.showinfo("Error", str(e))
            return
        self.draw_network_text()
        self.draw_active_layer_mark()
        state, layer_index = self.nn.get_state()
        self.nn_state_var.set(f"NN State: {state}")
        self.nn_samples_var.set(f"Sample (training): {self.nn.get_current_train_sample_index() + 1} / {len(self.nn.get_train_set())}")
        self.nn_epoch_var.set(f"Epoch (training): {self.nn.get_current_epoch()} / {self.nn.get_epoch_number()}")
        
        if state == "FORWARD" and layer_index == 0:
            self.update_sample_photo(self.nn.get_train_set()[self.nn.get_current_train_sample_index()][0])
        elif state == "NEW_SAMPLE":
            self.update_sample_photo(None)
     
    def __train_epoch(self):
        samples_amt = len(self.nn.get_train_set())
        try: 
            con_training = True
            while con_training and self.nn.get_current_train_sample_index() < samples_amt-1:
                con_training = self.nn.train_step()
        except Exception as e:
            messagebox.showinfo("Error", str(e))
            return
        print("Epoch complete")
        state, layer_index = self.nn.get_state()
        self.nn_state_var.set(f"NN State: {state}")
        self.nn_samples_var.set(f"Sample (training): {self.nn.get_current_train_sample_index() + 1} / {len(self.nn.get_train_set())}")
        self.nn_epoch_var.set(f"Epoch (training): {self.nn.get_current_epoch()} / {self.nn.get_epoch_number()}")
        self.draw_network_text()
        self.draw_active_layer_mark()
        self.update_sample_photo(None)
     
    def __train_full(self):
        try: 
            con_training = True
            while con_training:
                con_training = self.nn.train_step()
        except Exception as e:
            messagebox.showinfo("Error", str(e))
            return
        print("Training complete")
        state, layer_index = self.nn.get_state()
        self.nn_state_var.set(f"NN State: {state}")
        self.nn_samples_var.set(f"Sample (training): {self.nn.get_current_train_sample_index() + 1} / {len(self.nn.get_train_set())}")
        self.nn_epoch_var.set(f"Epoch (training): {self.nn.get_current_epoch()} / {self.nn.get_epoch_number()}")
        self.draw_network_text()
        self.draw_active_layer_mark()
        self.update_sample_photo(None)
    
    def __reset(self):
        self.neuron_positions = None
        self.neurons_dic = {}
        self.hovered_neuron = None
        self.clicked_neuron = None
        self.active_connections = []
        self.clicked_connections = []
        self.nn = NeuralNetwork(nn_structure=self.nn.get_structure(), 
                                learning_rate=self.nn.get_learning_rate(), 
                                epochs_num=self.nn.get_epoch_number(), 
                                dataset=self.nn.get_train_set())
        self.after(100, self.calculate_cords)
        self.nn_state_var.set(f"NN State: UNTRAINED")
        self.nn_samples_var.set(f"Sample (training): 1 / {len(self.nn.get_train_set())}")
        self.nn_epoch_var.set(f"Epoch (training): 0 / {self.nn.get_epoch_number()}")
        self.update_sample_photo(None)
    
    def __exit_window(self):
        self.destroy()
        
    # -------------------------------------- INPUT EVENTS -------------------------------------- #

    def __on_click(self, event):
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        items = self.canvas.find_overlapping(cx, cy, cx, cy)
        clicked = [i for i in items if "neurons" in self.canvas.gettags(i)]
        
        if not clicked or clicked[0] not in self.neurons_dic:
            return
        
        if self.clicked_neuron == clicked[0]:
            self.canvas.itemconfig(clicked[0], fill=COL_NEURONS)
            self.clicked_neuron = None
        else:        
            if self.clicked_neuron is not None:
                self.canvas.itemconfig(self.clicked_neuron, fill=COL_NEURONS)
            self.canvas.itemconfig(clicked[0], fill=COL_CONNECTIONS)
            self.clicked_neuron = clicked[0]
        self.draw_network_connections()
    
    def __on_motion(self, event):
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

    def __on_button_press(self, event):
        self._press_x, self._press_y = event.x, event.y
        self._did_pan = False
        self.canvas.scan_mark(event.x, event.y)

    def __on_mouse_drag(self, event):
        dx = event.x - getattr(self, "_press_x", event.x)
        dy = event.y - getattr(self, "_press_y", event.y)
        if abs(dx) > 4 or abs(dy) > 4:
            self._did_pan = True
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def __on_button_release(self, event):
        if not getattr(self, "_did_pan", False):
            self.__on_click(event)
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=bbox)
    
    # -------------------------------------- CALCULATE CORDS -------------------------------------- #
    
    def calculate_cords(self):
        structure = self.nn.get_structure()
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width < 100 or height < 100:
            width, height = 800, 600

        layer_spacing = width / (len(structure) + 1)
        vertical_spacing = 80

        self.neuron_positions = []
        self.error_positions = []

        for i, num_neurons in enumerate(structure):
            layer_x = (i + 1) * layer_spacing - 0.25*layer_spacing
            total_height = (num_neurons - 1) * vertical_spacing
            top_y = (height - total_height) / 2

            layer_positions = []
            for j in range(num_neurons):
                y = top_y + j * vertical_spacing
                layer_positions.append((layer_x, y))
            self.neuron_positions.append(layer_positions)

        error_x = layer_spacing * (len(structure) + 1) - 0.75*layer_spacing
        for j in range(structure[-1]):
            y = (height - (structure[-1] - 1) * vertical_spacing) / 2 + j * vertical_spacing
            self.error_positions.append((error_x, y))

        self.draw_network_all()

    # -------------------------------------- DRAW NETWORK -------------------------------------- #
        
    def draw_network_all(self):
        self.canvas.delete("all")

        self.neuron_radius = 20
        self.bias_size = 30

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
                self.neurons_dic[(i,j)] = nn_neuron

                if i > 0:
                    nn_bias = self.canvas.create_rectangle(layer_x + self.neuron_radius, y - self.neuron_radius, 
                                                           layer_x + self.neuron_radius + self.bias_size, 
                                                           y - self.neuron_radius - self.bias_size, 
                                                           fill=COL_BACKGROUND, outline=COL_TEXT, width=1, tags="biases")
                    
                    
        for j, (x, y) in enumerate(self.error_positions):
            nn_error = self.canvas.create_rectangle(
                x, y - self.neuron_radius,
                x + self.bias_size, y + self.neuron_radius,
                fill=COL_BACKGROUND, outline=COL_NEURONS, width=1.5, tags="errors"
            )
                    
        self.draw_network_text()
        self.update_sample_photo(None)
    
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
                    nn_line = self.canvas.create_line(src_x, src_y, dest_x, dest_y, 
                                                      fill=COL_CONNECTIONS, width=1, tags="connections")
                    self.canvas.tag_lower(nn_line) 
                    self.active_connections.append([i,j, i+1, o, 
                                                    (src_x + dest_x)/2,
                                                    (src_y + dest_y)/2
                                                    ])
                    
            if i - 1 >= 0:
                for o, (dest_x, dest_y) in enumerate(self.neuron_positions[i - 1]):
                    nn_line = self.canvas.create_line(src_x, src_y, dest_x, dest_y, 
                                                      fill=COL_CONNECTIONS, width=1, tags="connections")
                    self.canvas.tag_lower(nn_line) 
                    self.active_connections.append([i-1, o, i, j, 
                                                    (src_x + dest_x)/2,
                                                    (src_y + dest_y)/2
                                                    ])
                    
        if draw_all:
            for i in range(len(self.neuron_positions) - 1):
                for src_x, src_y in self.neuron_positions[i]:
                    for o, (dest_x, dest_y) in enumerate(self.neuron_positions[i + 1]):
                        nn_line = self.canvas.create_line(src_x, src_y, dest_x, dest_y, 
                                                          fill=COL_CONNECTIONS, width=1, tags="connections")
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
                    fill="white" if len(layers_values[i]) > 0 and layers_values[i][j][0] < 0.5 else COL_TEXT,
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
        
        state, _ = self.nn.get_state()
        error_values = self.nn.get_errors_matrix()
        for j, (x, y) in enumerate(self.error_positions):
            nn_text_error = self.canvas.create_text(
                x + self.bias_size/2, y,
                text=f"{error_values[j][0]:.2f}" if error_values is not None and state == "ERROR" else "??",
                font=("Arial", 10, "bold"),
                fill=COL_RED,
                tags="text"
            )
        
        self.draw_network_text_connections()
        self.draw_activated_neurons()
        
    
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
            
    def draw_active_layer_mark(self):
        self.canvas.delete("active_layer")
        
        state, layer_index = self.nn.get_state()
        if layer_index is None:
            return
        offset = 15
        
        layer = self.neuron_positions[layer_index]
        if state == "BACKWARD":
            layer_next = self.neuron_positions[layer_index + 1]
            min_x = min(pos[0] for pos in layer)  +  self.neuron_radius + offset
            max_x = max(pos[0] for pos in layer_next) + self.neuron_radius + offset
            min_y = min(pos[1] for pos in layer_next) - self.neuron_radius - offset
            max_y = max(pos[1] for pos in layer_next) + self.neuron_radius + offset        
        else:
            min_x = min(pos[0] for pos in layer) - self.neuron_radius - offset
            max_x = max(pos[0] for pos in layer) + self.neuron_radius + offset
            min_y = min(pos[1] for pos in layer) - self.neuron_radius - offset
            max_y = max(pos[1] for pos in layer) + self.neuron_radius + offset
        
        self.canvas.create_rectangle(
            min_x, min_y, max_x, max_y,
            outline=COL_FOCUS,
            width=3,
            dash=(5, 3),
            tags="active_layer"
        )
        self.canvas.tag_lower("active_layer")
        
        
    def draw_activated_neurons(self):
        layers_values = self.nn.get_layers_with_values()
        for i, layer in enumerate(self.neuron_positions):
            for j, (layer_x, y) in enumerate(layer):
                nn_neuron = self.neurons_dic[(i,j)]
                value = layers_values[i][j][0] if len(layers_values[i]) > 0 else 0
                intensity = int(min(max(value * 255, 0), 255))
                color = f'#{int((int(COL_NEURONS[1:3],16) * (255 - intensity) + int(COL_FOCUS[1:3],16) * intensity)/255):02x}' + \
                        f'{int((int(COL_NEURONS[3:5],16) * (255 - intensity) + int(COL_FOCUS[3:5],16) * intensity)/255):02x}' + \
                        f'{int((int(COL_NEURONS[5:7],16) * (255 - intensity) + int(COL_FOCUS[5:7],16) * intensity)/255):02x}'             
                
                self.canvas.itemconfig(nn_neuron, fill=color)
                
                     
        
        
    