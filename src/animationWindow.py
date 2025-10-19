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
    weight updates, as well as direct user interaction with the network's structure.
    """

    def __init__(self, root, nn: NeuralNetwork):
        """
        Initialize the animation window for neural network visualization.

        Parameters
        ----------
        root : tk.Tk
            The root application window.
        nn : NeuralNetwork
            Instance of the neural network to visualize.
        """
        super().__init__(root)
        
        # ----- Window setup -----
        self.title("Neural Network Animation")
        self.attributes("-fullscreen", True)   # Open in fullscreen mode
        self.transient(root)                   # Keep window on top of root
        self.grab_set()                        # Prevent interaction with root window
        self.nn = nn                           # Reference to the neural network instance

        # ----- Internal visualization state -----
        self.neuron_positions = None           # List of neuron coordinates per layer
        self.neurons_dic = {}                  # Maps canvas items to neuron info
        self.hovered_neuron = None             # Currently hovered neuron ID
        self.clicked_neuron = None             # Currently selected neuron ID
        self.active_connections = []           # Lines representing visible connections
        self.clicked_connections = []          # Highlighted or selected connections

        # ----- Keyboard bindings -----
        self.bind("<KeyPress-Escape>", lambda e: self.destroy())   # Close window on Esc
        self.bind("<Right>", lambda e: self.__on_right_arrow())    # Step through training/prediction

        # ----- Visual configuration -----
        self.configure(bg=COL_CONNECTIONS)     # Set background color

        # ----- Build UI layout -----
        self.create_layout()                   # Create canvas and control panels

        # Schedule coordinate computation once layout dimensions are available
        self.after(100, self.calculate_cords)
        self.focus_set()                       # Ensure window captures keyboard input


        
    # -------------------------------------- LAYOUT SETUP -------------------------------------- #

    def create_layout(self):
        """Constructs the main layout: left visualization canvas and right control panel."""

        # --- Left visualization canvas ---
        # Used for drawing neurons, connections, and labels
        self.canvas = tk.Canvas(self, bg=COL_BACKGROUND, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind mouse events for user interaction (click, drag, hover)
        self.canvas.bind("<ButtonPress-1>", self.__on_button_press)
        self.canvas.bind("<B1-Motion>", self.__on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.__on_button_release)
        self.canvas.bind("<Motion>", self.__on_motion)

        # --- Right-side control panel ---
        # Contains controls for training/testing and sample display
        ctrl_frame = ttk.Frame(self, style="ControlPanel.TFrame", width=250)
        ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=15, pady=20)
        ctrl_frame.pack_propagate(False)  # Prevent shrinking to contents

        # --- Current Sample Display ---
        ttk.Label(ctrl_frame, text="Current sample", style="H1.TLabel").pack(pady=(10, 0))
        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill=tk.X, pady=(5, 5))
        self.sample_label = ttk.Label(ctrl_frame)
        self.sample_label.pack()

        # --- Training Section ---
        ttk.Label(ctrl_frame, text="Training", style="H1.TLabel").pack(pady=(40, 0))
        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill=tk.X, pady=(5, 5))
        self.create_right_train_section(ctrl_frame)

        # --- Testing Section ---
        ttk.Label(ctrl_frame, text="Testing", style="H1.TLabel").pack(pady=(40, 0))
        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill=tk.X, pady=(5, 5))
        self.create_right_test_section(ctrl_frame)

        # --- Exit Button ---
        ttk.Button(ctrl_frame, text="Exit", command=self.__exit_window).pack(
            side=tk.BOTTOM, pady=10, fill=tk.X
    )

    def create_right_test_section(self, ctrl_frame):
        """Create the right-side controls related to neural network testing/prediction."""
        # --- Container for testing controls ---
        test_frame = ttk.Frame(ctrl_frame, style="ControlPanel.TFrame")
        test_frame.pack(pady=0, fill=tk.X)

        # --- Display current testing sample index ---
        # Shows which sample from the test set is currently being visualized/predicted
        self.nn_samples_testing_var = tk.StringVar(
            value=f"Sample (testing): {self.nn.get_current_test_sample_index() + 1} / {len(self.nn.get_test_set())}"
        )

        # Label bound to the dynamic sample index variable
        samples_label = ttk.Label(
            test_frame,
            textvariable=self.nn_samples_testing_var,
            style="H3.TLabel"
        )
        samples_label.pack(pady=(0, 5))

        # --- Prediction Step Button ---
        # Advances the visualization by one prediction step
        ttk.Button(
            test_frame,
            text="▶ Predict Step",
            command=self.__predict_step
        ).pack(pady=5, fill=tk.X)


    def create_right_train_section(self, ctrl_frame):
        """Create the right-side controls for neural network training operations."""

        # --- Container for training controls ---
        train_frame = ttk.Frame(ctrl_frame, style="ControlPanel.TFrame")
        train_frame.pack(pady=0, fill=tk.X)

        # --- Display current neural network state ---
        # Shows whether the network is idle, training, or predicting
        state, layer_index = self.nn.get_state()
        self.nn_state_var = tk.StringVar(value=f"NN State: {state}")
        ttk.Label(train_frame, textvariable=self.nn_state_var, style="H2.TLabel").pack(pady=(0, 5))

        # --- Display current training epoch progress ---
        self.nn_epoch_var = tk.StringVar(
            value=f"Epoch (training): 0 / {self.nn.get_epoch_number()}"
        )
        ttk.Label(train_frame, textvariable=self.nn_epoch_var, style="H3.TLabel").pack(pady=(0, 5))

        # --- Display current training sample index ---
        self.nn_samples_var = tk.StringVar(
            value=f"Sample (training): {self.nn.get_current_train_sample_index() + 1} / {len(self.nn.get_train_set())}"
        )
        ttk.Label(train_frame, textvariable=self.nn_samples_var, style="H3.TLabel").pack(pady=(0, 5))

        # --- Control buttons for training operations ---
        # Step-by-step, per-epoch, full training, and reset actions
        ttk.Button(train_frame, text="▶ Training Step", command=self.__training_step).pack(pady=5, fill=tk.X)
        ttk.Button(train_frame, text="▶ Train Epoch", command=self.__train_epoch).pack(pady=5, fill=tk.X)
        ttk.Button(train_frame, text="⚙ Train Full", command=self.__train_full).pack(pady=5, fill=tk.X)
        ttk.Button(train_frame, text="⟳ Reset Network", command=self.__reset).pack(pady=5, fill=tk.X)

    
    
    def update_sample_photo(self, sample):
        """
        Update the displayed sample image on the control panel.

        Parameters
        ----------
        sample : np.ndarray or None
            The input sample vector (flattened 8x8 image) to display.
            If None, a blank (black) sample image is shown.
        """
        
        # Use a blank 8x8 image if no sample is provided
        if sample is None:
            sample = np.zeros((64,))
        
        # Invert pixel values (for correct display contrast)
        sample = 1 - sample

        # Reshape to 8x8 and scale pixel values to 0–255
        image_array = (sample.reshape(8, 8) * 255).astype("uint8")

        # Convert to a PIL image (grayscale) and resize for visibility
        pil_image = Image.fromarray(image_array, mode='L').resize((128, 128))

        # Convert PIL image to Tkinter-compatible format
        sample_photo = ImageTk.PhotoImage(pil_image)

        # Update image label
        self.sample_label.configure(image=sample_photo)
        self.sample_label.image = sample_photo

    

    # -------------------------------------- BUTTONS EVENTS -------------------------------------- #
    def __on_right_arrow(self):
        """Handle Right Arrow key press to step through NN visualization."""

        # Retrieve current neural network state and active layer index
        state, layer_index = self.nn.get_state()

        # If network is trained or in prediction mode → perform prediction step
        if state == "TRAINED" or state == "PREDICT_FORWARD":
            self.__predict_step()
        # Otherwise → continue training step-by-step
        else:
            self.__training_step()

    def __predict_step(self):
        """Perform a single forward prediction step in the neural network visualization."""
        
        # Get the current test sample index
        index = self.nn.get_current_test_sample_index()
        
        # Retrieve the current sample and its expected target from the test set
        # Note: index - 1 ensures correct zero-based list access
        sample, target = self.nn.get_test_set()[index - 1]
        
        # Attempt to perform one forward prediction step
        try:
            self.nn.predict_step(sample)
        except Exception as e:
            # Show an error dialog if prediction fails
            messagebox.showinfo("Error", str(e))
            return

        # Redraw the neural network layers and their text labels after prediction update
        self.draw_network_text()
        
        # Highlight the currently active layer (during forward pass)
        self.draw_active_layer_mark()

        # Get updated network state and layer index
        state, layer_index = self.nn.get_state()
        
        # Update UI label displaying test sample progress
        self.nn_samples_testing_var.set(
            f"Sample (testing): {self.nn.get_current_test_sample_index() + 1} / {len(self.nn.get_test_set())}"
        )
        
        # If the network starts a new forward prediction — display the sample image
        if state == "PREDICT_FORWARD" and layer_index == 0:
            print(f"Predicting sample index: {index}")
            self.update_sample_photo(sample)
        
        # When prediction finishes, clear the displayed sample image
        elif state == "TRAINED":
            self.update_sample_photo(None)


    def __training_step(self):
        """Execute a single training step and update the visualization accordingly."""
        
        # Attempt to perform one training step on the neural network
        try:
            self.nn.train_step()
        except Exception as e:
            # Show a popup message if any error occurs during training
            messagebox.showinfo("Error", str(e))
            return

        # Redraw the network layers, neurons, and text after the step
        self.draw_network_text()
        # Visually mark the currently active layer in training
        self.draw_active_layer_mark()

        # Retrieve updated state and active layer index
        state, layer_index = self.nn.get_state()

        # Update training information displayed in the control panel
        self.nn_state_var.set(f"NN State: {state}")
        self.nn_samples_var.set(
            f"Sample (training): {self.nn.get_current_train_sample_index() + 1} / {len(self.nn.get_train_set())}"
        )
        self.nn_epoch_var.set(
            f"Epoch (training): {self.nn.get_current_epoch()} / {self.nn.get_epoch_number()}"
        )

        # If a new forward pass starts — show the current input sample on the preview
        if state == "FORWARD" and layer_index == 0:
            sample = self.nn.get_train_set()[self.nn.get_current_train_sample_index()][0]
            self.update_sample_photo(sample)

        # If a new sample is being loaded — clear the image preview
        elif state == "NEW_SAMPLE":
            self.update_sample_photo(None)

     
    def __train_epoch(self):
        """Perform training for one full epoch (iterate over all training samples)."""
        
        # Get total number of samples in the training set
        samples_amt = len(self.nn.get_train_set())

        try:
            # Continue training until either stopped or last sample is reached
            con_training = True
            while con_training and self.nn.get_current_train_sample_index() < samples_amt - 1:
                con_training = self.nn.train_step()
        except Exception as e:
            # Show popup if any training error occurs
            messagebox.showinfo("Error", str(e))
            return

        print("Epoch complete")

        # Retrieve the current state and layer after the epoch
        state, layer_index = self.nn.get_state()

        # Update displayed training statistics (state, sample, epoch info)
        self.nn_state_var.set(f"NN State: {state}")
        self.nn_samples_var.set(
            f"Sample (training): {self.nn.get_current_train_sample_index() + 1} / {len(self.nn.get_train_set())}"
        )
        self.nn_epoch_var.set(
            f"Epoch (training): {self.nn.get_current_epoch()} / {self.nn.get_epoch_number()}"
        )

        # Redraw network visualization and active layer markers
        self.draw_network_text()
        self.draw_active_layer_mark()

        # Clear the sample preview image at the end of the epoch
        self.update_sample_photo(None)

     
    def __train_full(self):
        """Train the neural network continuously until all training steps are completed."""
        
        try:
            # Continue training until train_step() returns False (no more steps)
            con_training = True
            while con_training:
                con_training = self.nn.train_step()
        except Exception as e:
            # Show popup if any error occurs during training
            messagebox.showinfo("Error", str(e))
            return

        print("Training complete")

        # Retrieve the current state and layer after full training
        state, layer_index = self.nn.get_state()

        # Update displayed training statistics (state, sample, epoch info)
        self.nn_state_var.set(f"NN State: {state}")
        self.nn_samples_var.set(
            f"Sample (training): {self.nn.get_current_train_sample_index() + 1} / {len(self.nn.get_train_set())}"
        )
        self.nn_epoch_var.set(
            f"Epoch (training): {self.nn.get_current_epoch()} / {self.nn.get_epoch_number()}"
        )

        # Redraw network visualization and active layer markers
        self.draw_network_text()
        self.draw_active_layer_mark()

        # Clear the sample preview image at the end of training
        self.update_sample_photo(None)

    
    def __reset(self):
        """Reset the neural network and visualization to its initial untrained state."""
        
        # Clear neuron visualization and interaction states
        self.neuron_positions = None
        self.neurons_dic = {}
        self.hovered_neuron = None
        self.clicked_neuron = None
        self.active_connections = []
        self.clicked_connections = []

        # Recreate the neural network using the same structure, learning rate, epochs, and dataset
        self.nn = NeuralNetwork(
            nn_structure=self.nn.get_structure(), 
            learning_rate=self.nn.get_learning_rate(), 
            epochs_num=self.nn.get_epoch_number(), 
            dataset=self.nn.get_train_set()
        )

        # Recalculate neuron coordinates after layout redraw
        self.after(100, self.calculate_cords)

        # Reset UI labels to initial values
        self.nn_state_var.set("NN State: UNTRAINED")
        self.nn_samples_var.set(f"Sample (training): 1 / {len(self.nn.get_train_set())}")
        self.nn_epoch_var.set(f"Epoch (training): 0 / {self.nn.get_epoch_number()}")

        # Clear the sample display
        self.update_sample_photo(None)


    def __exit_window(self):
        """Close the animation window."""
        self.destroy()

        
    # -------------------------------------- INPUT EVENTS -------------------------------------- #

    def __on_click(self, event):
        """Handle mouse click on the canvas to select/deselect neurons."""
        
        # Convert event coordinates to canvas coordinates
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Find all canvas items under the click point
        items = self.canvas.find_overlapping(cx, cy, cx, cy)

        # Filter only neurons from the items
        clicked = [i for i in items if "neurons" in self.canvas.gettags(i)]

        # If no neuron was clicked or neuron is not recognized, exit
        if not clicked or clicked[0] not in self.neurons_dic:
            return

        # If clicked neuron is already selected, deselect it
        if self.clicked_neuron == clicked[0]:
            self.canvas.itemconfig(clicked[0], fill=COL_NEURONS)
            self.clicked_neuron = None
        else:
            # If another neuron was previously clicked, reset its color
            if self.clicked_neuron is not None:
                self.canvas.itemconfig(self.clicked_neuron, fill=COL_NEURONS)
            
            # Mark the newly clicked neuron
            self.canvas.itemconfig(clicked[0], fill=COL_CONNECTIONS)
            self.clicked_neuron = clicked[0]

        # Redraw connections to reflect current selection
        self.draw_network_connections()

    def __on_motion(self, event):
        """Handle mouse movement over the canvas to highlight hovered neurons."""
        
        # Convert event coordinates to canvas coordinates
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Find all canvas items under the mouse pointer
        items = self.canvas.find_overlapping(cx, cy, cx, cy)

        # Filter only neurons from the items
        hovered = [i for i in items if "neurons" in self.canvas.gettags(i)]

        # If no neuron is hovered or hovered neuron is unrecognized
        if not hovered or hovered[0] not in self.neurons_dic:
            # If there was a previously hovered neuron, clear it
            if self.hovered_neuron is not None:
                self.hovered_neuron = None
                self.draw_network_connections()  # Redraw to remove hover highlight
            return

        # If hovering over the same neuron as before, do nothing
        if self.hovered_neuron == hovered[0]:
            return        

        # Update the currently hovered neuron
        self.hovered_neuron = hovered[0]
        self.draw_network_connections()  # Redraw connections with hover highlight

    def __on_button_press(self, event):
        """Handle mouse button press for panning or clicking neurons."""
        # Store initial mouse press position
        self._press_x, self._press_y = event.x, event.y
        self._did_pan = False  # Flag to distinguish click vs drag
        self.canvas.scan_mark(event.x, event.y)  # Prepare canvas for potential drag


    def __on_mouse_drag(self, event):
        """Handle mouse drag to pan the canvas."""
        dx = event.x - getattr(self, "_press_x", event.x)
        dy = event.y - getattr(self, "_press_y", event.y)

        # Consider as drag only if movement exceeds threshold
        if abs(dx) > 4 or abs(dy) > 4:
            self._did_pan = True
            self.canvas.scan_dragto(event.x, event.y, gain=1)  # Move canvas with mouse


    def __on_button_release(self, event):
        """Handle mouse button release: click neuron if not a drag, update scroll region."""
        # If the mouse was not dragged, treat as a click
        if not getattr(self, "_did_pan", False):
            self.__on_click(event)
        
        # Update scrollable area to fit all canvas items
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=bbox)

    
    # -------------------------------------- DRAW NETWORK -------------------------------------- #
    
    def calculate_cords(self):
        """Compute the (x, y) positions for each neuron and error display positions on the canvas."""

        structure = self.nn.get_structure()  # Neural network layers (list of neuron counts per layer)
        width = self.canvas.winfo_width()    # Canvas width
        height = self.canvas.winfo_height()  # Canvas height

        # Fallback canvas size if not yet properly rendered
        if width < 100 or height < 100:
            width, height = 800, 600

        layer_spacing = width / (len(structure) + 1)  # Horizontal spacing between layers
        vertical_spacing = 80                         # Vertical spacing between neurons

        self.neuron_positions = []  # Stores coordinates of neurons per layer
        self.error_positions = []   # Stores coordinates for error visualization (output layer)

        # Compute neuron positions for each layer
        for i, num_neurons in enumerate(structure):
            layer_x = (i + 1) * layer_spacing - 0.25 * layer_spacing  # X coordinate of the layer
            total_height = (num_neurons - 1) * vertical_spacing       # Total vertical span of neurons
            top_y = (height - total_height) / 2                       # Y coordinate of the top neuron

            layer_positions = []
            for j in range(num_neurons):
                y = top_y + j * vertical_spacing
                layer_positions.append((layer_x, y))  # Append (x, y) of neuron
            self.neuron_positions.append(layer_positions)

        # Compute error positions aligned with output layer
        error_x = layer_spacing * (len(structure) + 1) - 0.75 * layer_spacing
        for j in range(structure[-1]):
            y = (height - (structure[-1] - 1) * vertical_spacing) / 2 + j * vertical_spacing
            self.error_positions.append((error_x, y))

        # Draw the entire network on the canvas using the calculated positions
        self.draw_network_all()
        
    def draw_network_all(self):
        """Draw the entire neural network on the canvas including neurons, biases, and output errors."""

        self.canvas.delete("all")  # Clear all previous drawings

        self.neuron_radius = 20  # Radius of each neuron circle
        self.bias_size = 30      # Size of the bias rectangle

        self.neurons_dic = {}    # Dictionary to map canvas items to neuron info

        # Draw neurons and biases layer by layer
        for i, layer in enumerate(self.neuron_positions):
            for j, (layer_x, y) in enumerate(layer):
                # Draw neuron as an oval
                nn_neuron = self.canvas.create_oval(
                    layer_x - self.neuron_radius,
                    y - self.neuron_radius,
                    layer_x + self.neuron_radius,
                    y + self.neuron_radius,
                    fill=COL_NEURONS,
                    outline=COL_TEXT,
                    width=1.5,
                    tags="neurons"
                )

                # Map canvas neuron ID to its position and layer index
                self.neurons_dic[nn_neuron] = {"index": (i, j), "cords": (layer_x, y)}
                self.neurons_dic[(i, j)] = nn_neuron

                # Draw bias rectangle for non-input layers
                if i > 0:
                    nn_bias = self.canvas.create_rectangle(
                        layer_x + self.neuron_radius, 
                        y - self.neuron_radius, 
                        layer_x + self.neuron_radius + self.bias_size, 
                        y - self.neuron_radius - self.bias_size, 
                        fill=COL_BACKGROUND, 
                        outline=COL_TEXT, 
                        width=1, 
                        tags="biases"
                    )

        # Draw output error rectangles aligned with the output layer
        for j, (x, y) in enumerate(self.error_positions):
            nn_error = self.canvas.create_rectangle(
                x, y - self.neuron_radius,
                x + self.bias_size, y + self.neuron_radius,
                fill=COL_BACKGROUND,
                outline=COL_NEURONS,
                width=1.5,
                tags="errors"
            )

        # Draw the neuron and bias values as text
        self.draw_network_text()

        # Initialize the sample image display to empty
        self.update_sample_photo(None)

    def draw_network_connections(self, draw_all=False):
        """Draw connections between neurons on the canvas.
        
        If draw_all is True, draws all connections between layers. Otherwise, only
        draws connections for the hovered or clicked neuron.
        """
        
        self.canvas.delete("connections")  # Remove previous connection lines
        self.active_connections = []       # Reset list of active connections
        
        neurons = []

        # Add clicked neuron to the list if any
        if self.clicked_neuron is not None:
            neuron = self.neurons_dic[self.clicked_neuron]
            neurons.append(neuron)
        
        # Add hovered neuron if different from the clicked one
        if self.hovered_neuron is not None and self.hovered_neuron != self.clicked_neuron:
            neuron = self.neurons_dic[self.hovered_neuron]
            neurons.append(neuron)

        # Draw connections for each neuron in focus (clicked or hovered)
        for neuron in neurons:
            i, j = neuron["index"]
            src_x, src_y = neuron["cords"]

            # Draw connections to the next layer
            if i + 1 < len(self.neuron_positions):
                for o, (dest_x, dest_y) in enumerate(self.neuron_positions[i + 1]):
                    nn_line = self.canvas.create_line(
                        src_x, src_y, dest_x, dest_y,
                        fill=COL_CONNECTIONS, width=1, tags="connections"
                    )
                    self.canvas.tag_lower(nn_line)  # Ensure lines appear below neurons
                    self.active_connections.append([
                        i, j, i + 1, o, (src_x + dest_x)/2, (src_y + dest_y)/2
                    ])

            # Draw connections to the previous layer
            if i - 1 >= 0:
                for o, (dest_x, dest_y) in enumerate(self.neuron_positions[i - 1]):
                    nn_line = self.canvas.create_line(
                        src_x, src_y, dest_x, dest_y,
                        fill=COL_CONNECTIONS, width=1, tags="connections"
                    )
                    self.canvas.tag_lower(nn_line)
                    self.active_connections.append([
                        i - 1, o, i, j, (src_x + dest_x)/2, (src_y + dest_y)/2
                    ])

        # Optionally draw all connections for the entire network
        if draw_all:
            for i in range(len(self.neuron_positions) - 1):
                for src_x, src_y in self.neuron_positions[i]:
                    for o, (dest_x, dest_y) in enumerate(self.neuron_positions[i + 1]):
                        nn_line = self.canvas.create_line(
                            src_x, src_y, dest_x, dest_y,
                            fill=COL_CONNECTIONS, width=1, tags="connections"
                        )
                        self.canvas.tag_lower(nn_line)
                        self.active_connections.append([
                            i, j, i + 1, o, (src_x + dest_x)/2, (src_y + dest_y)/2
                        ])

        # Draw the connection values/text if needed
        self.draw_network_text_connections()

    def draw_network_text(self):
        """Draw neuron values, biases, and error values on the canvas."""
        
        self.canvas.delete("text")  # Remove all previous text elements

        layers_values = self.nn.get_layers_with_values()  # Current neuron activations
        biases_values = self.nn.get_biases()             # Current biases per layer

        # Draw neuron values and biases
        for i, layer in enumerate(self.neuron_positions):
            for j, (layer_x, y) in enumerate(layer):
                # Draw neuron activation value
                nn_text_neuron = self.canvas.create_text(
                    layer_x, y,
                    text=f"{layers_values[i][j][0]:.2f}" if len(layers_values[i]) > 0 else "??",
                    font=("Arial", 10, "bold"),
                    fill="white" if len(layers_values[i]) > 0 and layers_values[i][j][0] < 0.5 else COL_TEXT,
                    tags="text"
                )

                # Draw bias for neurons in layers after input
                if i > 0:
                    nn_text_bias = self.canvas.create_text(
                        layer_x + self.neuron_radius + self.bias_size / 2,
                        y - self.neuron_radius - self.bias_size / 2,
                        text=f"{biases_values[i - 1][j][0]:.2f}",
                        font=("Arial", 10, "bold"),
                        fill=COL_TEXT,
                        tags="text"
                    )

        # Draw error values for output layer if in ERROR state
        state, _ = self.nn.get_state()
        error_values = self.nn.get_errors_matrix()
        for j, (x, y) in enumerate(self.error_positions):
            nn_text_error = self.canvas.create_text(
                x + self.bias_size / 2, y,
                text=f"{error_values[j][0]:.2f}" if error_values is not None and state == "ERROR" else "??",
                font=("Arial", 10, "bold"),
                fill=COL_RED,
                tags="text"
            )

        # Update connection values and highlight activated neurons
        self.draw_network_text_connections()
        self.draw_activated_neurons()

    def draw_network_text_connections(self):
        """Draw weights for active and clicked connections on the canvas."""
        
        self.canvas.delete("text_connections")  # Remove any previous weight texts
        weights = self.nn.get_weights()          # Get current NN weights

        # Iterate over active and clicked connections
        for active_connection in self.active_connections + self.clicked_connections:
            offset = 15
            i, j, o, p, mid_x, mid_y = active_connection  # unpack connection info

            # Draw background rectangle for weight text for better visibility
            nn_text_weight_bg = self.canvas.create_rectangle(
                mid_x - offset, mid_y + offset,
                mid_x + offset, mid_y - offset,
                fill=COL_BACKGROUND,
                outline=COL_TEXT,
                width=1,
                tags="text_connections"
            )

            # Draw the actual weight value
            weight_value = weights[i][p][j]
            nn_text_weight = self.canvas.create_text(
                mid_x, mid_y,
                text=f"{weight_value:.2f}",
                font=("Arial", 8),
                fill=COL_TEXT,
                tags="text_connections"
            )


    def draw_active_layer_mark(self):
        """Highlight the currently active layer with a dashed rectangle."""
        
        self.canvas.delete("active_layer")  # Remove previous highlight
        state, layer_index = self.nn.get_state()
        
        if layer_index is None:
            return  # Nothing to highlight if no active layer

        offset = 15
        layer = self.neuron_positions[layer_index]

        if state == "BACKWARD":
            # Highlight current and next layer during backpropagation
            layer_next = self.neuron_positions[layer_index + 1]
            min_x = min(pos[0] for pos in layer) + self.neuron_radius + offset
            max_x = max(pos[0] for pos in layer_next) + self.neuron_radius + offset
            min_y = min(pos[1] for pos in layer_next) - self.neuron_radius - offset
            max_y = max(pos[1] for pos in layer_next) + self.neuron_radius + offset
        else:
            # Highlight only current layer otherwise
            min_x = min(pos[0] for pos in layer) - self.neuron_radius - offset
            max_x = max(pos[0] for pos in layer) + self.neuron_radius + offset
            min_y = min(pos[1] for pos in layer) - self.neuron_radius - offset
            max_y = max(pos[1] for pos in layer) + self.neuron_radius + offset

        # Draw the dashed rectangle to indicate the active layer
        self.canvas.create_rectangle(
            min_x, min_y, max_x, max_y,
            outline=COL_FOCUS,
            width=3,
            dash=(5, 3),
            tags="active_layer"
        )
        self.canvas.tag_lower("active_layer")  # Ensure rectangle is below other items

    def draw_activated_neurons(self):
        """Update neuron colors based on their activation values."""
        
        layers_values = self.nn.get_layers_with_values()  # Get current neuron activations

        for i, layer in enumerate(self.neuron_positions):
            for j, (layer_x, y) in enumerate(layer):
                nn_neuron = self.neurons_dic[(i, j)]  # Get canvas item for this neuron
                
                # Retrieve neuron activation value; default to 0 if not available
                value = layers_values[i][j][0] if len(layers_values[i]) > 0 else 0
                
                # Map activation value to 0–255 intensity
                intensity = int(min(max(value * 255, 0), 255))
                
                # Compute a blended color between COL_NEURONS (base) and COL_FOCUS (highlight)
                color = (
                    f'#{int((int(COL_NEURONS[1:3],16) * (255 - intensity) + int(COL_FOCUS[1:3],16) * intensity)/255):02x}'
                    f'{int((int(COL_NEURONS[3:5],16) * (255 - intensity) + int(COL_FOCUS[3:5],16) * intensity)/255):02x}'
                    f'{int((int(COL_NEURONS[5:7],16) * (255 - intensity) + int(COL_FOCUS[5:7],16) * intensity)/255):02x}'
                )
                
                # Update the neuron fill color on the canvas
                self.canvas.itemconfig(nn_neuron, fill=color)

                
                     
        
        
    