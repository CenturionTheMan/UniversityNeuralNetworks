import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from animationWindow import AnimationWindow
from neural_network import NeuralNetwork
from digits_manager import DigitsDataset
from style import *


class ConfiguratorWindow:
    """Graphical user interface for configuring and visualizing a neural network."""

    def __init__(self, root):
        """Initialize the main configuration window and layout."""
        # Load dataset
        digitsCls = DigitsDataset()
        self.dataset = digitsCls.get_dataset()

        # Store hidden layers
        self.layers = []

        # --- Window setup ---
        self.root = root
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.title("Neural Network Configurator")
        self.root.minsize(600, 200)
        self.root.resizable(True, False)
        self.root.configure(bg=COL_BACKGROUND)

        # --- Style setup ---
        self.style = ttk.Style()
        configure_styles(self.style)
        self.root.style = self.style

        # --- Main layout frame ---
        self.main_frame = ttk.Frame(root, padding=10, style="BG.TFrame")
        self.main_frame.pack(fill="both", expand=True)
        self.main_frame.grid_columnconfigure(0, weight=2)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # Create sub-sections
        self.create_left_row()
        self.create_right_row()

        # Initialize with one hidden layer by default
        self.add_layer()

    # =====================================================
    # LEFT SIDE: Network Structure Configuration
    # =====================================================

    def create_left_row(self):
        """Create the left section for defining network layers."""
        self.left_frame = ttk.Frame(
            self.main_frame, borderwidth=2, relief="groove", padding=10, style="BG.TFrame"
        )
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Section title
        ttk.Label(
            self.left_frame,
            text="Neural Network structure",
            style="Title.TLabel",
            font=("Arial", 12, "bold")
        ).grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # Input layer (fixed size)
        self.input_scale = self.create_fixed_layer(
            self.left_frame, row=1, label="Input layer:", neurons=64
        )

        # Container for dynamically added hidden layers
        self.hidden_layers_frame = ttk.Frame(self.left_frame, style="BG.TFrame")
        self.hidden_layers_frame.grid(row=2, column=0, columnspan=3, pady=5, sticky="ew")

        # Output layer (fixed size)
        self.output_scale = self.create_fixed_layer(
            self.left_frame, row=3, label="Output layer:", neurons=10
        )

        # Button to add more hidden layers
        self.add_layer_button = ttk.Button(
            self.left_frame, text="Add Hidden Layer", command=self.add_layer
        )
        self.add_layer_button.grid(row=4, column=0, columnspan=3, pady=10)

    # =====================================================
    # RIGHT SIDE: Control Panel
    # =====================================================

    def create_right_row(self):
        """Create the right section containing control buttons."""
        self.right_frame = ttk.Frame(
            self.main_frame, borderwidth=2, relief="groove", padding=10, style="BG.TFrame"
        )
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Section title
        ttk.Label(
            self.right_frame,
            text="Controls",
            style="Title.TLabel",
            font=("Arial", 12, "bold")
        ).pack(pady=5)

        # Button to launch animation window
        self.run_button = ttk.Button(
            self.right_frame, text="Run animation", command=self.run_animation
        )
        self.run_button.pack(pady=10)

    # =====================================================
    # LAYER CREATION UTILITIES
    # =====================================================

    def create_fixed_layer(self, parent, row, label, neurons=8):
        """
        Create a fixed (non-editable) layer row with a label and a disabled scale.

        :param parent: Parent frame.
        :param row: Grid row index.
        :param label: Label text.
        :param neurons: Default neuron count.
        """
        ttk.Label(parent, text=label, width=12).grid(row=row, column=0, sticky="w", padx=5, pady=3)

        # Disabled scale for fixed layers
        scale = ttk.Scale(parent, from_=1, to=64, orient="horizontal", style="TScale")
        scale.set(neurons)
        scale.state(["disabled"])
        scale.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        parent.grid_columnconfigure(1, weight=1)

        # Label showing neuron count
        value_label = ttk.Label(parent, text=str(neurons), width=4, relief="sunken", anchor="center")
        value_label.grid(row=row, column=2, padx=5)

        return scale

    def add_layer(self):
        """Dynamically add a new hidden layer to the configuration."""
        idx = len(self.layers) + 1

        # Frame for the new hidden layer row
        frame = ttk.Frame(self.hidden_layers_frame)
        frame.pack(fill="x", pady=3)

        # Label (Hidden i)
        label = ttk.Label(frame, text=f"Hidden {idx}:", width=12)
        label.pack(side="left", padx=5)

        # Integer variable connected to the scale
        neuron_var = tk.IntVar(value=8)

        # Adjustable neuron count scale
        scale = ttk.Scale(frame, from_=1, to=32, orient="horizontal",
                          variable=neuron_var, style="TScale")
        scale.pack(side="left", fill="x", expand=True, padx=5)

        # Live neuron count label
        value_label = ttk.Label(frame, textvariable=neuron_var, width=4,
                                relief="sunken", anchor="center")
        value_label.pack(side="left", padx=5)

        # Button to remove this layer
        remove_btn = ttk.Button(frame, text="Remove",
                                command=lambda f=frame: self.remove_layer(f))
        remove_btn.pack(side="left", padx=5)

        # Ensure scale values remain integer
        scale.config(command=lambda v, var=neuron_var: var.set(round(float(v))))

        # Store layer configuration
        self.layers.append((frame, scale, neuron_var))
        self.update_labels()

    def remove_layer(self, frame):
        """
        Remove a hidden layer from configuration.
        Prevents removal if only one hidden layer remains.
        """
        if len(self.layers) == 1:
            messagebox.showwarning("Warning", "At least one hidden layer is necessary!")
            return

        # Locate and remove the target frame
        for i, (f, s, v) in enumerate(self.layers):
            if f == frame:
                f.destroy()
                self.layers.pop(i)
                break

        # Refresh labels after removal
        self.update_labels()

    def update_labels(self):
        """Update hidden layer labels"""
        for idx, (f, s, v) in enumerate(self.layers, start=1):
            label = f.winfo_children()[0]
            label.config(text=f"Hidden {idx}:")

    # =====================================================
    # ANIMATION LAUNCH
    # =====================================================

    def run_animation(self):
        """Construct the neural network and open the animation window."""
        input_size = int(self.input_scale.get())
        output_size = int(self.output_scale.get())

        # Extract hidden layer sizes from variables
        hidden_structure = [v.get() for (_, _, v) in self.layers]

        # Full structure: input + hidden + output
        structure = [input_size] + hidden_structure + [output_size]

        # Initialize neural network instance
        nn = NeuralNetwork(
            nn_structure=structure,
            learning_rate=0.01,
            epochs_num=10,
            dataset=self.dataset
        )

        # Launch the animation window
        animWin = AnimationWindow(self.root, nn)
        self.root.wait_window(animWin)
