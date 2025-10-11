COL_BACKGROUND = '#ede7e3'
COL_NEURONS = "#489fb5"
COL_TEXT = "#333333"
COL_CONNECTIONS = "#82c0cc"
COL_FOCUS = "#ffa62b"
COL_HERO="#16697a"
COL_RED = "#bf1f2e"


def configure_styles(style):
        # self.style.theme_use("clam")
        
        # self.style.configure("TButton",
        #                         font=("Segoe UI", 10),
        #                         padding=8,
        #                         background=COL_HERO,
        #                         foreground="white",
        #                         borderwidth=0)
        # self.style.map("TButton",
        #                 background=[("active", COL_CONNECTIONS)],
        #                 relief=[("pressed", "sunken")])
        
        
        style.configure("BG.TFrame",
                                background=COL_BACKGROUND)
        
        style.configure("ControlPanel.TFrame",
                                background=COL_CONNECTIONS)
        
        style.configure("H1.TLabel",
                                font=("Segoe UI", 16, "bold"),
                                background=COL_CONNECTIONS,
                                foreground="white")
        
        style.configure("H2.TLabel",
                                font=("Segoe UI", 14, "bold"),
                                background=COL_CONNECTIONS,
                                foreground="white")
        
        style.configure("H3.TLabel",
                                font=("Segoe UI", 12),
                                background=COL_CONNECTIONS,
                                foreground="white")
        
        # self.style.configure("TScale",
        #                         troughcolor=COL_CONNECTIONS,
        #                         background=COL_BACKGROUND)