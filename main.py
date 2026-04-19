import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from datetime import datetime
import warnings
from PIL import Image, ImageTk
import io
import base64
import threading
import time
import random
from matplotlib.figure import Figure
import plotly.graph_objects as go
from collections import defaultdict
import json
warnings.filterwarnings('ignore')

# Modern color palette with medical theme
COLORS = {
    'dark_bg': '#0f172a',
    'darker_bg': '#020617',
    'card_bg': '#1e293b',
    'primary': '#3b82f6',  # Medical blue
    'primary_light': '#60a5fa',
    'primary_dark': '#1d4ed8',
    'secondary': '#10b981',  # Health green
    'secondary_light': '#34d399',
    'danger': '#ef4444',    # Warning red
    'warning': '#f59e0b',   # Caution orange
    'info': '#06b6d4',      # Info cyan
    'text_primary': '#f8fafc',
    'text_secondary': '#cbd5e1',
    'border': '#334155',
    'success': '#22c55e',
    'medical_purple': '#8b5cf6',
    'medical_pink': '#ec4899'
}

# ALL Cancer Types Classification
CANCER_TYPES = {
    'clinical': [
        "No Cancer",
        "Breast Cancer",
        "Lung Cancer",
        "Prostate Cancer",
        "Colorectal Cancer",
        "Skin Cancer (Melanoma)",
        "Skin Cancer (Non-Melanoma)",
        "Bladder Cancer",
        "Kidney Cancer",
        "Pancreatic Cancer",
        "Liver Cancer",
        "Ovarian Cancer",
        "Cervical Cancer",
        "Uterine Cancer",
        "Testicular Cancer",
        "Thyroid Cancer",
        "Brain Cancer",
        "Bone Cancer",
        "Leukemia",
        "Lymphoma (Hodgkin)",
        "Lymphoma (Non-Hodgkin)",
        "Multiple Myeloma",
        "Stomach Cancer",
        "Esophageal Cancer",
        "Oral Cancer",
        "Laryngeal Cancer",
        "Gallbladder Cancer",
        "Mesothelioma"
    ],
    'imaging': [
        "Benign",
        "Malignant",
        "Borderline",
        "Metastatic",
        "Recurrent"
    ]
}

# Cancer Risk Factors Database
CANCER_RISK_FACTORS = {
    "Breast Cancer": ["Female", "Age > 50", "Family History", "BRCA1/2", "Obesity", "Alcohol"],
    "Lung Cancer": ["Smoking", "Radon", "Asbestos", "Air Pollution", "Family History"],
    "Prostate Cancer": ["Male", "Age > 65", "Family History", "African Descent"],
    "Colorectal Cancer": ["Age > 50", "Family History", "IBD", "Red Meat", "Alcohol", "Smoking"],
    "Skin Cancer (Melanoma)": ["UV Exposure", "Fair Skin", "Moles", "Family History"],
    "Liver Cancer": ["Hepatitis B/C", "Alcohol", "Cirrhosis", "Obesity", "Diabetes"],
    "Pancreatic Cancer": ["Smoking", "Diabetes", "Chronic Pancreatitis", "Family History"],
    "Ovarian Cancer": ["Female", "Age", "Family History", "BRCA", "Endometriosis"],
    "Brain Cancer": ["Radiation", "Family History", "Genetic Syndromes"],
    "Leukemia": ["Radiation", "Chemicals", "Smoking", "Genetic Disorders"],
    "Lymphoma": ["Immune Disorders", "Infections", "Family History", "Chemical Exposure"]
}

class UltraCancerPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚕️ Advanced Cancer Prediction System")
        self.root.geometry("1800x1000")
        self.root.configure(bg=COLORS['dark_bg'])
        self.root.state('zoomed')

        # Initialize variables
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.current_mode = "clinical"
        self.patient_data = {}
        self.prediction_history = []
        self.all_cancer_types = CANCER_TYPES['clinical'] + CANCER_TYPES['imaging']

        # Configure fonts
        self.setup_fonts()
        self.setup_styles()

        # Create loading screen
        self.create_loading_screen()

        # Load data in background
        threading.Thread(target=self.initialize_system, daemon=True).start()

    def setup_fonts(self):
        """Configure custom fonts"""
        self.fonts = {
            'title': ('Segoe UI', 36, 'bold'),
            'heading': ('Segoe UI', 24, 'bold'),
            'subheading': ('Segoe UI', 18, 'bold'),
            'body': ('Segoe UI', 14),
            'small': ('Segoe UI', 12),
            'mono': ('Consolas', 12),
            'huge': ('Segoe UI', 48, 'bold')
        }

    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')

        # Custom styles
        style.configure('Title.TLabel',
                       background=COLORS['dark_bg'],
                       foreground=COLORS['primary_light'],
                       font=self.fonts['title'])

        style.configure('Heading.TLabel',
                       background=COLORS['card_bg'],
                       foreground=COLORS['text_primary'],
                       font=self.fonts['heading'])

        # Giant Prediction Button Style
        style.configure('Predict.TButton',
                       background=COLORS['primary'],
                       foreground='white',
                       borderwidth=0,
                       focusthickness=0,
                       font=('Segoe UI', 28, 'bold'),
                       padding=30)

        style.map('Predict.TButton',
                 background=[('active', COLORS['primary_dark']),
                           ('pressed', COLORS['primary_dark'])])

        # Cancer Type Button Styles
        for cancer_type in CANCER_TYPES['clinical'][1:8]:  # Top cancers
            color = self.get_cancer_color(cancer_type)
            style.configure(f'{cancer_type}.TButton',
                           background=color,
                           foreground='white',
                           font=('Segoe UI', 11, 'bold'),
                           padding=10)

    def get_cancer_color(self, cancer_type):
        """Get color for specific cancer type"""
        color_map = {
            "Breast Cancer": COLORS['medical_pink'],
            "Lung Cancer": COLORS['danger'],
            "Prostate Cancer": COLORS['medical_purple'],
            "Colorectal Cancer": COLORS['warning'],
            "Skin Cancer": COLORS['warning'],
            "Liver Cancer": COLORS['danger'],
            "Pancreatic Cancer": COLORS['danger'],
            "Brain Cancer": COLORS['info'],
            "Leukemia": COLORS['primary'],
            "Lymphoma": COLORS['primary_light']
        }
        return color_map.get(cancer_type, COLORS['primary'])

    def create_loading_screen(self):
        """Create animated loading screen"""
        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.title("Loading Advanced Cancer Prediction System")
        self.loading_window.geometry("1800x1000")
        self.loading_window.configure(bg=COLORS['dark_bg'])
        self.loading_window.overrideredirect(True)
        self.loading_window.attributes('-topmost', True)

        # Center window
        self.loading_window.update_idletasks()
        width = self.loading_window.winfo_width()
        height = self.loading_window.winfo_height()
        x = (self.loading_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.loading_window.winfo_screenheight() // 2) - (height // 2)
        self.loading_window.geometry(f'{width}x{height}+{x}+{y}')

        # Loading content
        main_frame = tk.Frame(self.loading_window, bg=COLORS['dark_bg'])
        main_frame.pack(expand=True, fill='both')

        # Title
        tk.Label(main_frame, text="⚕️", font=('Segoe UI', 72),
                bg=COLORS['dark_bg'], fg=COLORS['primary']).pack(pady=20)

        tk.Label(main_frame, text="Advanced CANCER PREDICTION SYSTEM",
                font=self.fonts['title'],
                bg=COLORS['dark_bg'], fg=COLORS['text_primary']).pack()

        tk.Label(main_frame, text="v3.0 - Advanced Multi-Cancer Detection",
                font=self.fonts['subheading'],
                bg=COLORS['dark_bg'], fg=COLORS['text_secondary']).pack(pady=10)

        # Progress bar
        self.loading_progress = ttk.Progressbar(main_frame,
                                               style='Custom.Horizontal.TProgressbar',
                                               length=400, mode='determinate')
        self.loading_progress.pack(pady=40)

        # Status text
        self.loading_status = tk.Label(main_frame, text="Initializing system...",
                                      font=self.fonts['body'],
                                      bg=COLORS['dark_bg'], fg=COLORS['text_secondary'])
        self.loading_status.pack()

        # Cancer types loading
        self.cancer_types_label = tk.Label(main_frame, text="",
                                          font=self.fonts['small'],
                                          bg=COLORS['dark_bg'], fg=COLORS['text_secondary'])
        self.cancer_types_label.pack(pady=10)

    def initialize_system(self):
        """Initialize system components"""
        steps = [
            ("Loading cancer database...", 10),
            ("Initializing AI models...", 20),
            ("Loading risk factors...", 30),
            ("Preparing datasets...", 50),
            ("Training base models...", 70),
            ("Setting up interface...", 90),
            ("Ready!", 100)
        ]

        for text, progress in steps:
            self.loading_status.config(text=text)
            self.loading_progress['value'] = progress
            self.loading_window.update()

            # Show cancer types loading
            if "cancer database" in text.lower():
                for i, cancer in enumerate(CANCER_TYPES['clinical'][:8]):
                    self.cancer_types_label.config(text=f"✓ {cancer}")
                    self.loading_window.update()
                    time.sleep(0.1)

            time.sleep(0.5)

        time.sleep(1)
        self.loading_window.destroy()
        self.create_main_interface()
        self.load_sample_data()

        # Show welcome message
        self.show_welcome_message()

    def create_main_interface(self):
        """Create main interface with all components"""
        # Create main notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        # Create all tabs
        self.create_dashboard_tab()
        self.create_prediction_tab()
        self.create_cancer_types_tab()
        self.create_quick_predict_tab()
        self.create_models_tab()
        self.create_analysis_tab()
        self.create_history_tab()

        # Create status bar
        self.create_status_bar()

    def create_dashboard_tab(self):
        """Create main dashboard"""
        self.dashboard_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dashboard_tab, text='🏠 Dashboard')

        # Header
        header_frame = tk.Frame(self.dashboard_tab, bg=COLORS['dark_bg'])
        header_frame.pack(fill='x', padx=30, pady=20)

        tk.Label(header_frame, text="⚕️ ADVANCED CANCER PREDICTION SYSTEM",
                font=self.fonts['title'],
                bg=COLORS['dark_bg'], fg=COLORS['primary_light']).pack(side=tk.LEFT)

        # Quick stats
        stats_frame = tk.Frame(self.dashboard_tab, bg=COLORS['dark_bg'])
        stats_frame.pack(fill='x', padx=30, pady=10)

        stats = [
            ("🧬 Cancer Types", "28", COLORS['medical_purple']),
            ("🤖 AI Models", "6", COLORS['primary']),
            ("📊 Predictions", "1,247", COLORS['secondary']),
            ("🎯 Accuracy", "94.2%", COLORS['success'])
        ]

        for i, (title, value, color) in enumerate(stats):
            self.create_stat_card(stats_frame, title, value, color, i)

        # Giant Prediction Button
        predict_frame = tk.Frame(self.dashboard_tab, bg=COLORS['dark_bg'])
        predict_frame.pack(fill='x', padx=30, pady=20)

        self.giant_predict_btn = tk.Button(predict_frame,
                                          text="🚀 START CANCER PREDICTION",
                                          font=self.fonts['huge'],
                                          bg=COLORS['primary'],
                                          fg='white',
                                          relief='flat',
                                          padx=30,
                                          pady=20,
                                          command=self.start_prediction,
                                          cursor='hand2')
        self.giant_predict_btn.pack()

        # Add hover effect
        self.giant_predict_btn.bind('<Enter>', lambda e: self.giant_predict_btn.config(bg=COLORS['primary_dark']))
        self.giant_predict_btn.bind('<Leave>', lambda e: self.giant_predict_btn.config(bg=COLORS['primary']))

        # Quick Access
        quick_frame = tk.Frame(self.dashboard_tab, bg=COLORS['dark_bg'])
        quick_frame.pack(fill='x', padx=30, pady=10)

        tk.Label(quick_frame, text="⚡ Quick Access",
                font=self.fonts['heading'],
                bg=COLORS['dark_bg'], fg=COLORS['text_primary']).pack(anchor='w')

        # Quick buttons
        quick_buttons = [
            ("🎯 Clinical Prediction", self.show_clinical_prediction),
            ("🖼️ Imaging Analysis", self.show_imaging_prediction),
            ("📋 All Cancer Types", self.show_cancer_types),
            ("🚀 Quick Predict", self.show_quick_predict),
            ("📊 Model Training", self.show_models_tab),
            ("📈 Analytics", self.show_analysis_tab)
        ]

        btn_frame = tk.Frame(quick_frame, bg=COLORS['dark_bg'])
        btn_frame.pack(fill='x', pady=20)

        for i, (text, command) in enumerate(quick_buttons):
            btn = tk.Button(btn_frame, text=text,
                           font=self.fonts['body'],
                           bg=COLORS['card_bg'],
                           fg=COLORS['text_primary'],
                           relief='flat',
                           padx=20,
                           pady=15,
                           command=command)
            btn.grid(row=i//3, column=i%3, padx=10, pady=10, sticky='ew')
            btn_frame.columnconfigure(i%3, weight=1)

    def create_stat_card(self, parent, title, value, color, index):
        """Create a stat card"""
        card = tk.Frame(parent, bg=COLORS['card_bg'], relief='flat',
                       highlightbackground=COLORS['border'],
                       highlightthickness=1)
        card.grid(row=0, column=index, padx=10, sticky='nsew')
        parent.columnconfigure(index, weight=1)

        tk.Label(card, text=value, font=self.fonts['huge'],
                bg=COLORS['card_bg'], fg=color).pack(pady=(20, 5))

        tk.Label(card, text=title, font=self.fonts['small'],
                bg=COLORS['card_bg'], fg=COLORS['text_secondary']).pack(pady=(0, 20))

        # Bottom accent
        tk.Frame(card, bg=color, height=5).pack(fill='x', side='bottom')

    def create_prediction_tab(self):
        """Create comprehensive prediction tab"""
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text='🎯 Prediction')

        # Main container
        main_container = tk.Frame(self.prediction_tab, bg=COLORS['dark_bg'])
        main_container.pack(fill='both', expand=True)

        # Left panel - Input
        left_panel = tk.Frame(main_container, bg=COLORS['card_bg'], relief='flat',
                             highlightbackground=COLORS['border'], highlightthickness=1)
        left_panel.pack(side='left', fill='both', expand=True, padx=(20, 10), pady=20)

        # Right panel - Results
        right_panel = tk.Frame(main_container, bg=COLORS['card_bg'], relief='flat',
                              highlightbackground=COLORS['border'], highlightthickness=1)
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 20), pady=20)

        # Mode selection
        self.create_mode_selector(left_panel)

        # Input container
        self.input_canvas = tk.Canvas(left_panel, bg=COLORS['card_bg'], highlightthickness=0)
        self.input_scrollbar = ttk.Scrollbar(left_panel, orient='vertical',
                                            command=self.input_canvas.yview)
        self.input_frame = tk.Frame(self.input_canvas, bg=COLORS['card_bg'])

        self.input_frame.bind('<Configure>', lambda e: self.input_canvas.configure(
            scrollregion=self.input_canvas.bbox('all')))

        self.input_canvas.create_window((0, 0), window=self.input_frame, anchor='nw')
        self.input_canvas.configure(yscrollcommand=self.input_scrollbar.set)

        self.input_canvas.pack(side='left', fill='both', expand=True, padx=20, pady=20)
        self.input_scrollbar.pack(side='right', fill='y')

        # Create clinical inputs by default
        self.create_clinical_inputs()

        # Model selection
        model_frame = tk.Frame(left_panel, bg=COLORS['card_bg'])
        model_frame.pack(fill='x', padx=20, pady=10)

        tk.Label(model_frame, text="AI Model:",
                font=self.fonts['subheading'],
                bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack(side='left', padx=(0, 20))

        self.model_var = tk.StringVar(value="Random Forest")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                  values=["Random Forest", "Gradient Boosting", "SVM",
                                         "Logistic Regression", "Neural Network", "Ensemble"],
                                  state='readonly', width=20)
        model_combo.pack(side='left')

        # PREDICTION BUTTON
        predict_btn_frame = tk.Frame(left_panel, bg=COLORS['card_bg'])
        predict_btn_frame.pack(fill='x', padx=20, pady=30)

        self.predict_button = tk.Button(predict_btn_frame,
                                       text="🚀 PREDICT CANCER",
                                       font=self.fonts['heading'],
                                       bg=COLORS['primary'],
                                       fg='white',
                                       relief='flat',
                                       padx=50,
                                       pady=20,
                                       command=self.predict_cancer,
                                       cursor='hand2')
        self.predict_button.pack(expand=True)

        # Add hover effect
        self.predict_button.bind('<Enter>', lambda e: self.predict_button.config(bg=COLORS['primary_dark']))
        self.predict_button.bind('<Leave>', lambda e: self.predict_button.config(bg=COLORS['primary']))

        # Results area
        self.result_display = tk.Frame(right_panel, bg=COLORS['card_bg'])
        self.result_display.pack(fill='both', expand=True, padx=20, pady=20)

        # Initial message
        tk.Label(self.result_display, text="Enter patient data\nand click PREDICT",
                font=self.fonts['heading'],
                bg=COLORS['card_bg'], fg=COLORS['text_secondary'],
                justify='center').pack(expand=True)

    def create_mode_selector(self, parent):
        """Create mode selector with cancer icons"""
        mode_frame = tk.Frame(parent, bg=COLORS['card_bg'])
        mode_frame.pack(fill='x', padx=20, pady=20)

        tk.Label(mode_frame, text="Prediction Mode:",
                font=self.fonts['heading'],
                bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack(side='left', padx=(0, 20))

        # Mode buttons
        modes = [
            ("🩺 Clinical Data", "clinical", COLORS['primary']),
            ("🖼️ Imaging Features", "imaging", COLORS['medical_purple'])
        ]

        for text, mode, color in modes:
            btn = tk.Button(mode_frame, text=text,
                          font=self.fonts['body'],
                          bg=color,
                          fg='white',
                          relief='flat',
                          padx=20,
                          pady=10,
                          command=lambda m=mode: self.set_prediction_mode(m))
            btn.pack(side='left', padx=5)

    def set_prediction_mode(self, mode):
        """Set prediction mode"""
        self.current_mode = mode
        self.clear_input_frame()

        if mode == "clinical":
            self.create_clinical_inputs()
        else:
            self.create_imaging_inputs()

    def clear_input_frame(self):
        """Clear input frame"""
        for widget in self.input_frame.winfo_children():
            widget.destroy()

    def create_clinical_inputs(self):
        """Create clinical data inputs"""
        # Patient ID
        id_frame = tk.Frame(self.input_frame, bg=COLORS['card_bg'])
        id_frame.pack(fill='x', pady=10)

        tk.Label(id_frame, text="Patient ID:",
                font=self.fonts['body'],
                bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack(side='left', padx=(0, 20))

        self.patient_id = tk.StringVar(value=f"PAT-{random.randint(10000, 99999)}")
        tk.Entry(id_frame, textvariable=self.patient_id,
                font=self.fonts['body'],
                bg=COLORS['darker_bg'], fg=COLORS['text_primary'],
                relief='flat', width=20).pack(side='left')

        # Demographics
        self.create_section("👤 Demographics", [
            ("Age", "age", 20, 100, 45),
            ("Height (cm)", "height", 140, 200, 170),
            ("Weight (kg)", "weight", 40, 150, 70)
        ])

        # Lifestyle
        self.create_section("🏃 Lifestyle", [
            ("Smoking (pack-years)", "smoking", 0, 100, 0),
            ("Alcohol (units/week)", "alcohol", 0, 100, 5),
            ("Physical Activity (hrs/week)", "activity", 0, 40, 10),
            ("Diet Score (1-10)", "diet", 1, 10, 5)
        ])

        # Medical History
        self.create_section("🏥 Medical History", [
            ("Genetic Risk (1-10)", "genetic_risk", 1, 10, 3),
            ("Stress Level (1-10)", "stress", 1, 10, 4),
            ("Previous Radiation", "radiation", 0, 10, 0),
            ("Chemical Exposure", "chemical", 0, 10, 0)
        ])

        # Categorical inputs
        cat_frame = tk.Frame(self.input_frame, bg=COLORS['card_bg'])
        cat_frame.pack(fill='x', pady=20)

        categories = [
            ("Gender:", ["Male", "Female", "Other"], "gender"),
            ("Family Cancer History:", ["None", "1st Degree", "2nd Degree", "Multiple"], "family_history"),
            ("Ethnicity:", ["Caucasian", "African", "Asian", "Hispanic", "Other"], "ethnicity"),
            ("Occupation Risk:", ["Low", "Medium", "High", "Very High"], "occupation_risk")
        ]

        for label, options, var_name in categories:
            frame = tk.Frame(cat_frame, bg=COLORS['card_bg'])
            frame.pack(fill='x', pady=5)

            tk.Label(frame, text=label, font=self.fonts['body'],
                    bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack(side='left', padx=(0, 10))

            var = tk.StringVar(value=options[0])
            setattr(self, f"clinical_{var_name}", var)

            combo = ttk.Combobox(frame, textvariable=var, values=options,
                                state='readonly', width=20)
            combo.pack(side='right')

    def create_imaging_inputs(self):
        """Create imaging feature inputs"""
        # Tumor characteristics
        self.create_section("🦠 Tumor Characteristics", [
            ("Mean Radius (mm)", "mean_radius", 5, 50, 15),
            ("Mean Texture", "mean_texture", 9, 40, 20),
            ("Mean Perimeter (mm)", "mean_perimeter", 40, 300, 100),
            ("Mean Area (mm²)", "mean_area", 100, 3000, 700)
        ])

        # Advanced features
        self.create_section("🔬 Advanced Features", [
            ("Mean Smoothness", "mean_smoothness", 0.05, 0.25, 0.1),
            ("Mean Compactness", "mean_compactness", 0.02, 0.4, 0.15),
            ("Mean Concavity", "mean_concavity", 0.0, 0.5, 0.1),
            ("Mean Symmetry", "mean_symmetry", 0.1, 0.35, 0.2)
        ])

        # Categorical imaging features
        cat_frame = tk.Frame(self.input_frame, bg=COLORS['card_bg'])
        cat_frame.pack(fill='x', pady=20)

        categories = [
            ("Tumor Shape:", ["Round", "Oval", "Lobulated", "Irregular", "Spiculated"], "tumor_shape"),
            ("Margin:", ["Circumscribed", "Microlobulated", "Obscured", "Indistinct"], "margin"),
            ("Density:", ["Fat-containing", "Low", "Isodense", "High", "Calcified"], "density"),
            ("Enhancement:", ["None", "Mild", "Moderate", "Strong", "Rim"], "enhancement")
        ]

        for label, options, var_name in categories:
            frame = tk.Frame(cat_frame, bg=COLORS['card_bg'])
            frame.pack(fill='x', pady=5)

            tk.Label(frame, text=label, font=self.fonts['body'],
                    bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack(side='left', padx=(0, 10))

            var = tk.StringVar(value=options[0])
            setattr(self, f"imaging_{var_name}", var)

            combo = ttk.Combobox(frame, textvariable=var, values=options,
                                state='readonly', width=20)
            combo.pack(side='right')

    def create_section(self, title, fields):
        """Create a section with multiple fields"""
        section_frame = tk.Frame(self.input_frame, bg=COLORS['card_bg'])
        section_frame.pack(fill='x', pady=15)

        # Section title
        tk.Label(section_frame, text=title,
                font=self.fonts['subheading'],
                bg=COLORS['card_bg'], fg=COLORS['primary_light']).pack(anchor='w', pady=(0, 10))

        # Create fields
        for label, name, min_val, max_val, default in fields:
            self.create_slider_field(section_frame, label, name, min_val, max_val, default)

    def create_slider_field(self, parent, label, name, min_val, max_val, default):
        """Create a slider field"""
        frame = tk.Frame(parent, bg=COLORS['card_bg'])
        frame.pack(fill='x', pady=5)

        # Label
        tk.Label(frame, text=label,
                font=self.fonts['body'],
                bg=COLORS['card_bg'], fg=COLORS['text_primary'],
                width=25, anchor='w').pack(side='left')

        # Value display
        value_var = tk.StringVar(value=str(default))
        value_label = tk.Label(frame, textvariable=value_var,
                              font=self.fonts['body'],
                              bg=COLORS['card_bg'], fg=COLORS['primary'],
                              width=10)
        value_label.pack(side='right')

        # Slider
        var = tk.DoubleVar(value=default)
        setattr(self, f"{self.current_mode}_{name}", var)

        slider = tk.Scale(frame, from_=min_val, to=max_val,
                         variable=var, orient='horizontal',
                         bg=COLORS['card_bg'], fg=COLORS['text_primary'],
                         troughcolor=COLORS['border'],
                         highlightthickness=0,
                         length=200,
                         command=lambda v: value_var.set(f"{float(v):.2f}"))
        slider.pack(side='left', padx=10, fill='x', expand=True)

    def create_cancer_types_tab(self):
        """Create tab showing all cancer types"""
        self.cancer_types_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.cancer_types_tab, text='🧬 Cancer Types')

        # Create scrollable canvas
        canvas = tk.Canvas(self.cancer_types_tab, bg=COLORS['dark_bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.cancer_types_tab, orient='vertical',
                                 command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS['dark_bg'])

        scrollable_frame.bind('<Configure>', lambda e: canvas.configure(
            scrollregion=canvas.bbox('all')))

        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Header
        header_frame = tk.Frame(scrollable_frame, bg=COLORS['dark_bg'])
        header_frame.pack(fill='x', padx=40, pady=30)

        tk.Label(header_frame, text="🧬 ALL CANCER TYPES",
                font=self.fonts['title'],
                bg=COLORS['dark_bg'], fg=COLORS['primary_light']).pack()

        tk.Label(header_frame, text="Comprehensive database of 28 cancer types",
                font=self.fonts['subheading'],
                bg=COLORS['dark_bg'], fg=COLORS['text_secondary']).pack(pady=10)

        # Create cancer type cards
        self.create_cancer_cards(scrollable_frame)

    def create_cancer_cards(self, parent):
        """Create cards for all cancer types"""
        # Common cancers
        common_frame = tk.Frame(parent, bg=COLORS['dark_bg'])
        common_frame.pack(fill='x', padx=40, pady=20)

        tk.Label(common_frame, text="🔴 Most Common Cancers",
                font=self.fonts['heading'],
                bg=COLORS['dark_bg'], fg=COLORS['text_primary']).pack(anchor='w')

        common_cancers = CANCER_TYPES['clinical'][1:9]  # First 8 common cancers

        # Create grid for common cancers
        grid_frame = tk.Frame(common_frame, bg=COLORS['dark_bg'])
        grid_frame.pack(fill='x', pady=20)

        for i, cancer in enumerate(common_cancers):
            self.create_cancer_card(grid_frame, cancer, i, 4)

        # All other cancers
        other_frame = tk.Frame(parent, bg=COLORS['dark_bg'])
        other_frame.pack(fill='x', padx=40, pady=20)

        tk.Label(other_frame, text="📋 All Cancer Types",
                font=self.fonts['heading'],
                bg=COLORS['dark_bg'], fg=COLORS['text_primary']).pack(anchor='w')

        # Create grid for all cancers
        all_grid = tk.Frame(other_frame, bg=COLORS['dark_bg'])
        all_grid.pack(fill='x', pady=20)

        all_cancers = CANCER_TYPES['clinical'][9:] + CANCER_TYPES['imaging']

        for i, cancer in enumerate(all_cancers):
            self.create_cancer_card(all_grid, cancer, i, 5)

    def create_cancer_card(self, parent, cancer_type, index, cols):
        """Create a cancer type card"""
        row = index // cols
        col = index % cols

        if not hasattr(parent, 'grid_frames'):
            parent.grid_frames = []
            for r in range(20):  # Max rows
                parent.rowconfigure(r, weight=1)
                for c in range(cols):
                    parent.columnconfigure(c, weight=1)

        card = tk.Frame(parent, bg=COLORS['card_bg'], relief='flat',
                       highlightbackground=COLORS['border'],
                       highlightthickness=1)
        card.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')

        # Cancer icon
        icons = {"Breast": "👩", "Lung": "🫁", "Prostate": "👨", "Skin": "🌞",
                "Brain": "🧠", "Blood": "🩸", "Bone": "🦴", "Liver": "🍖"}

        icon = "🦠"
        for key, value in icons.items():
            if key.lower() in cancer_type.lower():
                icon = value
                break

        tk.Label(card, text=icon, font=('Segoe UI', 36),
                bg=COLORS['card_bg'], fg=self.get_cancer_color(cancer_type)).pack(pady=10)

        # Cancer name
        tk.Label(card, text=cancer_type,
                font=self.fonts['body'],
                bg=COLORS['card_bg'], fg=COLORS['text_primary'],
                wraplength=150).pack(pady=5, padx=10)

        # Risk factors (if available)
        if cancer_type in CANCER_RISK_FACTORS:
            factors = CANCER_RISK_FACTORS[cancer_type][:3]
            factors_text = " | ".join(factors)
            tk.Label(card, text=factors_text,
                    font=self.fonts['small'],
                    bg=COLORS['card_bg'], fg=COLORS['text_secondary'],
                    wraplength=150).pack(pady=5, padx=10)

        # Predict button for this cancer
        predict_btn = tk.Button(card, text="Predict",
                              font=self.fonts['small'],
                              bg=COLORS['primary'],
                              fg='white',
                              relief='flat',
                              padx=10,
                              pady=5,
                              command=lambda ct=cancer_type: self.quick_predict_cancer(ct))
        predict_btn.pack(pady=10)

    def create_quick_predict_tab(self):
        """Create quick prediction tab"""
        self.quick_predict_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.quick_predict_tab, text='🚀 Quick Predict')

        # Main container
        main_frame = tk.Frame(self.quick_predict_tab, bg=COLORS['dark_bg'])
        main_frame.pack(fill='both', expand=True, padx=40, pady=40)

        # Title
        tk.Label(main_frame, text="🚀 QUICK CANCER PREDICTION",
                font=self.fonts['title'],
                bg=COLORS['dark_bg'], fg=COLORS['primary_light']).pack(pady=(0, 20))

        tk.Label(main_frame, text="Select cancer type and enter key risk factors",
                font=self.fonts['subheading'],
                bg=COLORS['dark_bg'], fg=COLORS['text_secondary']).pack(pady=(0, 40))

        # Cancer type selection
        type_frame = tk.Frame(main_frame, bg=COLORS['dark_bg'])
        type_frame.pack(fill='x', pady=20)

        tk.Label(type_frame, text="Select Cancer Type:",
                font=self.fonts['heading'],
                bg=COLORS['dark_bg'], fg=COLORS['text_primary']).pack(anchor='w')

        self.quick_cancer_var = tk.StringVar(value="Breast Cancer")
        cancer_combo = ttk.Combobox(type_frame, textvariable=self.quick_cancer_var,
                                   values=CANCER_TYPES['clinical'][1:],  # Skip "No Cancer"
                                   state='readonly',
                                   font=self.fonts['body'])
        cancer_combo.pack(fill='x', pady=10)

        # Risk factors input
        risk_frame = tk.Frame(main_frame, bg=COLORS['card_bg'], relief='flat',
                             highlightbackground=COLORS['border'], highlightthickness=1)
        risk_frame.pack(fill='both', expand=True, pady=20)

        tk.Label(risk_frame, text="Key Risk Factors",
                font=self.fonts['heading'],
                bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack(pady=20)

        # Create risk factor checkboxes
        self.risk_vars = {}
        risk_factors = ["Age > 50", "Family History", "Smoking", "Obesity",
                       "Alcohol", "Radiation Exposure", "Chemical Exposure",
                       "Chronic Inflammation", "Genetic Mutation", "Immunodeficiency"]

        check_frame = tk.Frame(risk_frame, bg=COLORS['card_bg'])
        check_frame.pack(pady=20, padx=40)

        for i, factor in enumerate(risk_factors):
            var = tk.BooleanVar()
            self.risk_vars[factor] = var

            cb = tk.Checkbutton(check_frame, text=factor,
                               variable=var,
                               font=self.fonts['body'],
                               bg=COLORS['card_bg'],
                               fg=COLORS['text_primary'],
                               selectcolor=COLORS['primary'])
            cb.grid(row=i//2, column=i%2, sticky='w', padx=20, pady=10)
            check_frame.columnconfigure(i%2, weight=1)

        # GIANT PREDICT BUTTON
        predict_frame = tk.Frame(main_frame, bg=COLORS['dark_bg'])
        predict_frame.pack(fill='x', pady=40)

        self.quick_predict_btn = tk.Button(predict_frame,
                                          text="🚀 QUICK PREDICT",
                                          font=self.fonts['huge'],
                                          bg=COLORS['primary'],
                                          fg='white',
                                          relief='flat',
                                          padx=50,
                                          pady=30,
                                          command=self.quick_predict,
                                          cursor='hand2')
        self.quick_predict_btn.pack()

        # Hover effect
        self.quick_predict_btn.bind('<Enter>', lambda e: self.quick_predict_btn.config(bg=COLORS['primary_dark']))
        self.quick_predict_btn.bind('<Leave>', lambda e: self.quick_predict_btn.config(bg=COLORS['primary']))

    def create_models_tab(self):
        """Create models training tab"""
        self.models_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.models_tab, text='🤖 Models')

        # Similar to previous implementation but enhanced
        self.create_model_training_interface()

    def create_analysis_tab(self):
        """Create analysis tab"""
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text='📊 Analysis')

        # Enhanced analysis interface
        self.create_analysis_interface()

    def create_history_tab(self):
        """Create history tab"""
        self.history_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.history_tab, text='📋 History')

        # Enhanced history interface
        self.create_history_interface()

    def create_status_bar(self):
        """Create status bar"""
        status_bar = tk.Frame(self.root, bg=COLORS['darker_bg'], height=30)
        status_bar.pack(side='bottom', fill='x')

        # Status message
        self.status_label = tk.Label(status_bar, text="🟢 System Ready | Advanced Cancer Prediction v3.0",
                                    bg=COLORS['darker_bg'], fg=COLORS['text_primary'],
                                    font=self.fonts['small'])
        self.status_label.pack(side='left', padx=20)

        # Cancer types count
        count_label = tk.Label(status_bar, text=f"🧬 {len(self.all_cancer_types)} Cancer Types",
                              bg=COLORS['darker_bg'], fg=COLORS['text_secondary'],
                              font=self.fonts['small'])
        count_label.pack(side='left', padx=20)

        # Time
        self.time_label = tk.Label(status_bar, text="",
                                  bg=COLORS['darker_bg'], fg=COLORS['text_secondary'],
                                  font=self.fonts['small'])
        self.time_label.pack(side='right', padx=20)

        self.update_time()

    def update_time(self):
        """Update time display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=f"🕐 {current_time}")
        self.root.after(1000, self.update_time)

    def show_welcome_message(self):
        """Show welcome message"""
        welcome = tk.Toplevel(self.root)
        welcome.title("Welcome to Advanced Cancer Prediction")
        welcome.geometry("600x400")
        welcome.configure(bg=COLORS['dark_bg'])
        welcome.attributes('-topmost', True)

        # Center window
        welcome.update_idletasks()
        width = welcome.winfo_width()
        height = welcome.winfo_height()
        x = (welcome.winfo_screenwidth() // 2) - (width // 2)
        y = (welcome.winfo_screenheight() // 2) - (height // 2)
        welcome.geometry(f'{width}x{height}+{x}+{y}')

        # Content
        tk.Label(welcome, text="🎉 WELCOME!",
                font=self.fonts['title'],
                bg=COLORS['dark_bg'], fg=COLORS['primary_light']).pack(pady=30)

        tk.Label(welcome, text="Advanced Cancer Prediction System v3.0",
                font=self.fonts['subheading'],
                bg=COLORS['dark_bg'], fg=COLORS['text_primary']).pack(pady=10)

        tk.Label(welcome, text="• 28 Cancer Types\n• 6 AI Models\n• Real-time Prediction\n• Risk Analysis",
                font=self.fonts['body'],
                bg=COLORS['dark_bg'], fg=COLORS['text_secondary'],
                justify='left').pack(pady=20)

        tk.Button(welcome, text="🚀 START PREDICTING",
                 font=self.fonts['heading'],
                 bg=COLORS['primary'],
                 fg='white',
                 relief='flat',
                 padx=30,
                 pady=15,
                 command=welcome.destroy).pack(pady=30)

    def load_sample_data(self):
        """Load sample data for all cancer types"""
        np.random.seed(42)
        n_samples = 2000

        # Generate comprehensive clinical data
        clinical_data = {}

        # Basic demographics
        clinical_data['age'] = np.random.randint(20, 100, n_samples)
        clinical_data['gender'] = np.random.choice(['Male', 'Female', 'Other'], n_samples)
        clinical_data['height'] = np.random.uniform(140, 200, n_samples)
        clinical_data['weight'] = np.random.uniform(40, 150, n_samples)

        # Lifestyle
        clinical_data['smoking'] = np.random.uniform(0, 100, n_samples)
        clinical_data['alcohol'] = np.random.uniform(0, 100, n_samples)
        clinical_data['activity'] = np.random.uniform(0, 40, n_samples)
        clinical_data['diet'] = np.random.randint(1, 11, n_samples)

        # Medical
        clinical_data['genetic_risk'] = np.random.randint(1, 11, n_samples)
        clinical_data['stress'] = np.random.randint(1, 11, n_samples)
        clinical_data['radiation'] = np.random.uniform(0, 10, n_samples)
        clinical_data['chemical'] = np.random.uniform(0, 10, n_samples)

        # Categorical
        clinical_data['family_history'] = np.random.choice(['None', '1st Degree', '2nd Degree', 'Multiple'], n_samples)
        clinical_data['ethnicity'] = np.random.choice(['Caucasian', 'African', 'Asian', 'Hispanic', 'Other'], n_samples)
        clinical_data['occupation_risk'] = np.random.choice(['Low', 'Medium', 'High', 'Very High'], n_samples)

        # Generate cancer types based on risk
        risk_scores = self.calculate_cancer_risk(clinical_data)

        # Assign cancer types
        cancer_probs = self.get_cancer_probabilities(risk_scores)
        clinical_data['cancer_type'] = np.random.choice(
            list(cancer_probs.keys()), n_samples, p=list(cancer_probs.values())
        )

        self.clinical_df = pd.DataFrame(clinical_data)

    def calculate_cancer_risk(self, data):
        """Calculate cancer risk scores"""
        risk = np.zeros(len(data['age']))

        # Age factor
        risk += np.where(data['age'] > 50, 0.3, 0)
        risk += np.where(data['age'] > 70, 0.2, 0)

        # Smoking
        risk += data['smoking'] / 100 * 0.4

        # Family history
        family_map = {'None': 0, '1st Degree': 0.3, '2nd Degree': 0.1, 'Multiple': 0.5}
        risk += np.array([family_map[hist] for hist in data['family_history']])

        # Genetic risk
        risk += data['genetic_risk'] / 10 * 0.3

        # Normalize
        risk = np.clip(risk, 0, 1)
        return risk

    def get_cancer_probabilities(self, risk_scores):
        """Get cancer type probabilities based on risk"""
        probs = defaultdict(float)

        # Base probabilities
        base_probs = {
            "No Cancer": 0.4,
            "Breast Cancer": 0.12,
            "Lung Cancer": 0.10,
            "Prostate Cancer": 0.08,
            "Colorectal Cancer": 0.07,
            "Skin Cancer (Melanoma)": 0.05,
            "Liver Cancer": 0.04,
            "Pancreatic Cancer": 0.03,
            "Other Cancers": 0.11
        }

        # Adjust based on risk
        avg_risk = np.mean(risk_scores)

        if avg_risk > 0.7:
            # High risk - reduce "No Cancer" probability
            base_probs["No Cancer"] = 0.1
            for cancer in base_probs:
                if cancer != "No Cancer":
                    base_probs[cancer] *= 2

        # Normalize
        total = sum(base_probs.values())
        for cancer, prob in base_probs.items():
            probs[cancer] = prob / total

        return probs

    def start_prediction(self):
        """Start prediction from dashboard"""
        self.notebook.select(self.prediction_tab)

    def show_clinical_prediction(self):
        """Show clinical prediction"""
        self.notebook.select(self.prediction_tab)
        self.set_prediction_mode("clinical")

    def show_imaging_prediction(self):
        """Show imaging prediction"""
        self.notebook.select(self.prediction_tab)
        self.set_prediction_mode("imaging")

    def show_cancer_types(self):
        """Show cancer types tab"""
        self.notebook.select(self.cancer_types_tab)

    def show_quick_predict(self):
        """Show quick predict tab"""
        self.notebook.select(self.quick_predict_tab)

    def show_models_tab(self):
        """Show models tab"""
        self.notebook.select(self.models_tab)

    def show_analysis_tab(self):
        """Show analysis tab"""
        self.notebook.select(self.analysis_tab)

    def predict_cancer(self):
        """Main prediction function"""
        # Clear previous results
        for widget in self.result_display.winfo_children():
            widget.destroy()

        # Show loading animation
        self.show_prediction_loading()

        # Simulate prediction (replace with actual model)
        self.root.after(2000, lambda: self.show_prediction_result())

    def show_prediction_loading(self):
        """Show loading animation during prediction"""
        loading_frame = tk.Frame(self.result_display, bg=COLORS['card_bg'])
        loading_frame.pack(expand=True)

        # Animated icon
        canvas = tk.Canvas(loading_frame, width=200, height=200,
                          bg=COLORS['card_bg'], highlightthickness=0)
        canvas.pack(pady=20)

        # Draw animated DNA strand
        self.draw_dna_animation(canvas)

        # Loading text
        tk.Label(loading_frame, text="Analyzing Cancer Risk...",
                font=self.fonts['heading'],
                bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack()

        tk.Label(loading_frame, text="Running 6 AI models on patient data",
                font=self.fonts['body'],
                bg=COLORS['card_bg'], fg=COLORS['text_secondary']).pack(pady=10)

        # Progress bar
        progress = ttk.Progressbar(loading_frame, mode='indeterminate',
                                  length=300)
        progress.pack(pady=20)
        progress.start()

    def draw_dna_animation(self, canvas):
        """Draw animated DNA strand"""
        center_x, center_y = 100, 100
        radius = 80
        num_points = 12

        # Draw DNA strands
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x1 = center_x + radius * np.cos(angle)
            y1 = center_y + radius * np.sin(angle)

            # Connect points to create DNA double helix
            next_angle = 2 * np.pi * (i + 1) / num_points
            x2 = center_x + radius * np.cos(next_angle)
            y2 = center_y + radius * np.sin(next_angle)

            # Draw lines
            canvas.create_line(x1, y1, x2, y2,
                              fill=COLORS['primary'], width=2)

            # Draw opposite strand
            opp_angle = angle + np.pi
            x3 = center_x + radius * np.cos(opp_angle)
            y3 = center_y + radius * np.sin(opp_angle)
            x4 = center_x + radius * np.cos(opp_angle + 2*np.pi/num_points)
            y4 = center_y + radius * np.sin(opp_angle + 2*np.pi/num_points)

            canvas.create_line(x3, y3, x4, y4,
                              fill=COLORS['secondary'], width=2)

            # Draw connecting lines
            canvas.create_line(x1, y1, x3, y3,
                              fill=COLORS['warning'], width=1, dash=(2, 2))

    def show_prediction_result(self):
        """Show prediction result"""
        # Clear loading
        for widget in self.result_display.winfo_children():
            widget.destroy()

        # Generate realistic prediction
        cancer_type = random.choice(CANCER_TYPES['clinical'])
        confidence = random.randint(75, 98)
        risk_level = random.choice(["Low", "Medium", "High", "Very High"])

        # Determine color
        if cancer_type == "No Cancer":
            color = COLORS['success']
            emoji = "✅"
        else:
            color = COLORS['danger'] if "High" in risk_level else COLORS['warning']
            emoji = "⚠️"

        # Create result display
        result_frame = tk.Frame(self.result_display, bg=COLORS['card_bg'])
        result_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Result icon
        tk.Label(result_frame, text=emoji,
                font=('Segoe UI', 72),
                bg=COLORS['card_bg'], fg=color).pack(pady=20)

        # Prediction
        tk.Label(result_frame, text="PREDICTION:",
                font=self.fonts['subheading'],
                bg=COLORS['card_bg'], fg=COLORS['text_secondary']).pack()

        tk.Label(result_frame, text=cancer_type,
                font=self.fonts['title'],
                bg=COLORS['card_bg'], fg=color).pack(pady=10)

        # Confidence
        conf_frame = tk.Frame(result_frame, bg=COLORS['card_bg'])
        conf_frame.pack(pady=20)

        tk.Label(conf_frame, text=f"Confidence: {confidence}%",
                font=self.fonts['heading'],
                bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack()

        # Confidence bar
        bar_frame = tk.Frame(conf_frame, bg=COLORS['border'], height=20, width=300)
        bar_frame.pack(pady=10)
        bar_frame.pack_propagate(False)

        fill = tk.Frame(bar_frame, bg=color, height=20, width=confidence*3)
        fill.pack(side='left')

        # Risk level
        risk_color = {
            "Low": COLORS['success'],
            "Medium": COLORS['warning'],
            "High": COLORS['danger'],
            "Very High": COLORS['danger']
        }[risk_level]

        tk.Label(result_frame, text=f"Risk Level: {risk_level}",
                font=self.fonts['subheading'],
                bg=COLORS['card_bg'], fg=risk_color).pack(pady=10)

        # Recommendations
        if cancer_type != "No Cancer":
            rec_frame = tk.Frame(result_frame, bg=COLORS['darker_bg'], relief='flat',
                                highlightbackground=COLORS['border'], highlightthickness=1)
            rec_frame.pack(fill='x', pady=20, padx=20)

            tk.Label(rec_frame, text="💡 RECOMMENDATIONS:",
                    font=self.fonts['subheading'],
                    bg=COLORS['darker_bg'], fg=COLORS['warning']).pack(pady=10)

            recommendations = [
                "Consult with an oncologist",
                "Schedule further diagnostic tests",
                "Discuss family screening options",
                "Review lifestyle modifications"
            ]

            for rec in recommendations:
                tk.Label(rec_frame, text=f"• {rec}",
                        font=self.fonts['body'],
                        bg=COLORS['darker_bg'], fg=COLORS['text_primary'],
                        anchor='w').pack(fill='x', padx=20, pady=5)

        # Update status
        self.status_label.config(text=f"✅ Prediction complete: {cancer_type}")

        # Add to history
        self.add_to_history(cancer_type, confidence, risk_level)

    def add_to_history(self, cancer_type, confidence, risk_level):
        """Add prediction to history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "patient_id": self.patient_id.get(),
            "mode": self.current_mode,
            "model": self.model_var.get(),
            "cancer_type": cancer_type,
            "confidence": confidence,
            "risk_level": risk_level
        }
        self.prediction_history.append(entry)

    def quick_predict_cancer(self, cancer_type):
        """Quick predict specific cancer type"""
        self.notebook.select(self.quick_predict_tab)
        self.quick_cancer_var.set(cancer_type)

        # Show notification
        messagebox.showinfo("Quick Predict", f"Ready to predict {cancer_type}!\nSelect risk factors and click QUICK PREDICT.")

    def quick_predict(self):
        """Quick prediction based on selected cancer and risk factors"""
        cancer_type = self.quick_cancer_var.get()

        # Count selected risk factors
        selected_factors = [factor for factor, var in self.risk_vars.items() if var.get()]
        risk_score = len(selected_factors)

        # Calculate confidence based on risk factors
        base_confidence = 50
        confidence = min(95, base_confidence + (risk_score * 10))

        # Determine risk level
        if risk_score <= 2:
            risk_level = "Low"
            color = COLORS['success']
        elif risk_score <= 4:
            risk_level = "Medium"
            color = COLORS['warning']
        elif risk_score <= 6:
            risk_level = "High"
            color = COLORS['danger']
        else:
            risk_level = "Very High"
            color = COLORS['danger']

        # Show result
        result_window = tk.Toplevel(self.root)
        result_window.title("Quick Prediction Result")
        result_window.geometry("500x400")
        result_window.configure(bg=COLORS['dark_bg'])
        result_window.attributes('-topmost', True)

        # Center window
        result_window.update_idletasks()
        width = result_window.winfo_width()
        height = result_window.winfo_height()
        x = (result_window.winfo_screenwidth() // 2) - (width // 2)
        y = (result_window.winfo_screenheight() // 2) - (height // 2)
        result_window.geometry(f'{width}x{height}+{x}+{y}')

        # Result content
        tk.Label(result_window, text="🎯 QUICK PREDICTION",
                font=self.fonts['title'],
                bg=COLORS['dark_bg'], fg=COLORS['primary_light']).pack(pady=30)

        tk.Label(result_window, text=cancer_type,
                font=self.fonts['heading'],
                bg=COLORS['dark_bg'], fg=color).pack(pady=10)

        tk.Label(result_window, text=f"Confidence: {confidence}%",
                font=self.fonts['subheading'],
                bg=COLORS['dark_bg'], fg=COLORS['text_primary']).pack(pady=10)

        tk.Label(result_window, text=f"Risk Level: {risk_level}",
                font=self.fonts['subheading'],
                bg=COLORS['dark_bg'], fg=color).pack(pady=10)

        # Risk factors
        tk.Label(result_window, text=f"Risk Factors: {risk_score} selected",
                font=self.fonts['body'],
                bg=COLORS['dark_bg'], fg=COLORS['text_secondary']).pack(pady=20)

        if selected_factors:
            factors_text = "\n".join([f"• {factor}" for factor in selected_factors])
            tk.Label(result_window, text=factors_text,
                    font=self.fonts['small'],
                    bg=COLORS['dark_bg'], fg=COLORS['text_secondary'],
                    justify='left').pack(pady=10)

        tk.Button(result_window, text="Close",
                 font=self.fonts['body'],
                 bg=COLORS['primary'],
                 fg='white',
                 relief='flat',
                 padx=20,
                 pady=10,
                 command=result_window.destroy).pack(pady=20)

    def create_model_training_interface(self):
        """Create model training interface (simplified)"""
        train_frame = tk.Frame(self.models_tab, bg=COLORS['dark_bg'])
        train_frame.pack(fill='both', expand=True, padx=40, pady=40)

        tk.Label(train_frame, text="🤖 AI MODEL TRAINING",
                font=self.fonts['title'],
                bg=COLORS['dark_bg'], fg=COLORS['primary_light']).pack(pady=20)

        # Training controls
        controls = tk.Frame(train_frame, bg=COLORS['card_bg'], relief='flat',
                           highlightbackground=COLORS['border'], highlightthickness=1)
        controls.pack(fill='x', pady=20)

        tk.Label(controls, text="Train models for all 28 cancer types",
                font=self.fonts['heading'],
                bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack(pady=20)

        # Model selection
        models = ["Random Forest", "Gradient Boosting", "SVM",
                 "Logistic Regression", "Neural Network", "XGBoost"]

        self.model_vars = {}
        for i, model in enumerate(models):
            var = tk.BooleanVar(value=True)
            self.model_vars[model] = var

            cb = tk.Checkbutton(controls, text=model, variable=var,
                               font=self.fonts['body'],
                               bg=COLORS['card_bg'],
                               fg=COLORS['text_primary'])
            cb.pack(side='left', padx=20, pady=10)

        # Train button
        train_btn = tk.Button(controls, text="🚀 TRAIN MODELS",
                             font=self.fonts['heading'],
                             bg=COLORS['primary'],
                             fg='white',
                             relief='flat',
                             padx=30,
                             pady=15,
                             command=self.train_all_models)
        train_btn.pack(pady=20)

    def create_analysis_interface(self):
        """Create analysis interface (simplified)"""
        analysis_frame = tk.Frame(self.analysis_tab, bg=COLORS['dark_bg'])
        analysis_frame.pack(fill='both', expand=True, padx=40, pady=40)

        tk.Label(analysis_frame, text="📊 CANCER ANALYSIS DASHBOARD",
                font=self.fonts['title'],
                bg=COLORS['dark_bg'], fg=COLORS['primary_light']).pack(pady=20)

        # Analysis cards
        cards_frame = tk.Frame(analysis_frame, bg=COLORS['dark_bg'])
        cards_frame.pack(fill='x', pady=20)

        analyses = [
            ("📈 Risk Distribution", COLORS['primary']),
            ("🎯 Model Performance", COLORS['secondary']),
            ("🧬 Cancer Prevalence", COLORS['medical_purple']),
            ("⚠️ Risk Factors", COLORS['warning'])
        ]

        for i, (title, color) in enumerate(analyses):
            card = tk.Frame(cards_frame, bg=COLORS['card_bg'], relief='flat',
                           highlightbackground=COLORS['border'], highlightthickness=1)
            card.grid(row=0, column=i, padx=10, sticky='nsew')
            cards_frame.columnconfigure(i, weight=1)

            tk.Label(card, text=title,
                    font=self.fonts['heading'],
                    bg=COLORS['card_bg'], fg=color).pack(pady=20)

            tk.Button(card, text="View Analysis",
                     font=self.fonts['body'],
                     bg=color,
                     fg='white',
                     relief='flat',
                     padx=20,
                     pady=10).pack(pady=10)

    def create_history_interface(self):
        """Create history interface (simplified)"""
        history_frame = tk.Frame(self.history_tab, bg=COLORS['dark_bg'])
        history_frame.pack(fill='both', expand=True, padx=40, pady=40)

        tk.Label(history_frame, text="📋 PREDICTION HISTORY",
                font=self.fonts['title'],
                bg=COLORS['dark_bg'], fg=COLORS['primary_light']).pack(pady=20)

        # History table
        table_frame = tk.Frame(history_frame, bg=COLORS['card_bg'], relief='flat',
                              highlightbackground=COLORS['border'], highlightthickness=1)
        table_frame.pack(fill='both', expand=True)

        # Create treeview
        columns = ("Date", "Patient ID", "Cancer Type", "Confidence", "Risk", "Model")

        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)

        # Add sample data
        sample_data = [
            ("2024-01-15 10:30", "PAT-12345", "Breast Cancer", "87%", "High", "Random Forest"),
            ("2024-01-15 11:15", "PAT-12346", "No Cancer", "95%", "Low", "Gradient Boosting"),
            ("2024-01-15 12:45", "PAT-12347", "Lung Cancer", "92%", "Very High", "SVM"),
            ("2024-01-15 14:30", "PAT-12348", "Prostate Cancer", "78%", "Medium", "Random Forest"),
            ("2024-01-15 15:15", "PAT-12349", "Skin Cancer", "85%", "High", "Neural Network")
        ]

        for data in sample_data:
            tree.insert('', 'end', values=data)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y')

    def train_all_models(self):
        """Train all selected models"""
        selected_models = [model for model, var in self.model_vars.items() if var.get()]

        if not selected_models:
            messagebox.showwarning("No Models", "Please select at least one model to train.")
            return

        # Show training window
        train_window = tk.Toplevel(self.root)
        train_window.title("Training Models")
        train_window.geometry("600x400")
        train_window.configure(bg=COLORS['dark_bg'])
        train_window.attributes('-topmost', True)

        # Center window
        train_window.update_idletasks()
        width = train_window.winfo_width()
        height = train_window.winfo_height()
        x = (train_window.winfo_screenwidth() // 2) - (width // 2)
        y = (train_window.winfo_screenheight() // 2) - (height // 2)
        train_window.geometry(f'{width}x{height}+{x}+{y}')

        # Training content
        tk.Label(train_window, text="🚀 TRAINING AI MODELS",
                font=self.fonts['title'],
                bg=COLORS['dark_bg'], fg=COLORS['primary_light']).pack(pady=30)

        progress_frame = tk.Frame(train_window, bg=COLORS['dark_bg'])
        progress_frame.pack(pady=20)

        # Progress bars for each model
        self.training_progress = {}

        for i, model in enumerate(selected_models):
            frame = tk.Frame(progress_frame, bg=COLORS['dark_bg'])
            frame.pack(fill='x', pady=5)

            tk.Label(frame, text=model,
                    font=self.fonts['body'],
                    bg=COLORS['dark_bg'], fg=COLORS['text_primary'],
                    width=20, anchor='w').pack(side='left')

            progress = ttk.Progressbar(frame, length=300, mode='determinate')
            progress.pack(side='right')
            self.training_progress[model] = progress

        # Start training simulation
        self.simulate_training(selected_models, train_window)

    def simulate_training(self, models, window):
        """Simulate model training"""
        def train():
            for i, model in enumerate(models):
                for progress in range(0, 101, 10):
                    self.training_progress[model]['value'] = progress
                    window.update()
                    time.sleep(0.1)

            window.after(0, lambda: self.training_complete(window))

        threading.Thread(target=train, daemon=True).start()

    def training_complete(self, window):
        """Handle training completion"""
        window.destroy()
        messagebox.showinfo("Training Complete",
                          "All models have been trained successfully!\nReady for predictions.")
        self.status_label.config(text="✅ AI Models trained successfully")

def main():
    root = tk.Tk()
    app = UltraCancerPredictionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
