"""
Tkinter GUI for Signature Forgery Detection
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys
import threading
import platform
import numpy as np
import tensorflow as tf
from tensorflow import keras
# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils import load_image
from models import create_cnn_model, create_siamese_network

# macOS specific settings
IS_MAC = platform.system() == 'Darwin'


class SignatureForgeryDetectionApp:
    def __init__(self, root):
        self.root = root
        
        # macOS specific configurations
        if IS_MAC:
            # Set macOS-specific window behavior
            try:
                # Make window appear in foreground
                root.lift()
                root.attributes('-topmost', True)
                root.after_idle(root.attributes, '-topmost', False)
                
                # Set window to appear in dock
                root.createcommand('tk::mac::ReopenApplication', lambda: root.deiconify())
                
                # Configure for macOS appearance
                try:
                    # Use native macOS fonts
                    self.default_font = ('SF Pro Display', 12) if self._font_exists('SF Pro Display') else ('Helvetica', 12)
                    self.title_font = ('SF Pro Display', 20, 'bold') if self._font_exists('SF Pro Display') else ('Helvetica', 20, 'bold')
                except:
                    self.default_font = ('Helvetica', 12)
                    self.title_font = ('Helvetica', 20, 'bold')
            except Exception as e:
                print(f"macOS configuration warning: {e}")
                self.default_font = ('Helvetica', 12)
                self.title_font = ('Helvetica', 20, 'bold')
        else:
            self.default_font = ('Arial', 10)
            self.title_font = ('Arial', 20, 'bold')
        
        self.root.title("Signature Forgery Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Center window on screen
        self._center_window()
        
        # Model variables
        self.cnn_model = None
        self.siamese_model = None
        self.model_type = tk.StringVar(value="CNN")
        self.selected_image_path = None
        self.selected_image1_path = None
        self.selected_image2_path = None
        
        # Training variables
        self.genuine_folder_path = None
        self.forged_folder_path = None
        self.training_in_progress = False
        
        # Batch detection variables
        self.cnn_batch_folder_path = None
        self.batch_detection_in_progress = False
        
        # Create GUI
        self.create_widgets()
        
        # Load models if they exist
        self.load_models()
    
    def _font_exists(self, font_name):
        """Check if a font exists on the system"""
        try:
            from tkinter import font
            available_fonts = font.families()
            return font_name in available_fonts
        except:
            return False
    
    def _center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="Signature Forgery Detection System",
            font=self.title_font,
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=15)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Detection tab
        detection_frame = tk.Frame(self.notebook, bg='#f0f0f0')
        self.notebook.add(detection_frame, text="🔍 Detection")
        
        # Training tab
        training_frame = tk.Frame(self.notebook, bg='#f0f0f0')
        self.notebook.add(training_frame, text="🎓 Training")
        
        # Create detection widgets
        self.create_detection_widgets(detection_frame)
        
        # Create training widgets
        self.create_training_widgets(training_frame)
    
    def create_detection_widgets(self, parent):
        """Create detection tab widgets"""
        # Main container
        main_container = tk.Frame(parent, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Model selection and controls
        left_panel = tk.Frame(main_container, bg='#ecf0f1', width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Model selection
        model_frame = tk.LabelFrame(
            left_panel,
            text="Model Selection",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1',
            padx=10,
            pady=10
        )
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        cnn_radio = tk.Radiobutton(
            model_frame,
            text="CNN Model",
            variable=self.model_type,
            value="CNN",
            font=('Arial', 10),
            bg='#ecf0f1',
            command=self.on_model_change
        )
        cnn_radio.pack(anchor=tk.W, pady=5)
        
        siamese_radio = tk.Radiobutton(
            model_frame,
            text="Siamese Network",
            variable=self.model_type,
            value="Siamese",
            font=('Arial', 10),
            bg='#ecf0f1',
            command=self.on_model_change
        )
        siamese_radio.pack(anchor=tk.W, pady=5)
        
        # Model status
        self.model_status_label = tk.Label(
            model_frame,
            text="Status: Not Loaded",
            font=('Arial', 9),
            bg='#ecf0f1',
            fg='red'
        )
        self.model_status_label.pack(pady=5)
        
        # Load model button
        load_model_btn = tk.Button(
            model_frame,
            text="Load Model",
            command=self.load_model_dialog,
            bg='#3498db',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20,
            pady=5,
            cursor='hand2'
        )
        load_model_btn.pack(pady=10)
        
        # Image selection frame
        image_frame = tk.LabelFrame(
            left_panel,
            text="Image Selection",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1',
            padx=10,
            pady=10
        )
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # For CNN - single image
        self.cnn_frame = tk.Frame(image_frame, bg='#ecf0f1')
        self.cnn_frame.pack(fill=tk.BOTH, expand=True)
        
        cnn_select_btn = tk.Button(
            self.cnn_frame,
            text="Select Signature Image",
            command=self.select_image_cnn,
            bg='#27ae60',
            fg='white',
            font=('Arial', 10),
            padx=15,
            pady=8,
            cursor='hand2'
        )
        cnn_select_btn.pack(pady=10)
        
        self.cnn_image_label = tk.Label(
            self.cnn_frame,
            text="No image selected",
            font=('Arial', 9),
            bg='#ecf0f1',
            fg='gray'
        )
        self.cnn_image_label.pack(pady=5)
        
        # Batch detection button
        cnn_batch_btn = tk.Button(
            self.cnn_frame,
            text="📁 Select Folder (Batch Detection)",
            command=self.select_folder_batch_cnn,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 9),
            padx=10,
            pady=5,
            cursor='hand2'
        )
        cnn_batch_btn.pack(pady=5)
        
        self.cnn_batch_folder_label = tk.Label(
            self.cnn_frame,
            text="",
            font=('Arial', 8),
            bg='#ecf0f1',
            fg='gray'
        )
        self.cnn_batch_folder_label.pack(pady=2)
        
        # For Siamese - two images
        self.siamese_frame = tk.Frame(image_frame, bg='#ecf0f1')
        
        siamese_select1_btn = tk.Button(
            self.siamese_frame,
            text="Select First Signature",
            command=self.select_image1_siamese,
            bg='#27ae60',
            fg='white',
            font=('Arial', 10),
            padx=15,
            pady=8,
            cursor='hand2'
        )
        siamese_select1_btn.pack(pady=5)
        
        self.siamese_image1_label = tk.Label(
            self.siamese_frame,
            text="No image selected",
            font=('Arial', 9),
            bg='#ecf0f1',
            fg='gray'
        )
        self.siamese_image1_label.pack(pady=2)
        
        siamese_select2_btn = tk.Button(
            self.siamese_frame,
            text="Select Second Signature",
            command=self.select_image2_siamese,
            bg='#27ae60',
            fg='white',
            font=('Arial', 10),
            padx=15,
            pady=8,
            cursor='hand2'
        )
        siamese_select2_btn.pack(pady=5)
        
        self.siamese_image2_label = tk.Label(
            self.siamese_frame,
            text="No image selected",
            font=('Arial', 9),
            bg='#ecf0f1',
            fg='gray'
        )
        self.siamese_image2_label.pack(pady=2)
        
        # Predict button
        predict_btn = tk.Button(
            left_panel,
            text="🔍 Detect Forgery",
            command=self.predict,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=30,
            pady=12,
            cursor='hand2'
        )
        predict_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))
        
        # Batch predict button
        batch_predict_btn = tk.Button(
            left_panel,
            text="📊 Batch Detect (Folder)",
            command=self.predict_batch,
            bg='#8e44ad',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20,
            pady=8,
            cursor='hand2'
        )
        batch_predict_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
        
        # Right panel - Image display and results
        right_panel = tk.Frame(main_container, bg='white', width=650)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image display
        image_display_frame = tk.LabelFrame(
            right_panel,
            text="Image Preview",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        image_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.image_display_label = tk.Label(
            image_display_frame,
            text="No image to display",
            bg='white',
            fg='gray',
            font=('Arial', 11)
        )
        self.image_display_label.pack(expand=True)
        
        # Results frame
        results_frame = tk.LabelFrame(
            right_panel,
            text="Detection Results",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        results_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.results_text = tk.Text(
            results_frame,
            height=8,
            font=('Arial', 10),
            bg='#f8f9fa',
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize with CNN frame visible
        self.on_model_change()
    
    def create_training_widgets(self, parent):
        """Create training tab widgets"""
        # Main container
        main_container = tk.Frame(parent, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Dataset selection
        left_panel = tk.Frame(main_container, bg='#ecf0f1', width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Dataset selection frame
        dataset_frame = tk.LabelFrame(
            left_panel,
            text="Dataset Selection",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1',
            padx=10,
            pady=10
        )
        dataset_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Genuine folder selection
        genuine_btn = tk.Button(
            dataset_frame,
            text="📁 Select Genuine Signatures Folder",
            command=self.select_genuine_folder,
            bg='#27ae60',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=8,
            cursor='hand2'
        )
        genuine_btn.pack(fill=tk.X, pady=5)
        
        self.genuine_folder_label = tk.Label(
            dataset_frame,
            text="No folder selected",
            font=('Arial', 9),
            bg='#ecf0f1',
            fg='gray',
            wraplength=350
        )
        self.genuine_folder_label.pack(pady=5)
        
        # Forged folder selection
        forged_btn = tk.Button(
            dataset_frame,
            text="📁 Select Forged Signatures Folder",
            command=self.select_forged_folder,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=8,
            cursor='hand2'
        )
        forged_btn.pack(fill=tk.X, pady=5)
        
        self.forged_folder_label = tk.Label(
            dataset_frame,
            text="No folder selected",
            font=('Arial', 9),
            bg='#ecf0f1',
            fg='gray',
            wraplength=350
        )
        self.forged_folder_label.pack(pady=5)
        
        # Training parameters frame
        params_frame = tk.LabelFrame(
            left_panel,
            text="Training Parameters",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1',
            padx=10,
            pady=10
        )
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Model type selection
        tk.Label(
            params_frame,
            text="Model Type:",
            font=('Arial', 10),
            bg='#ecf0f1'
        ).pack(anchor=tk.W, pady=2)
        
        self.training_model_type = tk.StringVar(value="CNN")
        cnn_train_radio = tk.Radiobutton(
            params_frame,
            text="CNN Model",
            variable=self.training_model_type,
            value="CNN",
            font=('Arial', 9),
            bg='#ecf0f1'
        )
        cnn_train_radio.pack(anchor=tk.W, pady=2)
        
        siamese_train_radio = tk.Radiobutton(
            params_frame,
            text="Siamese Network",
            variable=self.training_model_type,
            value="Siamese",
            font=('Arial', 9),
            bg='#ecf0f1'
        )
        siamese_train_radio.pack(anchor=tk.W, pady=2)
        
        # Epochs
        epochs_frame = tk.Frame(params_frame, bg='#ecf0f1')
        epochs_frame.pack(fill=tk.X, pady=5)
        tk.Label(
            epochs_frame,
            text="Epochs:",
            font=('Arial', 10),
            bg='#ecf0f1',
            width=12,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        self.epochs_var = tk.StringVar(value="50")
        epochs_entry = tk.Entry(epochs_frame, textvariable=self.epochs_var, width=10)
        epochs_entry.pack(side=tk.LEFT, padx=5)
        
        # Batch size
        batch_frame = tk.Frame(params_frame, bg='#ecf0f1')
        batch_frame.pack(fill=tk.X, pady=5)
        tk.Label(
            batch_frame,
            text="Batch Size:",
            font=('Arial', 10),
            bg='#ecf0f1',
            width=12,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        self.batch_size_var = tk.StringVar(value="128")
        batch_entry = tk.Entry(batch_frame, textvariable=self.batch_size_var, width=10)
        batch_entry.pack(side=tk.LEFT, padx=5)
        
        # Patience (Early Stopping)
        patience_frame = tk.Frame(params_frame, bg='#ecf0f1')
        patience_frame.pack(fill=tk.X, pady=5)
        tk.Label(
            patience_frame,
            text="Patience:",
            font=('Arial', 10),
            bg='#ecf0f1',
            width=12,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        self.patience_var = tk.StringVar(value="7")
        patience_entry = tk.Entry(patience_frame, textvariable=self.patience_var, width=10)
        patience_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(
            patience_frame,
            text="(Early stopping patience)",
            font=('Arial', 8),
            bg='#ecf0f1',
            fg='#7f8c8d'
        ).pack(side=tk.LEFT, padx=5)
        
        # Start training button
        train_btn = tk.Button(
            left_panel,
            text="🚀 Start Training",
            command=self.start_training,
            bg='#3498db',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=30,
            pady=12,
            cursor='hand2'
        )
        train_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Right panel - Training progress and logs
        right_panel = tk.Frame(main_container, bg='white', width=600)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Progress frame
        progress_frame = tk.LabelFrame(
            right_panel,
            text="Training Progress",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        progress_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=500
        )
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.progress_label = tk.Label(
            progress_frame,
            text="Ready to train",
            font=('Arial', 10),
            bg='white',
            fg='gray'
        )
        self.progress_label.pack(pady=5)
        
        # Logs frame
        logs_frame = tk.LabelFrame(
            right_panel,
            text="Training Logs",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbar for logs
        log_scrollbar = tk.Scrollbar(logs_frame)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.training_logs = tk.Text(
            logs_frame,
            height=20,
            font=('Consolas', 9),
            bg='#2c3e50',
            fg='#ecf0f1',
            wrap=tk.WORD,
            yscrollcommand=log_scrollbar.set,
            state=tk.DISABLED
        )
        self.training_logs.pack(fill=tk.BOTH, expand=True)
        log_scrollbar.config(command=self.training_logs.yview)
    
    def on_model_change(self):
        """Handle model type change"""
        if self.model_type.get() == "CNN":
            self.cnn_frame.pack(fill=tk.BOTH, expand=True)
            self.siamese_frame.pack_forget()
        else:
            self.cnn_frame.pack_forget()
            self.siamese_frame.pack(fill=tk.BOTH, expand=True)
    
    def load_models(self):
        """Try to load models from default paths"""
        cnn_path = "models/cnn_model.h5"
        siamese_path = "models/siamese_model.h5"
        
        if os.path.exists(cnn_path):
            try:
                self.cnn_model = keras.models.load_model(cnn_path)
                if self.model_type.get() == "CNN":
                    self.model_status_label.config(text="Status: CNN Loaded", fg='green')
            except:
                pass
        
        if os.path.exists(siamese_path):
            try:
                self.siamese_model = keras.models.load_model(siamese_path)
                if self.model_type.get() == "Siamese":
                    self.model_status_label.config(text="Status: Siamese Loaded", fg='green')
            except:
                pass
    
    def load_model_dialog(self):
        """Open dialog to load model"""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if model_path:
            try:
                if self.model_type.get() == "CNN":
                    self.cnn_model = keras.models.load_model(model_path)
                    self.model_status_label.config(text="Status: CNN Loaded", fg='green')
                    messagebox.showinfo("Success", "CNN model loaded successfully!")
                else:
                    self.siamese_model = keras.models.load_model(model_path)
                    self.model_status_label.config(text="Status: Siamese Loaded", fg='green')
                    messagebox.showinfo("Success", "Siamese model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def select_image_cnn(self):
        """Select image for CNN model"""
        image_path = filedialog.askopenfilename(
            title="Select Signature Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        
        if image_path:
            self.selected_image_path = image_path
            self.display_image(image_path)
            self.cnn_image_label.config(text=os.path.basename(image_path))
            # Clear batch folder selection when single image is selected
            self.cnn_batch_folder_path = None
            self.cnn_batch_folder_label.config(text="")
    
    def select_folder_batch_cnn(self):
        """Select folder for batch CNN detection"""
        folder_path = filedialog.askdirectory(
            title="Select Folder with Signature Images"
        )
        
        if folder_path:
            self.cnn_batch_folder_path = folder_path
            # Count images
            image_count = len([f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.cnn_batch_folder_label.config(
                text=f"✓ {image_count} images selected",
                fg='green'
            )
            # Clear single image selection when batch folder is selected
            self.selected_image_path = None
            self.cnn_image_label.config(text="No image selected")
    
    def select_image1_siamese(self):
        """Select first image for Siamese model"""
        image_path = filedialog.askopenfilename(
            title="Select First Signature",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        
        if image_path:
            self.selected_image1_path = image_path
            self.display_image(image_path)
            self.siamese_image1_label.config(text=os.path.basename(image_path))
    
    def select_image2_siamese(self):
        """Select second image for Siamese model"""
        image_path = filedialog.askopenfilename(
            title="Select Second Signature",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        
        if image_path:
            self.selected_image2_path = image_path
            self.display_image(image_path)
            self.siamese_image2_label.config(text=os.path.basename(image_path))
    
    def display_image(self, image_path):
        """Display image in the preview area"""
        try:
            img = Image.open(image_path)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_display_label.config(image=photo, text="")
            self.image_display_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def predict(self):
        """Perform prediction"""
        model_type = self.model_type.get()
        
        if model_type == "CNN":
            if self.cnn_model is None:
                messagebox.showerror("Error", "Please load a CNN model first!")
                return
            
            if self.selected_image_path is None:
                messagebox.showerror("Error", "Please select an image first!")
                return
            
            # Predict in a separate thread to avoid freezing GUI
            threading.Thread(target=self.predict_cnn, daemon=True).start()
        
        else:  # Siamese
            if self.siamese_model is None:
                messagebox.showerror("Error", "Please load a Siamese model first!")
                return
            
            if self.selected_image1_path is None or self.selected_image2_path is None:
                messagebox.showerror("Error", "Please select both images first!")
                return
            
            # Predict in a separate thread
            threading.Thread(target=self.predict_siamese, daemon=True).start()
    
    def predict_batch(self):
        """Perform batch prediction on folder"""
        if self.model_type.get() != "CNN":
            messagebox.showwarning("Warning", "Batch detection is only available for CNN model!")
            return
        
        if self.cnn_model is None:
            messagebox.showerror("Error", "Please load a CNN model first!")
            return
        
        if self.cnn_batch_folder_path is None:
            messagebox.showerror("Error", "Please select a folder with images first!")
            return
        
        if self.batch_detection_in_progress:
            messagebox.showwarning("Warning", "Batch detection is already in progress!")
            return
        
        # Start batch detection in separate thread
        self.batch_detection_in_progress = True
        threading.Thread(target=self.predict_batch_cnn, daemon=True).start()
    
    def predict_batch_cnn(self):
        """Perform batch prediction on folder"""
        try:
            from evaluate import predict_signature_cnn
            import glob
            
            # Get all image files from folder
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(self.cnn_batch_folder_path, ext)))
            
            if len(image_files) == 0:
                raise Exception("No image files found in selected folder")
            
            total_images = len(image_files)
            genuine_count = 0
            forged_count = 0
            results_list = []
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Batch Detection Started\n")
            self.results_text.insert(tk.END, f"Processing {total_images} images...\n\n")
            self.results_text.config(state=tk.DISABLED)
            self.root.update()
            
            # Process each image
            for idx, image_path in enumerate(image_files, 1):
                try:
                    is_genuine, confidence = predict_signature_cnn(
                        self.cnn_model,
                        image_path
                    )
                    
                    if is_genuine == 1:
                        genuine_count += 1
                        status = "GENUINE"
                    else:
                        forged_count += 1
                        status = "FORGED"
                    
                    results_list.append({
                        'file': os.path.basename(image_path),
                        'status': status,
                        'confidence': confidence
                    })
                    
                    # Update progress
                    progress_text = f"Processed: {idx}/{total_images}\n"
                    progress_text += f"Genuine: {genuine_count}, Forged: {forged_count}\n"
                    progress_text += f"Current: {os.path.basename(image_path)} - {status}\n\n"
                    
                    self.results_text.config(state=tk.NORMAL)
                    self.results_text.insert(tk.END, progress_text)
                    self.results_text.see(tk.END)
                    self.results_text.config(state=tk.DISABLED)
                    self.root.update()
                    
                except Exception as e:
                    self.results_text.config(state=tk.NORMAL)
                    self.results_text.insert(tk.END, f"Error processing {os.path.basename(image_path)}: {str(e)}\n")
                    self.results_text.config(state=tk.DISABLED)
                    self.root.update()
            
            # Final summary
            result_text = f"\n{'='*60}\n"
            result_text += f"BATCH DETECTION SUMMARY\n"
            result_text += f"{'='*60}\n\n"
            result_text += f"Total Images: {total_images}\n"
            result_text += f"Genuine: {genuine_count} ({genuine_count/total_images*100:.1f}%)\n"
            result_text += f"Forged: {forged_count} ({forged_count/total_images*100:.1f}%)\n\n"
            result_text += f"{'='*60}\n"
            result_text += f"DETAILED RESULTS:\n"
            result_text += f"{'='*60}\n\n"
            
            # Sort by confidence (highest first)
            results_list.sort(key=lambda x: x['confidence'], reverse=True)
            
            for result in results_list:
                result_text += f"{result['file']:30s} | {result['status']:8s} | {result['confidence']*100:5.2f}%\n"
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.insert(tk.END, result_text)
            self.results_text.see(tk.END)
            self.results_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error: {str(e)}")
            self.results_text.config(state=tk.DISABLED)
            messagebox.showerror("Error", f"Batch detection failed:\n{str(e)}")
        finally:
            self.batch_detection_in_progress = False
    
    def predict_cnn(self):
        """Predict using CNN model"""
        try:
            # Import here to avoid circular imports
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            from evaluate import predict_signature_cnn
            
            # Get current image path (may have changed)
            current_image_path = self.selected_image_path
            
            if current_image_path is None:
                raise Exception("No image selected")
            
            if not os.path.exists(current_image_path):
                raise Exception(f"Image file not found: {current_image_path}")
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Analyzing: {os.path.basename(current_image_path)}...\n")
            self.results_text.config(state=tk.DISABLED)
            self.root.update()
            
            # Force new prediction (clear any potential cache)
            is_genuine, confidence = predict_signature_cnn(
                self.cnn_model,
                current_image_path
            )
            
            if is_genuine is None:
                raise Exception("Failed to process image")
            
            result_text = f"CNN Model Prediction:\n"
            result_text += f"{'='*40}\n\n"
            result_text += f"Image: {os.path.basename(current_image_path)}\n\n"
            
            if is_genuine == 1:
                result_text += f"Result: ✓ GENUINE SIGNATURE\n"
                result_text += f"Confidence: {confidence*100:.2f}%\n"
            else:
                result_text += f"Result: ✗ FORGED SIGNATURE\n"
                result_text += f"Confidence: {confidence*100:.2f}%\n"
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result_text)
            self.results_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error: {str(e)}")
            self.results_text.config(state=tk.DISABLED)
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
    
    def predict_siamese(self):
        """Predict using Siamese model"""
        try:
            # Import here to avoid circular imports
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            from evaluate import predict_signature_siamese
            
            # Get current image paths (may have changed)
            current_image1_path = self.selected_image1_path
            current_image2_path = self.selected_image2_path
            
            if current_image1_path is None or current_image2_path is None:
                raise Exception("Both images must be selected")
            
            if not os.path.exists(current_image1_path):
                raise Exception(f"First image file not found: {current_image1_path}")
            if not os.path.exists(current_image2_path):
                raise Exception(f"Second image file not found: {current_image2_path}")
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Comparing:\n{os.path.basename(current_image1_path)}\nvs\n{os.path.basename(current_image2_path)}...\n")
            self.results_text.config(state=tk.DISABLED)
            self.root.update()
            
            # Force new prediction (clear any potential cache)
            is_match, confidence = predict_signature_siamese(
                self.siamese_model,
                current_image1_path,
                current_image2_path
            )
            
            if is_match is None:
                raise Exception("Failed to process images")
            
            result_text = f"Siamese Network Prediction:\n"
            result_text += f"{'='*40}\n\n"
            result_text += f"Image 1: {os.path.basename(current_image1_path)}\n"
            result_text += f"Image 2: {os.path.basename(current_image2_path)}\n\n"
            
            if is_match == 1:
                result_text += f"Result: ✓ SIGNATURES MATCH\n"
                result_text += f"Confidence: {confidence*100:.2f}%\n"
                result_text += f"\nBoth signatures appear to be from the same person."
            else:
                result_text += f"Result: ✗ SIGNATURES DO NOT MATCH\n"
                result_text += f"Confidence: {confidence*100:.2f}%\n"
                result_text += f"\nThe signatures appear to be from different people or one is forged."
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result_text)
            self.results_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error: {str(e)}")
            self.results_text.config(state=tk.DISABLED)
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
    
    def select_genuine_folder(self):
        """Select folder containing genuine signatures"""
        folder_path = filedialog.askdirectory(
            title="Select Genuine Signatures Folder"
        )
        
        if folder_path:
            self.genuine_folder_path = folder_path
            # Count images
            image_count = len([f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.genuine_folder_label.config(
                text=f"✓ {image_count} images\n{os.path.basename(folder_path)}",
                fg='green'
            )
    
    def select_forged_folder(self):
        """Select folder containing forged signatures"""
        folder_path = filedialog.askdirectory(
            title="Select Forged Signatures Folder"
        )
        
        if folder_path:
            self.forged_folder_path = folder_path
            # Count images
            image_count = len([f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.forged_folder_label.config(
                text=f"✓ {image_count} images\n{os.path.basename(folder_path)}",
                fg='green'
            )
    
    def log_message(self, message):
        """Add message to training logs - thread-safe"""
        # Always print to console for debugging
        print(f"[LOG] {message}", flush=True)
        
        def _update_log():
            try:
                if hasattr(self, 'training_logs') and self.training_logs:
                    self.training_logs.config(state=tk.NORMAL)
                    self.training_logs.insert(tk.END, message + "\n")
                    self.training_logs.see(tk.END)
                    self.training_logs.config(state=tk.DISABLED)
                    # Force update
                    self.training_logs.update_idletasks()
            except Exception as e:
                # Fallback: print to console if GUI update fails
                print(f"[LOG ERROR] {e}", flush=True)
                print(f"[LOG] {message}", flush=True)
        
        # Always use root.after for thread safety (works on all platforms)
        try:
            if hasattr(self, 'root') and self.root:
                # Use after_idle for more reliable execution
                self.root.after_idle(_update_log)
            else:
                print(f"[LOG] (No root) {message}", flush=True)
        except Exception as e:
            # If root.after fails, print directly
            print(f"[LOG ERROR] root.after failed: {e}", flush=True)
            print(f"[LOG] {message}", flush=True)
    
    def update_progress(self, value, message=""):
        """Update training progress bar"""
        def _update_progress():
            self.progress_var.set(value)
            if message:
                self.progress_label.config(text=message)
        
        # macOS requires GUI updates on main thread
        if IS_MAC:
            self.root.after(0, _update_progress)
        else:
            _update_progress()
            self.root.update()
    
    def start_training(self):
        """Start model training"""
        # Validate inputs
        if not self.genuine_folder_path:
            messagebox.showerror("Error", "Please select genuine signatures folder!")
            return
        
        if not self.forged_folder_path:
            messagebox.showerror("Error", "Please select forged signatures folder!")
            return
        
        if self.training_in_progress:
            messagebox.showwarning("Warning", "Training is already in progress!")
            return
        
        # Get parameters
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            patience = int(self.patience_var.get())
            if patience < 1:
                raise ValueError("Patience must be at least 1")
        except ValueError as e:
            error_msg = "Please enter valid numbers for epochs, batch size, and patience!"
            if "Patience" in str(e):
                error_msg = "Patience must be a positive integer (at least 1)!"
            messagebox.showerror("Error", error_msg)
            return
        
        model_type = self.training_model_type.get()
        
        # Confirm training
        confirm = messagebox.askyesno(
            "Confirm Training",
            f"Start training {model_type} model?\n\n"
            f"Epochs: {epochs}\n"
            f"Batch Size: {batch_size}\n"
            f"Patience: {patience}\n\n"
            f"This may take a long time. Continue?"
        )
        
        if not confirm:
            return
        
        # Start training in separate thread
        self.training_in_progress = True
        self.progress_var.set(0)
        
        # Clear logs and add initial message
        def clear_logs():
            self.training_logs.config(state=tk.NORMAL)
            self.training_logs.delete(1.0, tk.END)
            self.training_logs.insert(tk.END, "=" * 60 + "\n")
            self.training_logs.insert(tk.END, f"Starting {model_type} Training\n")
            self.training_logs.insert(tk.END, "=" * 60 + "\n\n")
            self.training_logs.see(tk.END)
            self.training_logs.config(state=tk.DISABLED)
            self.training_logs.update_idletasks()
        
        if IS_MAC:
            self.root.after_idle(clear_logs)
        else:
            clear_logs()
        
        # Start training thread
        training_thread = threading.Thread(
            target=self.train_model_thread,
            args=(model_type, epochs, batch_size, patience),
            daemon=True
        )
        training_thread.start()
        
        # Log that thread started
        self.log_message(f"Training thread started for {model_type} model")
    
    def train_model_thread(self, model_type, epochs, batch_size, patience):
        """Training thread function"""
        try:
            import shutil
            
            # Initial log to verify logging works
            self.log_message("=" * 60)
            self.log_message(f"Starting {model_type} model training...")
            self.log_message(f"Parameters: Epochs={epochs}, Batch Size={batch_size}, Patience={patience}")
            self.log_message("=" * 60)
            
            # Create temporary data directory structure
            temp_data_dir = "data/temp_training"
            os.makedirs(temp_data_dir, exist_ok=True)
            
            genuine_dest = os.path.join(temp_data_dir, "genuine")
            forged_dest = os.path.join(temp_data_dir, "forged")
            
            # Copy files to temporary directory
            self.log_message("=" * 60)
            self.log_message(f"Preparing dataset for {model_type} training...")
            self.update_progress(5, "Copying genuine signatures...")
            
            if os.path.exists(genuine_dest):
                shutil.rmtree(genuine_dest)
            shutil.copytree(self.genuine_folder_path, genuine_dest)
            
            self.update_progress(10, "Copying forged signatures...")
            if os.path.exists(forged_dest):
                shutil.rmtree(forged_dest)
            shutil.copytree(self.forged_folder_path, forged_dest)
            
            genuine_count = len([f for f in os.listdir(genuine_dest) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            forged_count = len([f for f in os.listdir(forged_dest) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            self.log_message(f"Dataset prepared: {genuine_count} genuine, {forged_count} forged")
            self.update_progress(15, "Dataset ready!")
            
            # Train model
            if model_type == "CNN":
                self.log_message("\n" + "=" * 60)
                self.log_message("Training CNN Model...")
                self.log_message("=" * 60)
                
                model_path = "models/cnn_model.h5"
                os.makedirs("models", exist_ok=True)
                
                # Custom callback for progress
                class ProgressCallback(keras.callbacks.Callback):
                    def __init__(self, gui_app, total_epochs):
                        self.gui_app = gui_app
                        self.total_epochs = total_epochs
                    
                    def on_epoch_begin(self, epoch, logs=None):
                        progress = 15 + (epoch / self.total_epochs) * 80
                        self.gui_app.update_progress(
                            progress,
                            f"Training epoch {epoch + 1}/{self.total_epochs}"
                        )
                        self.gui_app.log_message(f"\nEpoch {epoch + 1}/{self.total_epochs}")
                    
                    def on_epoch_end(self, epoch, logs=None):
                        if logs:
                            self.gui_app.log_message(
                                f"  Loss: {logs.get('loss', 0):.4f}, "
                                f"Accuracy: {logs.get('accuracy', 0):.4f}"
                            )
                            if 'val_loss' in logs:
                                self.gui_app.log_message(
                                    f"  Val Loss: {logs.get('val_loss', 0):.4f}, "
                                    f"Val Accuracy: {logs.get('val_accuracy', 0):.4f}"
                                )
                
                # Import training function and modify to use our callback
                from models import create_cnn_model, compile_model
                from utils import prepare_dataset, split_dataset, evaluate_model, plot_confusion_matrix
                from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
                import numpy as np
                
                # Prepare dataset
                X, y = prepare_dataset(temp_data_dir, (128, 128))
                X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
                    X, y, test_size=0.2, val_size=0.1
                )
                
                # Create model
                model = create_cnn_model(input_shape=(128, 128, 3))
                model = compile_model(model, learning_rate=0.001)
                
                # Callbacks
                progress_callback = ProgressCallback(self, epochs)
                callbacks = [
                    progress_callback,
                    ModelCheckpoint(
                        model_path,
                        monitor='val_accuracy',
                        save_best_only=True,
                        mode='max',
                        verbose=0
                    ),
                    EarlyStopping(
                        monitor='val_accuracy',
                        patience=max(5, patience // 2),  # More aggressive early stopping
                        restore_best_weights=True,
                        verbose=0,
                        min_delta=0.001  # Minimum improvement required
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=3,  # Reduced from 5 to 3 for faster adaptation
                        min_lr=1e-7,
                        verbose=0
                    )
                ]
                
                # Data augmentation
                datagen = keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=False,
                    fill_mode='nearest'
                )
                
                # Train
                history = model.fit(
                    datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Evaluate
                self.update_progress(95, "Evaluating model...")
                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
                metrics = evaluate_model(y_test, y_pred, "CNN Model")
                
                self.log_message("\n" + "=" * 60)
                self.log_message("Training Complete!")
                self.log_message(f"Test Accuracy: {test_accuracy:.4f}")
                self.log_message(f"Precision: {metrics['precision']:.4f}")
                self.log_message(f"Recall: {metrics['recall']:.4f}")
                self.log_message(f"F1-Score: {metrics['f1_score']:.4f}")
                self.log_message("=" * 60)
                
            else:  # Siamese
                self.log_message("\n" + "=" * 60)
                self.log_message("Training Siamese Network...")
                self.log_message("=" * 60)
                
                model_path = "models/siamese_model.h5"
                embedding_path = "models/siamese_embedding.h5"
                os.makedirs("models", exist_ok=True)
                self.log_message(f"Model will be saved to: {model_path}")
                self.log_message(f"Embedding network will be saved to: {embedding_path}")
                
                # Import required modules
                try:
                    self.log_message("Importing required modules...")
                    from models import (
                        create_siamese_network, 
                        compile_model, 
                        create_data_generator_for_siamese,
                        prepare_siamese_pairs
                    )
                    from utils import prepare_dataset, split_dataset, evaluate_model, plot_confusion_matrix
                    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
                    import numpy as np
                    self.log_message("Modules imported successfully")
                except ImportError as e:
                    error_msg = f"Import error: {str(e)}"
                    self.log_message(f"ERROR: {error_msg}")
                    raise ImportError(error_msg)
                
                # Prepare dataset
                self.log_message("Loading dataset...")
                try:
                    X, y = prepare_dataset(temp_data_dir, (128, 128))
                except Exception as e:
                    self.log_message(f"ERROR loading dataset: {str(e)}")
                    raise
                self.log_message(f"Dataset loaded: {len(X)} images")
                
                X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
                    X, y, test_size=0.2, val_size=0.1
                )
                
                self.log_message(f"Train set: {len(X_train)} images")
                self.log_message(f"Validation set: {len(X_val)} images")
                self.log_message(f"Test set: {len(X_test)} images")
                
                # Create model - optimized CNN architecture
                self.log_message("\nCreating Siamese Network (Optimized CNN Architecture)...")
                try:
                    # Use optimized CNN architecture
                    siamese_model, embedding_network = create_siamese_network(
                        input_shape=(128, 128, 3)
                    )
                    self.log_message("Siamese Network created successfully")
                    # Optimized learning rate
                    siamese_model = compile_model(siamese_model, learning_rate=0.0005)
                    self.log_message("Model compiled successfully with learning_rate=0.0005 (optimized)")
                except Exception as e:
                    self.log_message(f"ERROR creating model: {str(e)}")
                    raise
                
                # Callbacks (ProgressCallback will be added after steps_per_epoch is calculated)
                callbacks = [
                    ModelCheckpoint(
                        model_path,
                        monitor='val_accuracy',
                        save_best_only=True,
                        mode='max',
                        verbose=0
                    ),
                    EarlyStopping(
                        monitor='val_accuracy',
                        patience=max(15, patience),  # More patience for better convergence
                        restore_best_weights=True,
                        verbose=0,
                        min_delta=0.0005  # Smaller minimum improvement
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.3,  # More aggressive reduction
                        patience=5,  # More patience before reducing
                        min_lr=1e-8,
                        verbose=0,
                        cooldown=2  # Cooldown period
                    ),
                    # Cosine annealing learning rate schedule for better convergence
                    keras.callbacks.LearningRateScheduler(
                        lambda epoch: 0.0005 * (0.98 ** epoch),  # Adjusted for 0.0005 base LR
                        verbose=0
                    )
                ]
                
                # Create data generators for Siamese Network with augmentation
                self.log_message("Creating data generators with augmentation...")
                try:
                    # Training: use augmentation
                    train_gen = create_data_generator_for_siamese(X_train, y_train, batch_size, augment=True)
                    # Validation: no augmentation
                    val_gen = create_data_generator_for_siamese(X_val, y_val, batch_size, augment=False)
                    self.log_message("Data generators created successfully (augmentation enabled for training)")
                except Exception as e:
                    self.log_message(f"ERROR creating data generators: {str(e)}")
                    raise
                
                # Calculate steps
                steps_per_epoch = len(X_train) // batch_size
                validation_steps = len(X_val) // batch_size
                
                self.log_message(f"Steps per epoch: {steps_per_epoch}")
                self.log_message(f"Validation steps: {validation_steps}")
                
                # Custom callback for progress with batch updates
                class ProgressCallback(keras.callbacks.Callback):
                    def __init__(self, gui_app, total_epochs, steps_per_epoch):
                        self.gui_app = gui_app
                        self.total_epochs = total_epochs
                        self.steps_per_epoch = steps_per_epoch
                        self.current_epoch = 0
                        self.batch_count = 0
                    
                    def on_epoch_begin(self, epoch, logs=None):
                        self.current_epoch = epoch
                        self.batch_count = 0
                        progress = 15 + (epoch / self.total_epochs) * 80
                        if IS_MAC:
                            self.gui_app.root.after(0, lambda: self.gui_app.update_progress(
                                progress,
                                f"Training epoch {epoch + 1}/{self.total_epochs}"
                            ))
                            self.gui_app.root.after(0, lambda: self.gui_app.log_message(f"\nEpoch {epoch + 1}/{self.total_epochs}"))
                        else:
                            self.gui_app.update_progress(
                                progress,
                                f"Training epoch {epoch + 1}/{self.total_epochs}"
                            )
                            self.gui_app.log_message(f"\nEpoch {epoch + 1}/{self.total_epochs}")
                    
                    def on_batch_end(self, batch, logs=None):
                        self.batch_count += 1
                        # Update every 5 batches to avoid GUI lag
                        if self.batch_count % 5 == 0 or self.batch_count == self.steps_per_epoch:
                            epoch_progress = self.batch_count / self.steps_per_epoch
                            progress = 15 + ((self.current_epoch + epoch_progress) / self.total_epochs) * 80
                            if IS_MAC:
                                self.gui_app.root.after(0, lambda p=progress, b=self.batch_count, s=self.steps_per_epoch, e=self.current_epoch: self.gui_app.update_progress(
                                    p,
                                    f"Epoch {e + 1}/{self.total_epochs} - Batch {b}/{s}"
                                ))
                            else:
                                self.gui_app.update_progress(
                                    progress,
                                    f"Epoch {self.current_epoch + 1}/{self.total_epochs} - Batch {self.batch_count}/{self.steps_per_epoch}"
                                )
                    
                    def on_epoch_end(self, epoch, logs=None):
                        if logs:
                            msg = f"  Loss: {logs.get('loss', 0):.4f}, Accuracy: {logs.get('accuracy', 0):.4f}"
                            if 'val_loss' in logs:
                                msg += f"\n  Val Loss: {logs.get('val_loss', 0):.4f}, Val Accuracy: {logs.get('val_accuracy', 0):.4f}"
                            if IS_MAC:
                                self.gui_app.root.after(0, lambda m=msg: self.gui_app.log_message(m))
                            else:
                                self.gui_app.log_message(msg)
                
                # Add progress callback to callbacks list
                progress_callback = ProgressCallback(self, epochs, steps_per_epoch)
                callbacks.insert(0, progress_callback)
                
                # Train model
                self.log_message("\nStarting training...")
                history = siamese_model.fit(
                    train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=validation_steps,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Prepare test pairs
                self.update_progress(95, "Evaluating model...")
                self.log_message("\nPreparing test pairs...")
                test_pairs_a, test_pairs_b, test_labels = prepare_siamese_pairs(
                    X_test, y_test, num_pairs=min(500, len(X_test) * 2)
                )
                
                # Evaluate on test set
                self.log_message("Evaluating on test set...")
                test_loss, test_accuracy = siamese_model.evaluate(
                    [test_pairs_a, test_pairs_b], test_labels, verbose=0
                )
                
                # Get predictions
                y_pred = (siamese_model.predict([test_pairs_a, test_pairs_b], verbose=0) > 0.5).astype(int).flatten()
                metrics = evaluate_model(test_labels, y_pred, "Siamese Network")
                
                # Save embedding network separately
                embedding_network.save(embedding_path)
                
                self.log_message("\n" + "=" * 60)
                self.log_message("Training Complete!")
                self.log_message(f"Test Loss: {test_loss:.4f}")
                self.log_message(f"Test Accuracy: {test_accuracy:.4f}")
                self.log_message(f"Precision: {metrics['precision']:.4f}")
                self.log_message(f"Recall: {metrics['recall']:.4f}")
                self.log_message(f"F1-Score: {metrics['f1_score']:.4f}")
                self.log_message("=" * 60)
                self.log_message(f"\nModel saved to: {model_path}")
                self.log_message(f"Embedding network saved to: {embedding_path}")
            
            self.update_progress(100, "Training completed!")
            
            # Show success message
            if IS_MAC:
                self.root.after(0, lambda: messagebox.showinfo("Success", f"{model_type} model training completed!"))
            else:
                messagebox.showinfo("Success", f"{model_type} model training completed!")
            
            # Cleanup
            if os.path.exists(temp_data_dir):
                shutil.rmtree(temp_data_dir)
            
            # Reload models
            self.log_message("\nReloading models...")
            self.load_models()
            self.log_message("Models reloaded successfully!")
            
        except Exception as e:
            import traceback
            error_msg = f"Training failed:\n{str(e)}"
            full_traceback = traceback.format_exc()
            
            # Log error details
            self.log_message(f"\n{'='*60}")
            self.log_message("ERROR OCCURRED:")
            self.log_message(f"{error_msg}")
            self.log_message(f"\nTraceback:")
            self.log_message(full_traceback)
            self.log_message(f"{'='*60}")
            
            # Also print to console
            print(f"\n{'='*60}")
            print("TRAINING ERROR:")
            print(error_msg)
            print("\nFull Traceback:")
            print(full_traceback)
            print(f"{'='*60}")
            
            # Show error dialog
            def show_error():
                try:
                    messagebox.showerror("Training Error", error_msg)
                except:
                    print("Could not show error dialog")
            
            if IS_MAC:
                self.root.after(0, show_error)
            else:
                show_error()
        finally:
            self.training_in_progress = False
            self.update_progress(0, "Ready to train")
            self.log_message("\nTraining thread finished.")


def main():
    # macOS specific initialization
    if IS_MAC:
        # Suppress TensorFlow warnings on macOS
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Configure for macOS
        try:
            # Set process name for macOS
            import setproctitle
            setproctitle.setproctitle("Signature Forgery Detection")
        except ImportError:
            pass  # Optional package
    
    root = tk.Tk()
    
    # macOS specific window setup
    if IS_MAC:
        try:
            # Ensure window appears in dock
            root.createcommand('tk::mac::ReopenApplication', lambda: root.deiconify())
            # Set app name for macOS
            root.tk.call('::tk::unsupported::MacWindowStyle', 'style', root._w, 'documentProc', 'closeBox collapseBox resizable')
        except:
            pass
    
    app = SignatureForgeryDetectionApp(root)
    
    # Handle window close event
    def on_closing():
        if app.training_in_progress:
            if messagebox.askokcancel("Quit", "Training is in progress. Do you want to quit?"):
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start main loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.destroy()


if __name__ == "__main__":
    main()

