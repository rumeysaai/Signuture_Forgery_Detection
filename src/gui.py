"""
Tkinter GUI for Signature Forgery Detection
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys
import threading
import numpy as np
import tensorflow as tf
from tensorflow import keras
# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils import load_image
from models import create_cnn_model, create_siamese_network


class SignatureForgeryDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Forgery Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
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
        
        # Create GUI
        self.create_widgets()
        
        # Load models if they exist
        self.load_models()
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="Signature Forgery Detection System",
            font=('Arial', 20, 'bold'),
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
        predict_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
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
        self.batch_size_var = tk.StringVar(value="32")
        batch_entry = tk.Entry(batch_frame, textvariable=self.batch_size_var, width=10)
        batch_entry.pack(side=tk.LEFT, padx=5)
        
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
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Analyzing signature...\n")
            self.results_text.config(state=tk.DISABLED)
            
            is_genuine, confidence = predict_signature_cnn(
                self.cnn_model,
                self.selected_image_path
            )
            
            if is_genuine is None:
                raise Exception("Failed to process image")
            
            result_text = f"CNN Model Prediction:\n"
            result_text += f"{'='*40}\n\n"
            
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
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Comparing signatures...\n")
            self.results_text.config(state=tk.DISABLED)
            
            is_match, confidence = predict_signature_siamese(
                self.siamese_model,
                self.selected_image1_path,
                self.selected_image2_path
            )
            
            if is_match is None:
                raise Exception("Failed to process images")
            
            result_text = f"Siamese Network Prediction:\n"
            result_text += f"{'='*40}\n\n"
            
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
        """Add message to training logs"""
        self.training_logs.config(state=tk.NORMAL)
        self.training_logs.insert(tk.END, message + "\n")
        self.training_logs.see(tk.END)
        self.training_logs.config(state=tk.DISABLED)
        self.root.update()
    
    def update_progress(self, value, message=""):
        """Update training progress bar"""
        self.progress_var.set(value)
        if message:
            self.progress_label.config(text=message)
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
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for epochs and batch size!")
            return
        
        model_type = self.training_model_type.get()
        
        # Confirm training
        confirm = messagebox.askyesno(
            "Confirm Training",
            f"Start training {model_type} model?\n\n"
            f"Epochs: {epochs}\n"
            f"Batch Size: {batch_size}\n\n"
            f"This may take a long time. Continue?"
        )
        
        if not confirm:
            return
        
        # Start training in separate thread
        self.training_in_progress = True
        self.progress_var.set(0)
        self.training_logs.config(state=tk.NORMAL)
        self.training_logs.delete(1.0, tk.END)
        self.training_logs.config(state=tk.DISABLED)
        
        threading.Thread(
            target=self.train_model_thread,
            args=(model_type, epochs, batch_size),
            daemon=True
        ).start()
    
    def train_model_thread(self, model_type, epochs, batch_size):
        """Training thread function"""
        try:
            import shutil
            
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
                        patience=10,
                        restore_best_weights=True,
                        verbose=0
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
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
                
                # Similar implementation for Siamese
                self.log_message("Siamese training implementation...")
                # TODO: Implement Siamese training with progress callbacks
                messagebox.showinfo("Info", "Siamese training will be implemented soon!")
            
            self.update_progress(100, "Training completed!")
            messagebox.showinfo("Success", f"{model_type} model training completed!")
            
            # Cleanup
            if os.path.exists(temp_data_dir):
                shutil.rmtree(temp_data_dir)
            
            # Reload models
            self.load_models()
            
        except Exception as e:
            self.log_message(f"\nERROR: {str(e)}")
            messagebox.showerror("Training Error", f"Training failed:\n{str(e)}")
        finally:
            self.training_in_progress = False


def main():
    root = tk.Tk()
    app = SignatureForgeryDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

