# Signature Forgery Detection System

A desktop application for detecting forged signatures using Convolutional Neural Networks (CNN) and Siamese Networks. The system achieves ≥90% accuracy in detecting signature forgeries.

## Features

- **CNN-Based Classification**: Binary classification model to detect genuine vs forged signatures
- **Siamese Network**: Signature verification by comparing two signatures
- **Modern GUI**: User-friendly Tkinter desktop interface
- **High Accuracy**: Trained models achieve ≥90% accuracy
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-score metrics
- **Visualization**: Confusion matrices and training history plots

## Project Structure

```
Signuture_Forgery_Detection/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Dataset directory
│   ├── genuine/          # Genuine signature images
│   └── forged/           # Forged signature images
├── models/                # Trained model files
│   ├── cnn_model.h5
│   ├── siamese_model.h5
│   └── siamese_embedding.h5
├── results/               # Evaluation results and plots
│   ├── cnn_metrics.json
│   ├── siamese_metrics.json
│   ├── cnn_confusion_matrix.png
│   └── training_history.png
└── src/                   # Source code
    ├── models.py          # CNN and Siamese Network models
    ├── train.py           # Training scripts
    ├── evaluate.py        # Evaluation and inference
    ├── gui.py             # Tkinter GUI
    └── utils.py           # Utility functions
```

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
```bash
# On macOS (use Python 3.9-3.12 for TensorFlow compatibility):
python3.11 -m venv venv  # or python3.9, python3.10, python3.12
# On Windows:
python -m venv venv
# On Linux:
python3 -m venv venv

# Activate virtual environment:
# On macOS/Linux:
source venv/bin/activate
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On Windows (CMD):
venv\Scripts\activate.bat
```

3. **Install dependencies**:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**macOS Specific Notes:**
- Ensure you have Python 3.9-3.12 (TensorFlow doesn't support Python 3.14+ yet)
- If using Homebrew Python, you may need to install Tkinter separately or use Anaconda Python
- The application is optimized for macOS with native window behavior and font support

4. **Prepare your dataset**:
   - Organize your signature images into two folders:
     - One folder for genuine signatures
     - One folder for forged signatures
   - You can use the GUI to select these folders when training (no need to place them in a specific location)
   - Or manually organize in `data/genuine/` and `data/forged/` if using command line training

## Usage

### Training Models

1. **Train CNN Model**:
```bash
python src/train.py
```

Or train individually:
```python
from src.train import train_cnn_model, train_siamese_model

# Train CNN
train_cnn_model(
    data_dir="data/signature-verification-dataset",
    model_save_path="models/cnn_model.h5",
    epochs=50,
    batch_size=32
)

# Train Siamese Network
train_siamese_model(
    data_dir="data/signature-verification-dataset",
    model_save_path="models/siamese_model.h5",
    epochs=50,
    batch_size=32
)
```

### Running the GUI Application

**On macOS:**
```bash
# Option 1: Using the launcher script
./run_mac.sh

# Option 2: Manual activation
source venv/bin/activate
python main.py
```

**On Windows/Linux:**
```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Run the application
python main.py
```

The GUI allows you to:
- Select and load trained models
- Upload signature images
- Get real-time predictions
- View confidence scores

### Using the Models Programmatically

```python
from src.evaluate import predict_signature_cnn, predict_signature_siamese
from src.models import load_trained_model

# Load model
cnn_model = load_trained_model("models/cnn_model.h5", "cnn")

# Predict single signature
is_genuine, confidence = predict_signature_cnn(
    cnn_model, 
    "path/to/signature.png"
)

# Siamese Network - compare two signatures
siamese_model = load_trained_model("models/siamese_model.h5", "siamese")
is_match, confidence = predict_signature_siamese(
    siamese_model,
    "path/to/signature1.png",
    "path/to/signature2.png"
)
```

## Model Architecture

### CNN Model
- **Input**: 128x128x3 RGB images
- **Architecture**: 
  - 4 Convolutional blocks with Batch Normalization
  - MaxPooling and Dropout layers
  - Dense layers (512, 256 neurons)
  - Binary classification output (sigmoid)
- **Output**: Genuine (1) or Forged (0)

### Siamese Network
- **Input**: Two 128x128x3 RGB images
- **Architecture**:
  - Shared embedding network
  - Distance computation between embeddings
  - Classification head
- **Output**: Match (1) or No Match (0)

## Evaluation Metrics

The system provides comprehensive evaluation:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

## Dataset

The project works with any signature dataset:
- **Format**: PNG, JPG, or JPEG images
- **Structure**: 
  - One folder for genuine signature images
  - One folder for forged signature images
- **Usage**: Use GUI to select folders from anywhere on your computer, or organize in `data/genuine/` and `data/forged/` for command line training

## Requirements

- Python 3.9+
- TensorFlow 2.10+
- NumPy, Matplotlib, OpenCV, scikit-learn
- Tkinter (usually included with Python)

## Performance

The trained models achieve:
- **CNN Model**: ≥90% accuracy
- **Siamese Network**: ≥90% accuracy

Results are saved in the `results/` directory including:
- JSON files with metrics
- Confusion matrix plots
- Training history visualizations

## Troubleshooting

1. **Model not loading**: Ensure model files exist in `models/` directory
2. **Dataset not found**: Use GUI training to select folders from anywhere, or organize data in `data/genuine/` and `data/forged/` for command line training
3. **GUI not opening**: 
   - Check Tkinter installation: `python -m tkinter`
   - On macOS: Ensure you're using Python 3.9-3.12 (not 3.14+)
   - On macOS: If Tkinter is missing, use Anaconda Python or install via Homebrew: `brew install python-tk`
4. **Memory errors**: Reduce batch size in training scripts
5. **macOS specific issues**:
   - If window doesn't appear: Try clicking on the Python icon in the Dock
   - If fonts look wrong: The app will automatically use Helvetica as fallback
   - If TensorFlow errors: Ensure Python version is 3.9-3.12

## License

This project is for educational purposes.

## Author

Signature Forgery Detection System
Developed for signature verification and forgery detection

