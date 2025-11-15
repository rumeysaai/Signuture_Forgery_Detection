# Usage Guide - Signature Forgery Detection

## Quick Start

### Windows PowerShell Execution Policy Issue

If you encounter "running scripts is disabled" error, use one of these methods:

**Method 1: Use venv Python directly (Recommended)**
```powershell
.\venv\Scripts\python.exe src/train.py
.\venv\Scripts\python.exe main.py
```

**Method 2: Use batch file**
```cmd
run_with_venv.bat src/train.py
run_with_venv.bat main.py
```

**Method 3: Change PowerShell execution policy (one-time)**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

## Step-by-Step Usage

### 1. Prepare Your Dataset

**Option A: Use GUI (Recommended - No specific folder structure needed)**
- Open GUI: `.\venv\Scripts\python.exe main.py`
- Go to "🎓 Training" tab
- Select folders containing genuine and forged signatures from anywhere on your computer

**Option B: Organize manually (for command line training)**
- Create `data/genuine/` folder and place genuine signature images
- Create `data/forged/` folder and place forged signature images

### 2. Train Models

```powershell
.\venv\Scripts\python.exe src/train.py
```

This will:
- Train CNN model (saves to `models/cnn_model.h5`)
- Train Siamese Network (saves to `models/siamese_model.h5`)
- Generate evaluation metrics and plots in `results/`

### 3. Run GUI Application

```powershell
.\venv\Scripts\python.exe main.py
```

GUI Features:
- Load trained models
- Select signature images
- Get predictions with confidence scores
- Switch between CNN and Siamese models

### 4. Evaluate Models

```powershell
.\venv\Scripts\python.exe src/evaluate.py
```

## Command Reference

### Training
```powershell
# Train both models
.\venv\Scripts\python.exe src/train.py

# Or import and use programmatically
python
>>> from src.train import train_cnn_model, train_siamese_model
>>> train_cnn_model("data", epochs=50, batch_size=32)
```

### Evaluation
```powershell
# Evaluate models
.\venv\Scripts\python.exe src/evaluate.py

# Or use programmatically
python
>>> from src.evaluate import predict_signature_cnn
>>> from src.models import load_trained_model
>>> model = load_trained_model("models/cnn_model.h5", "cnn")
>>> is_genuine, confidence = predict_signature_cnn(model, "path/to/signature.png")
```

## Troubleshooting

### PowerShell Script Execution Error
**Problem:** `Activate.ps1 cannot be loaded because running scripts is disabled`

**Solution:** Use venv Python directly:
```powershell
.\venv\Scripts\python.exe [script_name]
```

### Dataset Not Found
**Problem:** `'genuine' folder not found in data`

**Solution:**
1. Use GUI training to select folders from anywhere on your computer
2. Or manually organize files into `data/genuine/` and `data/forged/` for command line training

### Model Training Errors
**Problem:** Out of memory or CUDA errors

**Solution:**
- Reduce batch size in `src/train.py`
- Use smaller image size
- Close other applications

### Import Errors
**Problem:** Module not found errors

**Solution:**
- Ensure venv is activated or use `.\venv\Scripts\python.exe`
- Install requirements: `.\venv\Scripts\python.exe -m pip install -r requirements.txt`

## File Structure After Setup

```
Signuture_Forgery_Detection/
├── data/
│   ├── genuine/          # Genuine signatures
│   └── forged/           # Forged signatures
├── models/
│   ├── cnn_model.h5      # Trained CNN model
│   ├── siamese_model.h5  # Trained Siamese model
│   └── siamese_embedding.h5
├── results/
│   ├── cnn_metrics.json
│   ├── siamese_metrics.json
│   ├── cnn_confusion_matrix.png
│   └── training_history.png
└── src/                  # Source code
```

## Next Steps

1. ✅ Download dataset
2. ✅ Train models (wait for ≥90% accuracy)
3. ✅ Run GUI and test predictions
4. ✅ Review results in `results/` folder

