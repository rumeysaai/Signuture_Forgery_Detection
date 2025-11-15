"""
Quick start script for Signature Forgery Detection System
This script helps set up and run the application
"""
import os
import sys


def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'tensorflow',
        'numpy',
        'matplotlib',
        'cv2',
        'sklearn',
        'PIL',
        'tkinter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            elif package == 'PIL':
                from PIL import Image
            elif package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing packages:", ", ".join(missing_packages))
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✓ All required packages are installed!")
    return True


def check_dataset():
    """Check if dataset exists"""
    data_path = "data"
    genuine_path = os.path.join(data_path, "genuine")
    forged_path = os.path.join(data_path, "forged")
    
    if not os.path.exists(genuine_path) or not os.path.exists(forged_path):
        print("⚠ Dataset not found in data/ folder!")
        print("\nOptions:")
        print("1. Use GUI training to select folders from anywhere:")
        print("   python main.py (then go to Training tab)")
        print("2. Manually organize files into data/genuine/ and data/forged/")
        return False
    
    genuine_count = len([f for f in os.listdir(genuine_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    forged_count = len([f for f in os.listdir(forged_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if genuine_count == 0 or forged_count == 0:
        print("⚠ Dataset folders are empty!")
        return False
    
    print(f"✓ Dataset found: {genuine_count} genuine, {forged_count} forged signatures")
    return True


def check_models():
    """Check if trained models exist"""
    cnn_model = "models/cnn_model.h5"
    siamese_model = "models/siamese_model.h5"
    
    models_exist = os.path.exists(cnn_model) or os.path.exists(siamese_model)
    
    if not models_exist:
        print("⚠ No trained models found!")
        print("To train models, run:")
        print("  python src/train.py")
        return False
    
    if os.path.exists(cnn_model):
        print("✓ CNN model found")
    if os.path.exists(siamese_model):
        print("✓ Siamese model found")
    
    return True


def main():
    """Main quick start function"""
    print("=" * 60)
    print("Signature Forgery Detection System - Quick Start")
    print("=" * 60)
    print()
    
    # Check requirements
    print("Checking requirements...")
    if not check_requirements():
        return
    print()
    
    # Check dataset
    print("Checking dataset...")
    dataset_ok = check_dataset()
    print()
    
    # Check models
    print("Checking models...")
    models_ok = check_models()
    print()
    
    # Summary
    print("=" * 60)
    if dataset_ok and models_ok:
        print("✓ System is ready!")
        print("\nTo run the GUI application:")
        print("  python main.py")
    elif dataset_ok and not models_ok:
        print("⚠ Dataset is ready, but models need to be trained.")
        print("\nTo train models:")
        print("  python src/train.py")
        print("\nThen run the GUI:")
        print("  python main.py")
    elif not dataset_ok:
        print("⚠ Dataset not found in data/ folder.")
        print("\nOptions:")
        print("1. Use GUI training (recommended):")
        print("   python main.py (then select folders in Training tab)")
        print("2. Organize files in data/genuine/ and data/forged/")
    print("=" * 60)


if __name__ == "__main__":
    main()

