#!/usr/bin/env python3
"""
Console script to train Siamese Network model
Usage: python train_siamese.py
"""
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train import train_siamese_model

if __name__ == "__main__":
    # Data directory containing genuine/ and forged/ folders
    data_directory = "data"
    
    # Check if directory exists
    if not os.path.exists(data_directory):
        print(f"Error: Directory '{data_directory}' not found!")
        print(f"\nExpected structure:")
        print(f"  {data_directory}/")
        print(f"    ├── genuine/")
        print(f"    └── forged/")
        sys.exit(1)
    
    # Check if subdirectories exist
    genuine_dir = os.path.join(data_directory, "genuine")
    forged_dir = os.path.join(data_directory, "forged")
    
    if not os.path.exists(genuine_dir):
        print(f"Error: '{genuine_dir}' directory not found!")
        sys.exit(1)
    
    if not os.path.exists(forged_dir):
        print(f"Error: '{forged_dir}' directory not found!")
        sys.exit(1)
    
    print("=" * 60)
    print("SIAMESE NETWORK TRAINING")
    print("=" * 60)
    print(f"Data directory: {data_directory}")
    print(f"Genuine folder: {genuine_dir}")
    print(f"Forged folder: {forged_dir}")
    print("=" * 60)
    
    # Train Siamese Network with optimized parameters
    print("\nStarting training with optimized parameters...")
    print("- Batch size: 128 (optimized for speed)")
    print("- Learning rate: 0.0005 (optimized)")
    print("- Epochs: 50")
    print("- Image size: 128x128")
    print()
    
    try:
        siamese_model, siamese_history, siamese_metrics = train_siamese_model(
            data_directory,
            model_save_path='models/siamese_model.h5',
            epochs=50,
            batch_size=128,  # Optimized batch size
            learning_rate=0.0005,  # Optimized learning rate
            img_size=(128, 128)
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Model saved to: models/siamese_model.h5")
        print(f"Embedding network saved to: models/siamese_embedding.h5")
        print(f"Metrics saved to: results/siamese_metrics.json")
        print("\nFinal Metrics:")
        print(f"  Accuracy: {siamese_metrics['accuracy']:.4f}")
        print(f"  Precision: {siamese_metrics['precision']:.4f}")
        print(f"  Recall: {siamese_metrics['recall']:.4f}")
        print(f"  F1-Score: {siamese_metrics['f1_score']:.4f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

