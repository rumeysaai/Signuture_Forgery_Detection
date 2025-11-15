"""
Training scripts for CNN and Siamese Network models
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json
from datetime import datetime

from models import create_cnn_model, create_siamese_network, compile_model, create_data_generator_for_siamese
from utils import prepare_dataset, split_dataset, evaluate_model, plot_training_history, plot_confusion_matrix


def train_cnn_model(data_dir, model_save_path='models/cnn_model.h5', 
                   epochs=50, batch_size=32, learning_rate=0.001,
                   img_size=(128, 128)):
    """
    Train CNN model for signature forgery detection
    
    Args:
        data_dir: Directory containing genuine and forged folders
        model_save_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        img_size: Image size
    
    Returns:
        Trained model and training history
    """
    print("=" * 60)
    print("Training CNN Model")
    print("=" * 60)
    
    # Prepare dataset
    print("\nLoading dataset...")
    X, y = prepare_dataset(data_dir, img_size)
    print(f"Dataset loaded: {len(X)} images")
    print(f"Genuine: {np.sum(y == 1)}, Forged: {np.sum(y == 0)}")
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, test_size=0.2, val_size=0.1
    )
    
    print(f"\nTrain set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Create model
    print("\nCreating CNN model...")
    model = create_cnn_model(input_shape=(*img_size, 3))
    model = compile_model(model, learning_rate=learning_rate)
    
    print("\nModel Architecture:")
    model.summary()
    
    # Create callbacks
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
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
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    # Evaluate metrics
    metrics = evaluate_model(y_test, y_pred, "CNN Model")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, "CNN Model", 
                         save_path='results/cnn_confusion_matrix.png')
    
    # Plot training history
    plot_training_history(history, save_path='results/cnn_training_history.png')
    
    # Save metrics
    metrics_save_path = 'results/cnn_metrics.json'
    os.makedirs('results', exist_ok=True)
    with open(metrics_save_path, 'w') as f:
        json.dump({
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss)
        }, f, indent=2)
    
    print(f"\nModel saved to: {model_save_path}")
    print(f"Metrics saved to: {metrics_save_path}")
    
    return model, history, metrics


def train_siamese_model(data_dir, model_save_path='models/siamese_model.h5',
                       epochs=50, batch_size=32, learning_rate=0.001,
                       img_size=(128, 128)):
    """
    Train Siamese Network for signature verification
    
    Args:
        data_dir: Directory containing genuine and forged folders
        model_save_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        img_size: Image size
    
    Returns:
        Trained model and training history
    """
    print("=" * 60)
    print("Training Siamese Network")
    print("=" * 60)
    
    # Prepare dataset
    print("\nLoading dataset...")
    X, y = prepare_dataset(data_dir, img_size)
    print(f"Dataset loaded: {len(X)} images")
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, test_size=0.2, val_size=0.1
    )
    
    print(f"\nTrain set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Create model
    print("\nCreating Siamese Network...")
    siamese_model, embedding_network = create_siamese_network(input_shape=(*img_size, 3))
    siamese_model = compile_model(siamese_model, learning_rate=learning_rate)
    
    print("\nModel Architecture:")
    siamese_model.summary()
    
    # Create callbacks
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Create data generators
    train_gen = create_data_generator_for_siamese(X_train, y_train, batch_size)
    val_gen = create_data_generator_for_siamese(X_val, y_val, batch_size)
    
    # Calculate steps
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    
    # Train model
    print("\nStarting training...")
    history = siamese_model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Prepare test pairs
    from models import prepare_siamese_pairs
    test_pairs_a, test_pairs_b, test_labels = prepare_siamese_pairs(
        X_test, y_test, num_pairs=min(500, len(X_test) * 2)
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = siamese_model.evaluate(
        [test_pairs_a, test_pairs_b], test_labels, verbose=0
    )
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    y_pred = (siamese_model.predict([test_pairs_a, test_pairs_b]) > 0.5).astype(int).flatten()
    
    # Evaluate metrics
    metrics = evaluate_model(test_labels, y_pred, "Siamese Network")
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, y_pred, "Siamese Network",
                         save_path='results/siamese_confusion_matrix.png')
    
    # Plot training history
    plot_training_history(history, save_path='results/siamese_training_history.png')
    
    # Save metrics
    metrics_save_path = 'results/siamese_metrics.json'
    os.makedirs('results', exist_ok=True)
    with open(metrics_save_path, 'w') as f:
        json.dump({
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss)
        }, f, indent=2)
    
    # Save embedding network separately for inference
    embedding_save_path = 'models/siamese_embedding.h5'
    embedding_network.save(embedding_save_path)
    
    print(f"\nModel saved to: {model_save_path}")
    print(f"Embedding network saved to: {embedding_save_path}")
    print(f"Metrics saved to: {metrics_save_path}")
    
    return siamese_model, history, metrics


if __name__ == "__main__":
    # Example usage
    # Note: You can use any directory path here, or use GUI training instead
    # GUI training allows you to select folders from anywhere on your computer
    
    # Option 1: Use data/ folder (if you have it)
    data_directory = "data"  # Should contain 'genuine' and 'forged' subfolders
    
    # Option 2: Use any other path
    # data_directory = "path/to/your/dataset"  # Should contain 'genuine' and 'forged' subfolders
    
    # Check if directory exists
    if not os.path.exists(data_directory):
        print(f"Error: Directory '{data_directory}' not found!")
        print("\nOptions:")
        print("1. Create data/ folder with genuine/ and forged/ subfolders")
        print("2. Use GUI training: python main.py (then select folders from Training tab)")
        print("3. Modify data_directory variable above to point to your dataset")
        exit(1)
    
    # Train CNN model
    print("\n" + "="*60)
    print("TRAINING CNN MODEL")
    print("="*60)
    cnn_model, cnn_history, cnn_metrics = train_cnn_model(
        data_directory,
        model_save_path='models/cnn_model.h5',
        epochs=50,
        batch_size=32
    )
    
    # Train Siamese Network
    print("\n" + "="*60)
    print("TRAINING SIAMESE NETWORK")
    print("="*60)
    siamese_model, siamese_history, siamese_metrics = train_siamese_model(
        data_directory,
        model_save_path='models/siamese_model.h5',
        epochs=50,
        batch_size=32
    )

