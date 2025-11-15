"""
Utility functions for Signature Forgery Detection
"""
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


def load_image(image_path, target_size=(128, 128)):
    """
    Load and preprocess an image
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
    
    Returns:
        Preprocessed image array
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def prepare_dataset(data_dir, img_size=(128, 128)):
    """
    Prepare dataset from directory structure
    
    Expected structure:
    data/
        genuine/
            img1.png
            img2.png
        forged/
            img1.png
            img2.png
    
    Args:
        data_dir: Root directory containing genuine and forged folders
        img_size: Target image size
    
    Returns:
        X: Image arrays
        y: Labels (1 for genuine, 0 for forged)
    """
    X = []
    y = []
    
    genuine_dir = os.path.join(data_dir, 'genuine')
    forged_dir = os.path.join(data_dir, 'forged')
    
    # Load genuine signatures
    if os.path.exists(genuine_dir):
        for filename in os.listdir(genuine_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(genuine_dir, filename)
                img = load_image(img_path, img_size)
                if img is not None:
                    X.append(img)
                    y.append(1)  # Genuine
    
    # Load forged signatures
    if os.path.exists(forged_dir):
        for filename in os.listdir(forged_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(forged_dir, filename)
                img = load_image(img_path, img_size)
                if img is not None:
                    X.append(img)
                    y.append(0)  # Forged
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion of test set
        val_size: Proportion of validation set (from remaining after test)
        random_state: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train and val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate model performance and return metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for display
    
    Returns:
        Dictionary with metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Forged', 'Genuine'],
                yticklabels=['Forged', 'Genuine'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy)
    
    Args:
        history: Training history from Keras model
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()



