"""
Evaluation and inference scripts
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import load_image, evaluate_model, plot_confusion_matrix
from models import create_cnn_model, create_siamese_network


def load_trained_model(model_path, model_type='cnn'):
    """
    Load a trained model
    
    Args:
        model_path: Path to saved model
        model_type: Type of model ('cnn' or 'siamese')
    
    Returns:
        Loaded model
    """
    if model_type == 'cnn':
        model = keras.models.load_model(model_path)
    elif model_type == 'siamese':
        model = keras.models.load_model(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def predict_signature_cnn(model, image_path, threshold=0.5):
    """
    Predict if a signature is genuine or forged using CNN
    
    Args:
        model: Trained CNN model
        image_path: Path to signature image
        threshold: Classification threshold
    
    Returns:
        Prediction (1 for genuine, 0 for forged) and confidence score
    """
    img = load_image(image_path)
    if img is None:
        return None, None
    
    # Add batch dimension
    img_batch = np.expand_dims(img, axis=0)
    
    # Predict
    prediction = model.predict(img_batch, verbose=0)[0][0]
    
    # Classify
    is_genuine = 1 if prediction > threshold else 0
    confidence = prediction if is_genuine else (1 - prediction)
    
    return is_genuine, confidence


def predict_signature_siamese(model, image_path1, image_path2, threshold=0.5):
    """
    Predict if two signatures match using Siamese Network
    
    Args:
        model: Trained Siamese model
        image_path1: Path to first signature image
        image_path2: Path to second signature image
        threshold: Classification threshold
    
    Returns:
        Prediction (1 if match, 0 if not) and confidence score
    """
    img1 = load_image(image_path1)
    img2 = load_image(image_path2)
    
    if img1 is None or img2 is None:
        return None, None
    
    # Add batch dimension
    img1_batch = np.expand_dims(img1, axis=0)
    img2_batch = np.expand_dims(img2, axis=0)
    
    # Predict
    prediction = model.predict([img1_batch, img2_batch], verbose=0)[0][0]
    
    # Classify
    is_match = 1 if prediction > threshold else 0
    confidence = prediction if is_match else (1 - prediction)
    
    return is_match, confidence


def evaluate_model_on_directory(model, data_dir, model_type='cnn'):
    """
    Evaluate model on a directory of images
    
    Args:
        model: Trained model
        data_dir: Directory containing genuine and forged folders
        model_type: Type of model ('cnn' or 'siamese')
    
    Returns:
        Evaluation metrics
    """
    from utils import prepare_dataset, split_dataset
    
    # Load dataset
    X, y = prepare_dataset(data_dir)
    
    if model_type == 'cnn':
        # Predict all images
        y_pred = (model.predict(X) > 0.5).astype(int).flatten()
        metrics = evaluate_model(y, y_pred, f"{model_type.upper()} Model")
        plot_confusion_matrix(y, y_pred, f"{model_type.upper()} Model",
                           save_path=f'results/{model_type}_full_evaluation.png')
    
    elif model_type == 'siamese':
        # For Siamese, we need pairs
        from models import prepare_siamese_pairs
        pairs_a, pairs_b, labels = prepare_siamese_pairs(X, y, num_pairs=min(1000, len(X) * 2))
        y_pred = (model.predict([pairs_a, pairs_b]) > 0.5).astype(int).flatten()
        metrics = evaluate_model(labels, y_pred, f"{model_type.upper()} Model")
        plot_confusion_matrix(labels, y_pred, f"{model_type.upper()} Model",
                           save_path=f'results/{model_type}_full_evaluation.png')
    
    return metrics


if __name__ == "__main__":
    # Example usage
    cnn_model_path = "models/cnn_model.h5"
    siamese_model_path = "models/siamese_model.h5"
    data_directory = "data/signature-verification-dataset"
    
    # Evaluate CNN model
    if os.path.exists(cnn_model_path):
        print("Loading CNN model...")
        cnn_model = load_trained_model(cnn_model_path, 'cnn')
        print("Evaluating CNN model...")
        cnn_metrics = evaluate_model_on_directory(cnn_model, data_directory, 'cnn')
    
    # Evaluate Siamese model
    if os.path.exists(siamese_model_path):
        print("\nLoading Siamese model...")
        siamese_model = load_trained_model(siamese_model_path, 'siamese')
        print("Evaluating Siamese model...")
        siamese_metrics = evaluate_model_on_directory(siamese_model, data_directory, 'siamese')

