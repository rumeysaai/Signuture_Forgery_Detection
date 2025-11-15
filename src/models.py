"""
CNN and Siamese Network models for Signature Forgery Detection
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


def create_cnn_model(input_shape=(128, 128, 3), num_classes=1):
    """
    Create a baseline CNN model for binary classification
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes (1 for binary)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # Output Layer
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model


def create_embedding_network(input_shape=(128, 128, 3)):
    """
    Create embedding network for Siamese Network
    
    Args:
        input_shape: Shape of input images
    
    Returns:
        Keras model that outputs embeddings
    """
    input_layer = layers.Input(shape=input_shape)
    
    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Flatten and dense layers for embedding
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    
    # Output embedding
    embedding = layers.Dense(128, name='embedding')(x)
    
    model = Model(inputs=input_layer, outputs=embedding, name='embedding_network')
    
    return model


def create_siamese_network(input_shape=(128, 128, 3)):
    """
    Create Siamese Network for signature verification
    
    Args:
        input_shape: Shape of input images
    
    Returns:
        Compiled Siamese Network model
    """
    # Create embedding network
    embedding_network = create_embedding_network(input_shape)
    
    # Two input branches
    input_a = layers.Input(shape=input_shape, name='input_a')
    input_b = layers.Input(shape=input_shape, name='input_b')
    
    # Get embeddings
    embedding_a = embedding_network(input_a)
    embedding_b = embedding_network(input_b)
    
    # Compute distance between embeddings
    distance = layers.Lambda(
        lambda embeddings: tf.abs(embeddings[0] - embeddings[1])
    )([embedding_a, embedding_b])
    
    # Classification head
    x = layers.Dense(128, activation='relu')(distance)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    siamese_model = Model(
        inputs=[input_a, input_b],
        outputs=output,
        name='siamese_network'
    )
    
    return siamese_model, embedding_network


def compile_model(model, learning_rate=0.001):
    """
    Compile model with optimizer and loss function
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_data_generator_for_siamese(X, y, batch_size=32):
    """
    Create data generator for Siamese Network
    Generates pairs of images with labels (1 for same, 0 for different)
    
    Args:
        X: Image arrays
        y: Labels
        batch_size: Batch size
    
    Yields:
        ([pair_a, pair_b], labels)
    """
    while True:
        batch_a = []
        batch_b = []
        batch_labels = []
        
        for _ in range(batch_size):
            # Randomly select two samples
            idx1 = np.random.randint(0, len(X))
            idx2 = np.random.randint(0, len(X))
            
            # Get images
            img1 = X[idx1]
            img2 = X[idx2]
            
            # Label: 1 if same class, 0 if different
            label = 1 if y[idx1] == y[idx2] else 0
            
            batch_a.append(img1)
            batch_b.append(img2)
            batch_labels.append(label)
        
        yield [np.array(batch_a), np.array(batch_b)], np.array(batch_labels)


def prepare_siamese_pairs(X, y, num_pairs=1000):
    """
    Prepare pairs for Siamese Network evaluation
    
    Args:
        X: Image arrays
        y: Labels
        num_pairs: Number of pairs to generate
    
    Returns:
        (pairs_a, pairs_b, labels)
    """
    pairs_a = []
    pairs_b = []
    labels = []
    
    for _ in range(num_pairs):
        idx1 = np.random.randint(0, len(X))
        idx2 = np.random.randint(0, len(X))
        
        pairs_a.append(X[idx1])
        pairs_b.append(X[idx2])
        labels.append(1 if y[idx1] == y[idx2] else 0)
    
    return np.array(pairs_a), np.array(pairs_b), np.array(labels)

