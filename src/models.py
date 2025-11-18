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


def create_embedding_network(input_shape=(128, 128, 3), l2_reg=1e-4):
    """
    Create embedding network for Siamese Network
    Optimized CNN architecture for high accuracy (95%+ target)
    
    Args:
        input_shape: Shape of input images
        l2_reg: L2 regularization factor (default: 1e-4)
    
    Returns:
        Keras model that outputs embeddings
    """
    from tensorflow.keras.regularizers import l2
    
    input_layer = layers.Input(shape=input_shape)
    
    # Optimized CNN architecture for high accuracy (95%+ target)
    # Deeper and more sophisticated architecture
    
    # First block - more filters
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Second block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Third block
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.35)(x)
    
    # Fourth block
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Global Average Pooling instead of Flatten (better for generalization)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers - increased capacity
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu',
                    kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu',
                    kernel_regularizer=l2(l2_reg))(x)
    x = layers.Dropout(0.3)(x)
    
    # Output embedding - larger for better representation
    embedding = layers.Dense(128, name='embedding',
                           kernel_regularizer=l2(l2_reg))(x)
    
    model = Model(inputs=input_layer, outputs=embedding, name='embedding_network')
    
    return model


def create_siamese_network(input_shape=(128, 128, 3)):
    """
    Create Siamese Network for signature verification
    Uses classification head with binary crossentropy loss
    
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
    
    # Classification head with L2 regularization - optimized for high accuracy (95%+)
    from tensorflow.keras.regularizers import l2
    l2_reg = 1e-4
    
    # Deeper classification head for better discrimination
    x = layers.Dense(256, activation='relu',  # Increased capacity
                     kernel_regularizer=l2(l2_reg))(distance)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu',
                    kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu',
                    kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid', name='output',
                         kernel_regularizer=l2(l2_reg))(x)
    
    # Create model
    siamese_model = Model(
        inputs=[input_a, input_b],
        outputs=output,
        name='siamese_network'
    )
    
    return siamese_model, embedding_network


def create_siamese_network_with_contrastive(input_shape=(128, 128, 3)):
    """
    Create Siamese Network optimized for Contrastive Loss
    Outputs Euclidean distance directly (no classification head)
    
    Args:
        input_shape: Shape of input images
    
    Returns:
        Siamese Network model (outputs distance) and embedding network
    """
    # Create embedding network
    embedding_network = create_embedding_network(input_shape)
    
    # Two input branches
    input_a = layers.Input(shape=input_shape, name='input_a')
    input_b = layers.Input(shape=input_shape, name='input_b')
    
    # Get embeddings
    embedding_a = embedding_network(input_a)
    embedding_b = embedding_network(input_b)
    
    # Compute Euclidean distance between embeddings
    # L2 norm: sqrt(sum((a - b)²))
    distance = layers.Lambda(
        lambda embeddings: tf.sqrt(tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=1, keepdims=True)),
        name='euclidean_distance'
    )([embedding_a, embedding_b])
    
    # Create model - outputs distance directly
    siamese_model = Model(
        inputs=[input_a, input_b],
        outputs=distance,
        name='siamese_network_contrastive'
    )
    
    return siamese_model, embedding_network


def contrastive_loss(margin=1.0):
    """
    Contrastive Loss function for Siamese Networks
    
    Formula:
    - For positive pairs (y=1): loss = d²
    - For negative pairs (y=0): loss = max(0, margin - d)²
    
    Args:
        margin: Margin parameter for negative pairs (default: 1.0)
    
    Returns:
        Loss function
    """
    def loss(y_true, y_pred):
        """
        Args:
            y_true: Labels (1 for same class, 0 for different class)
            y_pred: Euclidean distance between embeddings
        """
        # Squared distance
        square_pred = tf.square(y_pred)
        # Margin squared
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        # Contrastive loss: y_true * d² + (1 - y_true) * max(0, margin - d)²
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    return loss


def compile_model(model, learning_rate=0.001):
    """
    Compile model with optimizer and loss function
    Uses optimized Adam settings for better generalization
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer (default: 0.001, recommended: 0.0005 for Siamese)
    
    Returns:
        Compiled model
    """
    # Use Adam optimizer with optimized settings
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def contrastive_accuracy(y_true, y_pred, threshold=0.5):
    """
    Accuracy metric for Contrastive Loss
    Predicts same class (1) if distance < threshold, different class (0) otherwise
    
    Args:
        y_true: True labels (1 for same, 0 for different)
        y_pred: Predicted distances
        threshold: Distance threshold for classification
    
    Returns:
        Accuracy value
    """
    # Predict same class (1) if distance < threshold
    predictions = tf.cast(y_pred < threshold, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, predictions), tf.float32))


def compile_siamese_with_contrastive(model, learning_rate=0.001, margin=1.0, threshold=0.5):
    """
    Compile Siamese model with Contrastive Loss
    
    Args:
        model: Siamese model (should output distance, not classification)
        learning_rate: Learning rate for optimizer
        margin: Margin parameter for Contrastive Loss
        threshold: Distance threshold for accuracy calculation
    
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=contrastive_loss(margin=margin),
        metrics=[lambda y_true, y_pred: contrastive_accuracy(y_true, y_pred, threshold)]
    )
    
    return model


def create_data_generator_for_siamese(X, y, batch_size=32, augment=True):
    """
    Create data generator for Siamese Network with optional augmentation
    Generates pairs of images with labels (1 for same, 0 for different)
    
    Args:
        X: Image arrays
        y: Labels
        batch_size: Batch size
        augment: Whether to apply data augmentation (default: True)
    
    Yields:
        ([pair_a, pair_b], labels)
    """
    def generator():
        """Optimized generator - minimal augmentation for speed"""
        while True:
            # Pre-allocate arrays for better performance
            batch_a = np.zeros((batch_size, *X[0].shape), dtype=np.float32)
            batch_b = np.zeros((batch_size, *X[0].shape), dtype=np.float32)
            batch_labels = np.zeros(batch_size, dtype=np.float32)
            
            for i in range(batch_size):
                # Randomly select two samples
                idx1 = np.random.randint(0, len(X))
                idx2 = np.random.randint(0, len(X))
                
                # Get images (no copy needed, we'll modify in place if augmenting)
                img1 = X[idx1]
                img2 = X[idx2]
                
                # Minimal augmentation - only if enabled and only 30% chance per image
                if augment and np.random.random() < 0.3:
                    # Very light augmentation - just brightness
                    delta = np.random.uniform(-0.05, 0.05)
                    img1 = np.clip(img1 + delta, 0.0, 1.0)
                
                if augment and np.random.random() < 0.3:
                    delta = np.random.uniform(-0.05, 0.05)
                    img2 = np.clip(img2 + delta, 0.0, 1.0)
                
                batch_a[i] = img1
                batch_b[i] = img2
                
                # Label: 1 if same class, 0 if different
                batch_labels[i] = 1.0 if y[idx1] == y[idx2] else 0.0
            
            yield (batch_a, batch_b), batch_labels
    
    # Get image shape
    img_shape = X[0].shape if len(X) > 0 else (128, 128, 3)
    
    # Create TensorFlow Dataset with proper output signature
    output_signature = (
        (
            tf.TensorSpec(shape=(None, *img_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *img_shape), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    # Optimize dataset performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch batches for faster training
    
    return dataset


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

