"""
CNN and Siamese Network models for Signature Forgery Detection
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import os


def create_cnn_model(input_shape=(128, 128, 3), num_classes=1):
    """
    Create an advanced CNN model for binary classification
    Optimized for 90%+ accuracy
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes (1 for binary)
    
    Returns:
        Compiled Keras model
    """
    from tensorflow.keras.regularizers import l2
    
    model = keras.Sequential([
        # First Convolutional Block - Enhanced
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape,
                     kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block - Enhanced
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Third Convolutional Block - Enhanced
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.35),
        
        # Fourth Convolutional Block - Enhanced
        layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Global Average Pooling (better than Flatten for generalization)
        layers.GlobalAveragePooling2D(),
        
        # Dense Layers - Increased capacity
        layers.Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output Layer
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model


def create_embedding_network(input_shape=(128, 128, 3), l2_reg=1e-4):
    """
    Create advanced embedding network for Siamese Network
    Optimized for 90%+ accuracy with deep architecture
    
    Args:
        input_shape: Shape of input images
        l2_reg: L2 regularization factor (default: 1e-4)
    
    Returns:
        Keras model that outputs embeddings
    """
    from tensorflow.keras.regularizers import l2
    
    input_layer = layers.Input(shape=input_shape)
    
    # First Convolutional Block - Enhanced with double conv
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Second Convolutional Block - Enhanced
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Third Convolutional Block - Enhanced
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.35)(x)
    
    # Fourth Convolutional Block - Enhanced
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Global Average Pooling (better for generalization)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense Layers - Increased capacity for better representation
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output embedding layer - larger for better representation
    embedding = layers.Dense(256, name='embedding', kernel_regularizer=l2(l2_reg))(x)
    
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
    
    # Classification head - Enhanced for 90%+ accuracy
    from tensorflow.keras.regularizers import l2
    l2_reg = 1e-4
    
    # Concatenate distance with embeddings for richer features
    combined = layers.Concatenate()([distance, embedding_a, embedding_b])
    
    # Deeper and wider classification head for better discrimination
    x = layers.Dense(512, activation='relu',  # Increased capacity
                     kernel_regularizer=l2(l2_reg))(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu',
                    kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu',
                    kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu',
                    kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
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
    Uses optimized Adam settings for better generalization and 90%+ accuracy
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer (default: 0.001)
    
    Returns:
        Compiled model
    """
    # Use Adam optimizer with optimized settings for high accuracy
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False
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
        """Enhanced generator with stronger data augmentation"""
        while True:
            # Pre-allocate arrays for better performance
            batch_a = np.zeros((batch_size, *X[0].shape), dtype=np.float32)
            batch_b = np.zeros((batch_size, *X[0].shape), dtype=np.float32)
            batch_labels = np.zeros(batch_size, dtype=np.float32)
            
            for i in range(batch_size):
                # Randomly select two samples
                idx1 = np.random.randint(0, len(X))
                idx2 = np.random.randint(0, len(X))
                
                # Get images (copy for augmentation)
                img1 = X[idx1].copy()
                img2 = X[idx2].copy()
                
                # Enhanced augmentation - applied more frequently and with more transformations
                if augment:
                    # Apply augmentation with 70% probability per image
                    if np.random.random() < 0.7:
                        # Brightness adjustment
                        delta = np.random.uniform(-0.15, 0.15)
                        img1 = np.clip(img1 + delta, 0.0, 1.0)
                        
                        # Contrast adjustment
                        if np.random.random() < 0.5:
                            factor = np.random.uniform(0.85, 1.15)
                            img1 = np.clip((img1 - 0.5) * factor + 0.5, 0.0, 1.0)
                        
                        # Small rotation simulation (via translation)
                        if np.random.random() < 0.3:
                            shift = np.random.randint(-2, 3)
                            if shift != 0:
                                img1 = np.roll(img1, shift, axis=0)
                                img1 = np.roll(img1, shift, axis=1)
                    
                    if np.random.random() < 0.7:
                        # Brightness adjustment
                        delta = np.random.uniform(-0.15, 0.15)
                        img2 = np.clip(img2 + delta, 0.0, 1.0)
                        
                        # Contrast adjustment
                        if np.random.random() < 0.5:
                            factor = np.random.uniform(0.85, 1.15)
                            img2 = np.clip((img2 - 0.5) * factor + 0.5, 0.0, 1.0)
                        
                        # Small rotation simulation (via translation)
                        if np.random.random() < 0.3:
                            shift = np.random.randint(-2, 3)
                            if shift != 0:
                                img2 = np.roll(img2, shift, axis=0)
                                img2 = np.roll(img2, shift, axis=1)
                
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


def export_model_to_onnx(model, output_path, input_shape=(128, 128, 3)):
    """
    Export Keras model to ONNX format
    
    Args:
        model: Keras model to export
        output_path: Path to save ONNX model (e.g., 'models/cnn_model.onnx')
        input_shape: Input shape of the model
    
    Returns:
        Path to saved ONNX model
    """
    try:
        import tf2onnx
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Convert model to ONNX
        spec = (tf.TensorSpec((None, *input_shape), tf.float32, name="input"),)
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        
        # Save ONNX model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"Model exported to ONNX: {output_path}")
        return output_path
        
    except ImportError:
        print("Warning: tf2onnx not installed. Install with: pip install tf2onnx")
        return None
    except Exception as e:
        print(f"Error exporting to ONNX: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def export_model_to_openvino(model, output_dir, model_name="model", input_shape=(128, 128, 3)):
    """
    Export Keras model to OpenVINO IR format
    
    Args:
        model: Keras model to export
        output_dir: Directory to save OpenVINO model (e.g., 'models/openvino')
        model_name: Name of the model (default: 'model')
        input_shape: Input shape of the model
    
    Returns:
        Path to saved OpenVINO model directory
    """
    try:
        from openvino.tools import mo
        from openvino import save_model
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model as SavedModel format first (required for OpenVINO conversion)
        saved_model_path = os.path.join(output_dir, "saved_model")
        model.save(saved_model_path)
        
        # Convert SavedModel to OpenVINO IR
        ov_model = mo.convert_model(
            saved_model_path,
            input_shape=[1, *input_shape]
        )
        
        # Save OpenVINO IR
        xml_path = os.path.join(output_dir, f"{model_name}.xml")
        save_model(ov_model, xml_path)
        
        print(f"Model exported to OpenVINO: {output_dir}")
        print(f"  - XML: {xml_path}")
        print(f"  - BIN: {xml_path.replace('.xml', '.bin')}")
        
        # Clean up temporary SavedModel
        import shutil
        if os.path.exists(saved_model_path):
            shutil.rmtree(saved_model_path)
        
        return output_dir
        
    except ImportError:
        print("Warning: openvino not installed. Install with: pip install openvino")
        return None
    except Exception as e:
        print(f"Error exporting to OpenVINO: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def export_siamese_model_to_onnx(siamese_model, embedding_model, output_path, input_shape=(128, 128, 3)):
    """
    Export Siamese model and embedding network to ONNX format
    
    Args:
        siamese_model: Full Siamese model (takes two inputs)
        embedding_model: Embedding network model (takes one input)
        output_path: Base path for ONNX models (e.g., 'models/siamese')
        input_shape: Input shape of the model
    
    Returns:
        Dictionary with paths to saved ONNX models
    """
    results = {}
    
    try:
        import tf2onnx
        
        # Export embedding network (single input)
        embedding_onnx_path = f"{output_path}_embedding.onnx"
        os.makedirs(os.path.dirname(embedding_onnx_path) if os.path.dirname(embedding_onnx_path) else '.', exist_ok=True)
        spec_embedding = (tf.TensorSpec((None, *input_shape), tf.float32, name="input"),)
        onnx_embedding, _ = tf2onnx.convert.from_keras(embedding_model, input_signature=spec_embedding, opset=13)
        with open(embedding_onnx_path, "wb") as f:
            f.write(onnx_embedding.SerializeToString())
        print(f"Embedding network exported to ONNX: {embedding_onnx_path}")
        results['embedding'] = embedding_onnx_path
        
        # Export full Siamese model (two inputs)
        siamese_onnx_path = f"{output_path}_model.onnx"
        spec_siamese = (
            tf.TensorSpec((None, *input_shape), tf.float32, name="input_a"),
            tf.TensorSpec((None, *input_shape), tf.float32, name="input_b")
        )
        onnx_siamese, _ = tf2onnx.convert.from_keras(siamese_model, input_signature=spec_siamese, opset=13)
        with open(siamese_onnx_path, "wb") as f:
            f.write(onnx_siamese.SerializeToString())
        print(f"Siamese model exported to ONNX: {siamese_onnx_path}")
        results['siamese'] = siamese_onnx_path
        
    except ImportError:
        print("Warning: tf2onnx not installed. Install with: pip install tf2onnx")
    except Exception as e:
        print(f"Error exporting Siamese model to ONNX: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return results


def export_siamese_model_to_openvino(siamese_model, embedding_model, output_dir, model_name="siamese", input_shape=(128, 128, 3)):
    """
    Export Siamese model and embedding network to OpenVINO IR format
    
    Args:
        siamese_model: Full Siamese model
        embedding_model: Embedding network model
        output_dir: Base directory for OpenVINO models (e.g., 'models/openvino')
        model_name: Base name for models (default: 'siamese')
        input_shape: Input shape of the model
    
    Returns:
        Dictionary with paths to saved OpenVINO model directories
    """
    results = {}
    
    # Export full Siamese model
    siamese_dir = os.path.join(output_dir, f"{model_name}_model")
    results['siamese'] = export_model_to_openvino(siamese_model, siamese_dir, f"{model_name}_model", input_shape)
    
    # Export embedding network
    embedding_dir = os.path.join(output_dir, f"{model_name}_embedding")
    results['embedding'] = export_model_to_openvino(embedding_model, embedding_dir, f"{model_name}_embedding", input_shape)
    
    return results

