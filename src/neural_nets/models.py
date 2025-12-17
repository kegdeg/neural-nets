"""
Model architectures for Reuters news classification.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l1_l2, l2


def get_metrics():
    """
    Get standard metrics for model evaluation.
    
    Returns:
        List of metrics
    """
    return [
        'accuracy',
        'precision',
        'recall',
        tf.keras.metrics.F1Score(average='weighted', threshold=None, name='f1_score')
    ]


def build_baseline_model(input_shape=(10000,), num_classes=46):
    """
    Build a simple baseline model with single dense layer.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Dense(num_classes, activation='softmax', input_shape=input_shape)
    ])
    
    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=get_metrics()
    )
    
    return model


def build_small_model(input_shape=(10000,), num_classes=46):
    """
    Build a small model with one hidden layer.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=get_metrics()
    )
    
    return model


def build_bigger_model(input_shape=(10000,), num_classes=46):
    """
    Build a bigger model with two hidden layers.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=get_metrics()
    )
    
    return model


def build_bigger_model2(input_shape=(10000,), num_classes=46):
    """
    Build a bigger model variant with larger hidden layers.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=get_metrics()
    )
    
    return model


def build_regularised_model(input_shape=(10000,), num_classes=46, l2_reg=0.001):
    """
    Build a regularised model with L2 regularization.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
        l2_reg: L2 regularization strength
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(l2_reg), input_shape=input_shape),
        Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=get_metrics()
    )
    
    return model


def build_regularised_model2(input_shape=(10000,), num_classes=46, l2_reg=0.001, dropout_rate=0.5):
    """
    Build a regularised model with L2 regularization and dropout.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
        l2_reg: L2 regularization strength
        dropout_rate: Dropout rate
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(l2_reg), input_shape=input_shape),
        Dropout(dropout_rate),
        Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=get_metrics()
    )
    
    return model


def build_custom_model(
    input_shape=(10000,),
    num_classes=46,
    layers=[128, 64],
    activation='relu',
    l1_reg=0.0,
    l2_reg=0.0,
    dropout_rate=0.0,
    learning_rate=0.001,
    optimizer='adam'
):
    """
    Build a custom model with specified architecture.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
        layers: List of hidden layer sizes
        activation: Activation function
        l1_reg: L1 regularization strength
        l2_reg: L2 regularization strength
        dropout_rate: Dropout rate (0.0 means no dropout)
        learning_rate: Learning rate
        optimizer: Optimizer name ('adam' or 'sgd')
    
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # Add regularizer if specified
    regularizer = None
    if l1_reg > 0 or l2_reg > 0:
        regularizer = l1_l2(l1=l1_reg, l2=l2_reg)
    
    # Add first layer
    model.add(Dense(
        layers[0],
        activation=activation,
        input_shape=input_shape,
        kernel_regularizer=regularizer
    ))
    
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Add additional hidden layers
    for layer_size in layers[1:]:
        model.add(Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=regularizer
        ))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Choose optimizer
    if optimizer.lower() == 'adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        opt = SGD(learning_rate=learning_rate)
    
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=get_metrics()
    )
    
    return model

