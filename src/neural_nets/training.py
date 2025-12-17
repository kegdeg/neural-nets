"""
Training utilities for neural network models.
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


def create_early_stopping(monitor='val_f1_score', patience=5, min_delta=1e-4, mode='max', restore_best_weights=True):
    """
    Create early stopping callback.
    
    Args:
        monitor: Metric to monitor
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'max' or 'min'
        restore_best_weights: Whether to restore best weights
    
    Returns:
        EarlyStopping callback
    """
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        restore_best_weights=restore_best_weights
    )


def train_model(
    model,
    x_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=128,
    callbacks=None,
    verbose=1
):
    """
    Train a model with given data.
    
    Args:
        model: Keras model to train
        x_train: Training features
        y_train: Training labels (one-hot encoded)
        validation_split: Fraction of data to use for validation
        epochs: Number of training epochs
        batch_size: Batch size
        callbacks: List of callbacks
        verbose: Verbosity level
    
    Returns:
        Training history
    """
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return history

