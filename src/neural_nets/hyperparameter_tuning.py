"""
Hyperparameter tuning utilities using Keras Tuner.
"""

import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1_l2


def get_metrics():
    """Get standard metrics for model evaluation."""
    return [
        'accuracy',
        'precision',
        'recall',
        tf.keras.metrics.F1Score(average='weighted', threshold=None, name='f1_score')
    ]


def build_hypermodel(hp):
    """
    Build a hypermodel for hyperparameter tuning.
    
    Args:
        hp: HyperParameters object from Keras Tuner
    
    Returns:
        Compiled Keras model
    """
    model = Sequential()

    # Regularisation hyperparameters
    l1_reg = hp.Float('l1_reg', min_value=1e-6, max_value=1e-3, sampling='log')
    l2_reg = hp.Float('l2_reg', min_value=1e-6, max_value=1e-3, sampling='log')
    use_dropout = hp.Boolean('use_dropout')
    dropout = hp.Float('dropout', 0.3, 0.5, step=0.1) if use_dropout else 0.0

    # Architecture hyperparameters
    n_layers = hp.Int('n_layers', 1, 3)
    learning_rate = hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])
    activation = hp.Choice('activation', ['relu', 'tanh', 'elu'])
    first_layer = hp.Choice('first_layer', [128, 64, 46])

    # First layer with regularisation
    model.add(Dense(
        first_layer,
        activation=activation,
        input_shape=(10000,),
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    ))
    if use_dropout:
        model.add(Dropout(dropout))

    # Additional layers with regularisation
    for i in range(n_layers - 1):
        layer_size = hp.Choice(f'layer_{i}', [64, 46])
        model.add(Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
        ))
        if use_dropout:
            model.add(Dropout(dropout))

    model.add(Dense(46, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=get_metrics()
    )

    return model


def create_tuner(
    directory='tuning',
    project_name='reuters_tuning',
    max_epochs=50,
    factor=3
):
    """
    Create a Hyperband tuner for hyperparameter optimization.
    
    Args:
        directory: Directory to save tuning results
        project_name: Project name for tuning
        max_epochs: Maximum epochs per trial
        factor: Reduction factor for Hyperband
    
    Returns:
        Keras Tuner Hyperband object
    """
    tuner = kt.Hyperband(
        build_hypermodel,
        objective=kt.Objective('val_f1_score', direction='max'),
        max_epochs=max_epochs,
        directory=directory,
        project_name=project_name,
        factor=factor
    )
    
    return tuner

