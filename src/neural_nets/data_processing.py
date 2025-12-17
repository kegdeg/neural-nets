"""
Data processing utilities for Reuters dataset.
"""

import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical


def vectorise_sequences(sequences, dimension=10000):
    """
    Convert sequences of word indices to binary matrix representation.
    
    Args:
        sequences: List of sequences (each sequence is a list of word indices)
        dimension: Vocabulary size (default: 10000)
    
    Returns:
        Binary matrix of shape (len(sequences), dimension)
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def load_reuters_data(num_words=10000):
    """
    Load and preprocess Reuters news classification dataset.
    
    Args:
        num_words: Maximum number of words to keep (default: 10000)
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test, one_hot_train_labels, one_hot_test_labels)
    """
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=num_words)
    
    x_train = vectorise_sequences(train_data, dimension=num_words)
    x_test = vectorise_sequences(test_data, dimension=num_words)
    
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)
    
    return x_train, train_labels, x_test, test_labels, one_hot_train_labels, one_hot_test_labels

