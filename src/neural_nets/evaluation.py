"""
Evaluation and visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_metrics(history, title, save_path=None):
    """
    Plot training metrics (accuracy, precision, recall, F1 score).
    
    Args:
        history: Training history object
        title: Plot title
        save_path: Optional path to save the plot
    """
    print("Available metrics:", history.history.keys())
    epochs = range(1, len(history.epoch) + 1)
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(epochs, history.history['precision'], label='Training Precision')
    plt.plot(epochs, history.history['val_precision'], label='Validation Precision')
    plt.title(f'{title} - Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(epochs, history.history['recall'], label='Training Recall')
    plt.plot(epochs, history.history['val_recall'], label='Validation Recall')
    plt.title(f'{title} - Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(epochs, history.history['f1_score'], label='Training F1')
    plt.plot(epochs, history.history['val_f1_score'], label='Validation F1')
    plt.title(f'{title} - F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def print_metrics(name, metrics):
    """
    Print model metrics in a formatted way.
    
    Args:
        name: Model name
        metrics: List of metrics [loss, accuracy, precision, recall, f1_score]
    """
    print(f"\n{name} Model Metrics:")
    print(f"Loss: {metrics[0]:.4f}")
    print(f"Accuracy: {metrics[1]:.4f}")
    print(f"Precision: {metrics[2]:.4f}")
    print(f"Recall: {metrics[3]:.4f}")
    print(f"F1 Score: {metrics[4]:.4f}")


def analyse_overfitting(history, model_name):
    """
    Analyze overfitting by comparing training and validation metrics.
    
    Args:
        history: Training history object
        model_name: Name of the model
    """
    val_loss = history.history['val_loss']
    train_loss = history.history['loss']
    val_f1 = history.history['val_f1_score']
    train_f1 = history.history['f1_score']
    val_acc = history.history['val_accuracy']
    train_acc = history.history['accuracy']

    optimal_epoch_f1 = np.argmax(val_f1) + 1
    optimal_epoch_acc = np.argmax(val_acc) + 1

    print(f"\nOverfitting Analysis for {model_name}:")
    print("-" * 50)
    print(f"Optimal epoch (F1): {optimal_epoch_f1}")
    print(f"Optimal epoch (Accuracy): {optimal_epoch_acc}")
    print(f"Best validation F1: {max(val_f1):.4f}")
    print(f"Best validation Accuracy: {max(val_acc):.4f}")
    print(f"Final F1 gap (train-val): {train_f1[-1] - val_f1[-1]:.4f}")
    print(f"Final Accuracy gap (train-val): {train_acc[-1] - val_acc[-1]:.4f}")


def print_model_comparison_table(results):
    """
    Print a formatted comparison table of multiple models.
    
    Args:
        results: Dictionary mapping model names to metrics lists
    """
    headers = ['Model', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    col_widths = []

    # Calculate column widths
    col_widths.append(max(len(h) for h in [headers[0]] + list(results.keys())))
    for i in range(1, len(headers)):
        col_widths.append(max(len(headers[i]),
                             max(len(f"{metric:.4f}") for model_metrics in results.values() for metric in model_metrics)))

    # Print header
    header = ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))
    print('\n' + header)
    print('-' * len(header))

    # Print rows
    for model_name, metrics in results.items():
        # Assuming metrics are in order: loss, accuracy, precision, recall, f1
        values = [f"{v:.4f}" for v in metrics]
        row = [model_name] + values
        formatted_row = ' | '.join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(formatted_row)

