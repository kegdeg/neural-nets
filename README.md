# Neural Networks for Reuters News Classification

A comprehensive toolkit for training and evaluating neural network models on the Reuters news classification dataset. This project provides a clean, modular codebase with a CLI tool for easy experimentation.

## Features

- **Multiple Model Architectures**: Baseline, small, bigger, and regularized models
- **Hyperparameter Tuning**: Automated hyperparameter optimization using Keras Tuner
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1 score, and overfitting analysis
- **Visualization**: Training metrics plotting
- **CLI Tool**: Easy-to-use command-line interface for training and evaluation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gitgatgit/neural-nets.git
cd neural-nets
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### CLI Tool

The main CLI tool provides three commands:

#### Train a Single Model

Train a specific model architecture:

```bash
python main.py train --model small --epochs 20 --batch-size 128
```

Available models:
- `baseline`: Simple single-layer model
- `small`: One hidden layer (128 units)
- `bigger`: Two hidden layers (128, 64 units)
- `bigger2`: Two hidden layers (256, 128 units)
- `regularised`: L2 regularized model
- `regularised2`: L2 regularized model with dropout

Options:
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 128)
- `--validation-split`: Validation split ratio (default: 0.2)
- `--early-stopping`: Enable early stopping
- `--patience`: Early stopping patience (default: 5)
- `--plot`: Plot training metrics
- `--save-plot`: Path to save plot
- `--save-model`: Save trained model (use "auto" for automatic path)

Example with all options:
```bash
python main.py train --model small --epochs 30 --batch-size 64 \
    --early-stopping --patience 5 --plot --save-model auto
```

#### Train All Models and Compare

Train all available models and generate a comparison table:

```bash
python main.py train-all --epochs 20 --save-models
```

This will:
- Train all model architectures
- Evaluate each on the test set
- Generate a comparison table
- Optionally save all models

#### Hyperparameter Tuning

Perform automated hyperparameter optimization:

```bash
python main.py tune --max-epochs 50 --tuning-dir tuning_results
```

Options:
- `--max-epochs`: Maximum epochs per trial (default: 50)
- `--tuning-dir`: Directory for tuning results (default: "tuning")
- `--project-name`: Project name for tuning (default: "reuters_tuning")
- `--factor`: Hyperband reduction factor (default: 3)
- `--plot`: Plot training metrics for best model
- `--save-model`: Save best model

### Python API

You can also use the modules directly in Python:

```python
from neural_nets.data_processing import load_reuters_data
from neural_nets.models import build_small_model
from neural_nets.training import train_model
from neural_nets.evaluation import print_metrics, plot_metrics

# Load data
x_train, _, x_test, _, y_train, y_test = load_reuters_data()

# Build and train model
model = build_small_model()
history = train_model(model, x_train, y_train, epochs=20)

# Evaluate
test_metrics = model.evaluate(x_test, y_test, verbose=0)
print_metrics("Small Model", test_metrics)

# Plot metrics
plot_metrics(history, "Small Model")
```

## Project Structure

```
neural-nets/
├── main.py                 # CLI tool entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── neural_nets.ipynb      # Original Jupyter notebook
├── src/
│   └── neural_nets/
│       ├── __init__.py
│       ├── data_processing.py    # Data loading and preprocessing
│       ├── models.py             # Model architectures
│       ├── training.py           # Training utilities
│       ├── evaluation.py         # Evaluation and visualization
│       └── hyperparameter_tuning.py  # Hyperparameter tuning
├── models/                # Saved model files (gitignored)
├── data/                  # Data files (gitignored)
├── outputs/               # Output files (gitignored)
└── tuning/                # Hyperparameter tuning results (gitignored)
```

## Model Architectures

### Baseline Model
- Single dense layer with softmax activation
- SGD optimizer with learning rate 0.01

### Small Model
- One hidden layer (128 units, ReLU)
- Output layer (46 classes, softmax)
- Adam optimizer

### Bigger Models
- Multiple hidden layers with ReLU activation
- Adam optimizer
- Variants with different layer sizes

### Regularized Models
- L2 regularization
- Optional dropout layers
- Helps reduce overfitting

## Evaluation Metrics

All models are evaluated using:
- **Loss**: Categorical crossentropy
- **Accuracy**: Classification accuracy
- **Precision**: Weighted precision
- **Recall**: Weighted recall
- **F1 Score**: Weighted F1 score

The toolkit also provides overfitting analysis by comparing training and validation metrics.

## Hyperparameter Tuning

The hyperparameter tuning uses Keras Tuner's Hyperband algorithm to search for optimal:
- Number of layers (1-3)
- Layer sizes
- Activation functions (ReLU, tanh, ELU)
- Learning rate
- L1 and L2 regularization strengths
- Dropout rates

## Results

After training, you'll see:
- Training history with metrics per epoch
- Test set evaluation metrics
- Overfitting analysis
- Optional visualization plots
- Model comparison tables (when training multiple models)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Reuters dataset provided by TensorFlow/Keras
- Built with TensorFlow and Keras

