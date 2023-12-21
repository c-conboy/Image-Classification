# CIFAR-100 Image Classification

This repository contains a image classification model, and Python scripts for training and evaluating said model on the CIFAR-100 dataset for image classification.

## Usage

### Prerequisites

- Python 3
- PyTorch

###Training 

```
python train.py -s ./experiments -l ./frontendsaved.pth -lr 1e-5 -e 50 -b 1000 -cuda Y -p loss.png
```

**Arguments**

-s: Directory to save the trained model.

-l: Path to the pre-trained encoder weights file.

-lr: Learning rate for optimization (default: 1e-5).

-e: Number of training epochs (default: 50).

-b: Batch size (default: 1000).

-cuda: Use CUDA for faster processing (optional, 'Y' or 'N').

-p: Path to save the training loss plot.



Great! If your repository contains both training and evaluation scripts, you can create a comprehensive README that covers both aspects. Here's a combined README template:

markdown
Copy code
# CIFAR-100 Image Classification with CJNet

This repository contains Python scripts for training and evaluating the CJNet model on the CIFAR-100 dataset for image classification.

## Usage

### Prerequisites

- Python 3
- PyTorch

### Installation

```bash
pip install -r requirements.txt
Training
Running the Training Script
bash
Copy code
python train.py -s ./experiments -l ./frontendsaved.pth -lr 1e-5 -e 50 -b 1000 -cuda Y -p loss.png
Arguments
-s: Directory to save the trained model.
-l: Path to the pre-trained encoder weights file.
-lr: Learning rate for optimization (default: 1e-5).
-e: Number of training epochs (default: 50).
-b: Batch size (default: 1000).
-cuda: Use CUDA for faster processing (optional, 'Y' or 'N').
-p: Path to save the training loss plot.
Output
The script will save the trained frontend weights in the specified save directory and generate a training loss plot.

### Evaluation

Running the Evaluation Script

```
python evaluate.py -frontend_file path/to/frontend_weights.pth -cuda Y
```

**Arguments**'

-frontend_file: Path to the pre-trained frontend weights file.

-cuda: Use CUDA for faster processing (optional, 'Y' or 'N').

### Output

The script will print the top-5 and top-1 error rates based on the evaluation of the CJNet model on the CIFAR-100 test set.