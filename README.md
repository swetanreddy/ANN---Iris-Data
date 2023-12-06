# Iris Flower Classification

This project classifies Iris flowers into three species - Setosa, Versicolor, and Virginica.

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- pandas 
- scikit-learn
- tensorflow
- numpy

## Usage

To train the model and evaluate on test set:

```bash
python iris_classifier.py
```

The script loads the Iris data, trains a neural network model, and prints out the test accuracy.

To make predictions:

1. Run the script
2. Enter Sepal Length, Sepal Width, Petal Length, Petal Width separated by commas when prompted
3. The prediction result will be printed

Example:
```
Enter data: 5.1,3.5,1.4,0.2 

Predicted Iris Class: Iris-setosa
```

## Model Architecture

The model is a sequential neural network with two dense layers.

The input layer has 4 input features, the hidden layer has 16 nodes, and the output layer has 3 nodes for the Iris classes.

ReLU activation is used on the hidden layer, and Softmax activation is used on the output layer for probability predictions.

Categorical cross-entropy loss and Adam optimizer are used.

## Future Work

Some ideas for improving the model:

- Try different model architectures 
- Tune hyperparameters like layers sizes, learning rate
- Experiment with different feature scaling methods
- Use k-Fold cross validation
