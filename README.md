   Here is a more detailed README for the Iris classifier code:

# Iris Flower Classification

Implementation of an artificial neural network model to categorize Iris flowers based on sepal/petal measurements.  

## Contents

**Code**
- `irisClassifier.py` - Contains complete implementation  

**Data**
- `IrisData.txt` - CSV file with Iris dataset 

**Models** 
- Saved ANN models after training

## Functionality

**Data Loading and Preprocessing**

- Loads the Iris dataset from IrisData.txt CSV file
- Each row contains 4 numeric measurements followed by Iris species 
- Encodes species as integer labels (one-hot encoding)
- Data is randomized and split into train/validation/test sets
- Feature values are normalized to range [0-1] 

**Artificial Neural Network Model**

- Sequential ANN with fully connected layers 
- Input layer with 4 nodes for measurements
- 1 hidden layer with sigmoid activations
- Output layer with 3 nodes (Setosa/Versicolor/Virginica)  
- Softmax output activation for probability distribution  

**Training Procedure**

- Initializes weights randomly with N(0, 1)
- Learns patterns via iterative backpropagation 
- Loss function is means squared error  
- Weights updated through gradient descent optimization
- Regularization methods prevent overfitting  

**Model Evaluation** 

- Accuracy, Precision, Recall metrics on test dataset
- Confusion matrix visualizations
- Classification reports  

**Interaction**

- Accepts user input of Iris measurements
- Forward pass through trained ANN
- Predicts Iris species with confidence score

## Usage

**To run model**
```
python irisClassifier.py 
```
