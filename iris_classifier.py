import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# Load the data 
file_path = 'irisData.txt'
data = pd.read_csv(file_path, header=None, 
                   names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# Encode class labels  
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
data['class'] = data['class'].map(class_mapping)

# Separate features and labels
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] 
y = data['class']

# Now ready to use data for training!

# Split the dataset  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0) 

# Feature scaling  
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# One hot encode the output
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray() 
y_test = ohe.transform(y_test.reshape(-1,1)).toarray()

# Build neural network model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, input_dim=4, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

# Compile model  
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=5)   

# Evaluate model
loss, acc = model.evaluate(X_test, y_test, verbose=0) 
print("Accuracy: %.2f%%" % (acc * 100.0))  

# Make prediction
user_data = [float(x) for x in input("Enter data: ").split(',')]  
user_data = sc.transform(np.array(user_data).reshape(1, -1))
out = model.predict(user_data) 

# print(out)
# print(le.inverse_transform([np.argmax(out)]))

# Get predicted class index  
predicted_idx = np.argmax(out)

# Map index to class name
predicted_class = {v:k for k,v in class_mapping.items()}[predicted_idx]
print(f"Predicted Iris Class: {predicted_class}")