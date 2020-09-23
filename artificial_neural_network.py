#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

#print(tf.__version__)

#Data Preprocessing
dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#print(x)
#print(y)

#Encoding categorical data
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

#After gender encoded
print(x)

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])],
                       remainder="passthrough")
x = np.array(ct.fit_transform(x))

print(x)

#Split the dataset into Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

#Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Initializing the ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

#Hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

#Output layer
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

#Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Training the ANN on the Training set
ann.fit(x_train, y_train, batch_size=32, epochs=100)

print(
    ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]
                              ])) > 0.5)

#Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(
    np.concatenate(
        (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

#Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
