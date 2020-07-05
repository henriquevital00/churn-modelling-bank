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

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
x = np.array(ct.fit_transform(x))

print(x)

#Split the dataset into Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

