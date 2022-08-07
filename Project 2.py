# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:46:40 2022

Project 2: Predict Employees Productivity

@author: mrob
"""
# Import necessary packages

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
#%%
#1. Load and prepare data

data_path= r"C:\Users\captc\Desktop\AI_07\TensorFlow\Datasets_Online_Download\garments_worker_productivity.csv"
df= pd.read_csv(data_path)

#Check for missing values
print(df.isna().sum())

#Fill missing values with 0
df['wip'] = df['wip'].fillna(0)

# Drop 'date'column from dataset
worker_productivity= df.drop('date', axis=1)
#%%
# Split the dataset into features and label datasets

features= worker_productivity.copy()
label= features.pop('actual_productivity')

# Check the datasets created 

print("This is Features Dataset")
print(features.head())

print("This is Label Data")
print(label.head())
#%%
# There are few columns who are in strings format rather than numerical, we need to convert them into float by using Label Encoder

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

cols= ['quarter','department', 'day']

features[cols] = features[cols].apply(LabelEncoder().fit_transform)

#%%
#To check the new dataset
print(features.head())

#%%
# Split the data into train and test dataset
TEST_SIZE= 0.2
SEED= 12345
X_train, X_test, y_train, y_test= train_test_split(features, label, test_size= TEST_SIZE, random_state= SEED)

#%%
#Create a layer for data normalization

standardizer = StandardScaler()
standardizer.fit(X_train)
X_train = standardizer.transform(X_train)
X_test = standardizer.transform(X_test)
#%%
#2. Build a Neural Network model

nIn = X_train.shape[1]
inputs = keras.Input(shape = (nIn,))

h1 = layers.Dense(128, activation ='elu')
h2 = layers.Dense(64, activation ='elu')
h3 = layers.Dense(32, activation ='elu')

output_layer = layers.Dense(1)

# I'm using the Functional API to link the layers together
x = h1(inputs)
x = h2(x)
x = h3(x)

outputs = output_layer(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
#%%
# Compile the model

model.compile(optimizer= 'adam', loss= 'mse', metrics= ['mae'])
#%%
# 3. Train the model

BATCH_SIZE = 32
EPOCHS = 30
history = model.fit(X_train, y_train, validation_data= 
(X_test, y_test), batch_size= BATCH_SIZE,epochs = EPOCHS)
#%%
import matplotlib.pyplot as plt

predictions = np.squeeze(model.predict(X_test))
labels = np.squeeze(y_test)
plt.plot(predictions,labels,".")
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.title("Graph of Predictions vs Labels with Test Data")

plt.show()
#%%

training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['mae']
val_acc = history.history['val_mae']
epochs_x_axis = history.epoch

plt.plot(epochs_x_axis,training_loss,label='Training Loss')
plt.plot(epochs_x_axis,val_loss,label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.legend()
plt.figure()

plt.plot(epochs_x_axis,training_acc,label='Training MAE')
plt.plot(epochs_x_axis,val_acc,label='Validation Metrics')
plt.title("Training vs Validation Metrics")
plt.legend()
plt.figure()

plt.show()














