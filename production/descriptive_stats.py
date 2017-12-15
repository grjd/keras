# -*- coding: utf-8 -*-
from __future__ import print_function
import pdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from random import randint
import seaborn as sns
#https://www.datacamp.com/community/tutorials/deep-learning-python
DB_WEB = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
wine_w = pd.read_csv(DB_WEB, sep=';')
# to add features
# wine_w['me_gusta'] = 11
# to append 2 csvs
# wines = red.append(white, ignore_index=True) do not duplicate labels
print("CSV loaded. The feature names are: ", wine_w.keys())
print(wine_w.keys())
print(wine_w.info())
#print(wine_w.head())
#print(wine_w.tail())
print(wine_w.describe())
# check if some element is null
pd.insull(wihite_w)

print("The column names are: ", wine_w.keys())
# Visualizing data, plot the hisogram of 2 features
nb_of_bins = 15
fig, ax = plt.subplots(1, 2)
# Histogram: to calculate the histogram wiithout plotting
# print(np.histogram(wine_w.alcohol, bins=np.arange(7,16)))
print("Plotting the histogram....")
# it is preferible to access wine_w[''] especially is the column name contains blanks
ax[0].hist(wine_w.alcohol, nb_of_bins, facecolor='red', alpha=0.5, label="wine")
ax[1].hist(wine_w.pH, nb_of_bins, facecolor='red', alpha=0.5, label="wine")
#fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("pH in % Vol")
ax[1].set_ylabel("Frequency")
fig.suptitle("Alcohol in % Vol and pH")
plt.show()

print("Plotting the scatter plot....")
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].scatter(wine_w['quality'], wine_w['total sulfur dioxide'], color="red")
ax[1].scatter(wine_w['density'], wine_w['pH'], color="blue")
ax[0].set_title("Wine")
ax[1].set_title("wine too")
ax[0].set_xlabel("Quality")
ax[1].set_xlabel("Density")
ax[0].set_ylabel("SO2")
ax[1].set_ylabel("pH")
ax[0].set_xlim([0,10])
ax[1].set_xlim([0,10])
ax[0].set_ylim([min(wine_w['total sulfur dioxide']),max(wine_w['total sulfur dioxide'])])
ax[1].set_ylim([0,14])
fig.subplots_adjust(wspace=0.5)
fig.suptitle("Scatter plots")
plt.show()

# Data preprocessing
x=[randint(0,1) for p in range(0,4898)]
wine_w['me_gusta'] = x

# plot sns map with correlation matrix
fig, ax = plt.subplots(1,1)
corr = wine_w.corr()
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values)
ax.set_xlabel("Features")
ax.set_ylabel("Features")
ax.set_title('Scatter plot features')
plt.show()

# Train and test sets
# Imbalanced data typically refers to a problem with classification problems where the classes 
# are not represented equally, otherwise you are favouring one class versus the other
# Import the train_test_split from sklearn.
# Specify the input data, note that wine_w is type pandas not ndarray
# X and y are ndarray
X = wine_w.ix[:,0:-2] # all columns except last one
#ravel return a contiguous flattened array. eg [[1,2], [3,4]] returns [1,2,3,4]
y = np.ravel(wines_w['me gusta']) 
# Split the data up in train and test sets, random_state is the seed used by the random number generator
#  default test_size value is set to 0.25. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Standarize : Standardization is a way to deal with these values that lie so far apart
# import the StandardScaler module from sklearn.preprocessing 
# Define the scaler 
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set, is now ndarray from the original pandas
X_test = scaler.transform(X_test)

# Model data
# the question we want to solve is if we can classify (separate based on the atrget variables) based on the features
# A  MLP tends to perorm well in binary classification
#  Dense layers implement : output = activation(dot(input, kernel) + bias)
# Dense input size + Dense size 8 hidden + Dense output 1
# Import `Sequential` from `keras.models` and  Import `Dense` from `keras.layers`
# Intialize the constructor, 2 decisions: nb lo layers and nb of hidden units
# when you don’t have that much training data available, you should prefer to use a a small network with very few hidden layers
model = Sequential()
# Add an input layer, initializer by default
model.add(Dense(12, activation='relu', input_shape=(12,)))
# add hidden layer
model.add(Dense(8, activation='relu'))
#add output layer
model.add(Dense(1, activation='sigmoid'))
model.output_shape
model.summary()
model.get_config()
model.get_weights()

# Compile and fit,  optimizers Stochastic Gradient Descent (SGD), ADAM and RMSprop
#  for a regression problem, you’ll usually use the Mean Squared Error (MSE)
#  used binary_crossentropy for the binary classification 
# multi-class classification, you’ll make use of categorical_crossentropy
# train the model for 20 epochs or iterations over all the samples in X_train and y_train,
# An epoch is a single pass through the entire training set, followed by testing of the verification set
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train,epochs=20, batch_size=128, show_accuracy=True, verbose=2,validation_data=(X_test, y_test))

# Predict values or put your model to use.Make predictions for the labels of the test set with it.
# Just use predict() and pass the test set to it to predict the labels for the data. 
y_pred = model.predict(X_test)
# checking how y_pred and y_test compare
print(" quick check feature 5 y_pred and y_test", y_pred[:5], y_test[:5])

# Evaluate model,  make predictions on data that your model hadn’t seen yet
score = model.evaluate(X_test, y_test,batch_size=128, verbose=1)
print(" Loss and Accuracy", score)
# try out different evaluation metrics.
# confusion matrix, which is a breakdown of predictions into a table showing 
# correct predictions and the types of incorrect predictions made. Ideally, you will only see numbers in the diagonal, 
# which means that all your predictions were correct
# Precision is a measure of a classifier’s exactness. The higher the precision, the more accurate the classifier.
# Recall is a measure of a classifier’s completeness. The higher the recall, the more cases the classifier covers.
# The F1 Score or F-score is a weighted average of precision and recall.
# The Kappa or Cohen’s kappa is the classification accuracy normalized by the imbalance of the classes in the data.

# binarize the output (sigmoid returns a probability)
binary = True
if binary:
    esti = model.predict(X_test)
    esti = esti>0.5
    esti_y_pred = est.astype(int)
    acc = np.mean(esti_y_pred[:,0] == y_test)
    print(" quick check feature 5 y_pred and y_test", esti_y_pred[:5], y_test[:5])
    print ('my acc:', acc)
confusion_matrix(y_test, esti_y_pred)
precision_score(y_test, esti_y_pred)
recall_score(y_test, esti_y_pred)
f1_score(y_test,esti_y_pred)
cohen_kappa_score(y_test, esti_y_pred)

# Predicting in a regression network, target is a continuous variable
# Isolate target labels
#y = wines.quality #wine is pandas
# Isolate data
X = wines.drop('quality', axis=1)
# Scale the data with `StandardScaler`
X = StandardScaler().fit_transform(X)
#Your network ends with a single unit Dense(1), and doesn’t include an activation. 
#This is a typical setup for scalar regression, where you are trying to predict a single continuous value).
# kfold
