# -*- coding: utf-8 -*-
from __future__ import print_function
import pdb
import numpy as np
import pandas as pd
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
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
# Classification example
# fix random seed for reproducibility
seed = 1617
np.random.seed(seed)

# Load data set
DATA_DIR = "/Users/jaime/vallecas/data/scc"
SCC_FILE = 'sccplus.csv'
SCC_FILE = os.path.join(DATA_DIR, SCC_FILE)
df = pd.read_csv(SCC_FILE, usecols=["ID", "Complaints_groups", "Diagnostic", "Age", "Years_Education", "Sex"], header=0)
#df.drop(['mmse', 'faq2','FCSRTrecinm2','FCSRTrecdif2','FLUISEM2', 'RELOJ2', 'CLANUM2', 'GDS2','STAI1','DX1','DX1def','APOE3niv','APOE2niv','GFARMACOS','GFARMACOSSINSUPL','HIPOTENSOR','ANTIDEPRESIVO'])
# convert Pandas into ndarray
dataset = df.values
# Split data set into Input features and output
X = dataset[:,0:5].astype(int) #float
Y = dataset[:,-1]

#if the out put vartiable is string we can encode it int
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
#enconded_Y == Y

# NN parameters
INPUT_DIM = X.shape[1]
HIDDEN_LAYER = INPUT_DIM
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 1 # outputs

N_HIDDEN = 128
OPTIMIZER = SGD()
VALIDATION_SPLIT = 0.2 # how much train is reserved for validation
OPTIMIZER = Adam()
OPTIMIZER = 'adam'
OPTIMIZER = RMSprop() # optimizer
KERNEL_INITIALIZER = 'normal'
def baseline_NN():
	# build model
	model = Sequential()
	model.add(Dense(HIDDEN_LAYER, input_dim=INPUT_DIM, kernel_initializer=KERNEL_INITIALIZER, activation='relu'))
	model.add(Dense(NB_CLASSES, kernel_initializer=KERNEL_INITIALIZER, activation='sigmoid'))
	#Compile model
	model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['mae', 'acc'])
	return model

def deep_NN():
	# build model more hidden layer neurons
	model = Sequential()
	model.add(Dense(HIDDEN_LAYER, input_dim=INPUT_DIM, kernel_initializer=KERNEL_INITIALIZER, activation='relu'))
	model.add(Dense(int(HIDDEN_LAYER/2), kernel_initializer=KERNEL_INITIALIZER, activation='relu'))
	model.add(Dense(NB_CLASSES, kernel_initializer=KERNEL_INITIALIZER, activation='sigmoid'))
	#Compile model
	model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['acc'])
	return model
  
# Evaluate model with standarized dataset using StratifiedKFold
print("Running Keras Classification with StratifiedKFold,HIDDEN_LAYER:{},KERNEL_INIT:{}, OPTIMIZER:{}, NB_EPOCH:{},BATCH_SIZE:{}".format(HIDDEN_LAYER,KERNEL_INITIALIZER,OPTIMIZER,NB_EPOCH,BATCH_SIZE))
# standarize the dataset
estimators = []
estimators.append(('standarize', StandardScaler()))
estimators.append((('mlp', KerasClassifier(build_fn=deep_NN, epochs=NB_EPOCH, batch_size=BATCH_SIZE, verbose=10))))
pipeline = Pipeline(estimators)
# estimator without initial processing of dataset
#estimator = KerasClassifier(build_fn=baseline_NN,nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Evaluate model with standarized dataset using Split 70-30
print("Running Keras Classification with Split 70-30,HIDDEN_LAYER:{},KERNEL_INIT:{}, OPTIMIZER:{}, NB_EPOCH:{},BATCH_SIZE:{}".format(HIDDEN_LAYER,KERNEL_INITIALIZER,OPTIMIZER,NB_EPOCH,BATCH_SIZE))
# scale the data
scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)
# store the meand and std to be used after for porediction
Xmeans = scaler.mean_
Xstds = scaler.scale_
# split training data into 70 training and 30 testing
train_size = int(0.7*X.shape[0])
Xtrain, Xtest, ytrain, ytest = X[0:train_size], X[train_size:],encoded_Y[0:train_size], encoded_Y[train_size:] 
#pdb.set_trace()
# define the network, a 2 layer dense netweork takes the 12 features and outputs ascaled prediction
# the hidden layer has 8 neurons, initialization, loss function (mse) and optimizer (adam)
readings = Input(shape=(X.shape[1],))
x = Dense(X.shape[1], activation="relu", kernel_initializer="glorot_uniform")(readings)
sccplus= Dense(1, kernel_initializer="glorot_uniform")(x)
model = Model(inputs=[readings], output=[sccplus])
model.compile(loss="binary_crossentropy", optimizer="adam")
# train the model with EPOCS and BATCH_SIZE
history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=0.2)
# check our model predictions
ytest_ = model.predict(Xtest).flatten()

for i in range(ytest.shape[0]):
 	label = (ytest[i]*Xstds[3]) + Xmeans[3]
 	prediction = (ytest_[i]*Xstds[3]) + Xmeans[3]
 	print("Target SCC expected:{:.3f}, predicted:{:.3f}".format(label, prediction))
 # Plot the results, actual values against the predictions
plt.plot(np.arange(ytest.shape[0]),(ytest * Xstds[3]) / Xmeans[3], color = "b", label="actual" )
plt.plot(np.arange(ytest_.shape[0]),(ytest_ * Xstds[3]) / Xmeans[3], color = "r", label="predicted" )
plt.xlabel("subjects")
plt.ylabel("SCC+ ")
plt.legend(loc="best")
plt.show()














# # training
# NB_EPOCH = 20
# BATCH_SIZE = 128
# VERBOSE = 1
# NB_CLASSES = 10 # outputs
# OPTIMIZER = SGD()
# N_HIDDEN = 128
# VALIDATION_SPLIT = 0.2 # how much train is reserved for validation
# OPTIMIZER = RMSprop() # optimizer
# OPTIMIZER = Adam()
# # data suffled between training and testing
# remove "bad" columns
# del aqdf["mmse"]
# del aqdf["Time"]
# del aqdf["Unnamed: 15"]
# del aqdf["Unnamed: 16"]
# # fill NaNs with the mean value
# aqdf = aqdf.fillna(aqdf.mean())
# Xorig = aqdf.as_matrix()
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# # X_train is 60000 rows of 28x28 vales, reshaped into 60000xRESHAPED
# RESHAPED = 784 # 28x28 
# X_train = X_train.reshape(60000, RESHAPED)
# X_test = X_test.reshape(10000, RESHAPED)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')

# # normalize X_test between 0 and 1
# X_train /= 255
# X_test /= 255
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')
# # convert class vectors to binary class matrices
# # y_train = 5,8,9,1 ..
# # Y_train 60000x 10 columns 1 for the digit, eg Y_train[0,5]=1
# Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
# Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
# # final layer is the softmax a generalization of the sigmoid
# model = Sequential()
# model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))

# model.add(Activation('softmax'))
# model.summary()
# # compile the model
# model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
# # if the compilation is successful we can trin it with the fit() function
# history = model.fit(X_train, Y_train, batch_size=BATH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
# score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
# print("Test score:", score[0])
# print("Test accuracy:", score[1])

# # Add hidden layers (not connected from input nor output)
# model = Sequential()
# model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,))) # after the input later a hidden layer
# model.add(Activation('relu'))
# model.add(Dense(N_HIDDEN)) #after the first hidden layer
# model.add(Activation('relu'))
# model.add(Dense(NB_CLASSES)) # output of 10 neurons
# model.add(Activation('softmax'))

# # regularizer
# model.add(Dense(64, input_dim = 64, kernel_regularizer=regularizers.l2(0.01)))
# # calculate predictions
# predictions = model.predict(X)
# model.evaluate() #compute loss values
# model.predict_classes() # compute category outputs 
# model.predict_proba() # compute class probabilities





# # Regression networks example

# DATA_DIR = "data"
# AIRQUALITY_FILE = os.path.join(DATA_DIR, "AirQualityUCI.csv")
# aqdf = pd.read_csv(AIRQUALITY_FILE, sep=";", decimal=",", header=0)
# # remove columsn. data, time and last two columns
# del aqdf["Date"]
# del aqdf["Time"]
# del aqdf["Unnamed: 15"]
# del aqdf["Unnamed: 16"]
# # fill NaNs with the mean value
# aqdf = aqdf.fillna(aqdf.mean())
# Xorig = aqdf.as_matrix()

# # scale the data
# scaler = StandardScaler()
# Xscaled = scaler.fit_transform(Xorig)
# # store the meand and std to be used after for porediction
# Xmeans = scaler.mean_
# Xstds = scaler.scale_
# # the target variable is the fourthn columun
# y= Xscaled[:,3]
# # delete the target variable from the input (training data)
# X = np.delete(Xscaled, 3, axis=1)
# # split training data inot 70 training and 30 testing
# train_size = int(0.7*X.shape[0])
# Xtrain, Xtest, ytrain, ytest = X[0:train_size], X[train_size:],y[0:train_size], y[train_size:] 
# # define the network, a 2 layer dense netweork takes the 12 features and outputs ascaled prediction
# # the hidden layer has 8 neurons, initialization, loss function (mse) and optimizer (adam)
# readings = Input(shape=(12,))
# x = Dense(8, activation="relu", kernel_initializer="glorot_uniform")(readings)
# benzene = Dense(1, kernel_initializer="glorot_uniform")(x)
# model = Model(inputs=[readings], output=[benzene])
# model.compile(loss="mse", optimizer="adam")
# # train the model with EPOCS and BATCH_SIZE
# NUM_EPOCHS = 20
# BATCH_SIZE = 10
# history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2)
# # check our model predictions
# ytest_ = model.predict(Xtest).flatten()
# for i in range(ytest.shape[0]):
# 	label = (ytest[i]*Xstds[3]) + Xmeans[3]
# 	prediction = (ytest_[i]*Xstds[3]) + Xmeans[3]
# 	print("Target benzene expected:{:.3f}, predicted:{:.3f}".format( label, prediction))
# # Plot the results, actual values against the predictions
# plt.plot(np.arange(ytest.shape[0]),(ytest * Xstds[3]) / Xmeans[3], color = "b", label="actual" )
# plt.plot(np.arange(ytest_.shape[0]),(ytest_ * Xstds[3]) / Xmeans[3], color = "r", label="predicted" )
# plt.xlabel("time")
# plt.ylabel("Benzene conc")
# plt.legend(loc="best")
# plt.show()





