from __future__ import print_function
import numpy as numpy
from keras import regularizers
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
import pandas as pandas
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.random.seed(1617)

# training
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # outputs
OPTIMIZER = SGD()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 # how much train is reserved for validation
OPTIMIZER = RMSprop() # optimizer
OPTIMIZER = Adam()
# data suffled between training and testing

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train is 60000 rows of 28x28 vales, reshaped into 60000xRESHAPED
RESHAPED = 784 # 28x28 
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize X_test between 0 and 1
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
# y_train = 5,8,9,1 ..
# Y_train 60000x 10 columns 1 for the digit, eg Y_train[0,5]=1
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
# final layer is the softmax a generalization of the sigmoid
model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))

model.add(Activation('softmax'))
model.summary()
# compile the model
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
# if the compilation is successful we can trin it with the fit() function
history = model.fit(X_train, Y_train, batch_size=BATH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])

# Add hidden layers (not connected from input nor output)
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,))) # after the input later a hidden layer
model.add(Activation('relu'))
model.add(Dense(N_HIDDEN)) #after the first hidden layer
model.add(Activation('relu'))
model.add(Dense(NB_CLASSES)) # output of 10 neurons
model.add(Activation('softmax'))

# regularizer
model.add(Dense(64, input_dim = 64, kernel_regularizer=regularizers.l2(0.01)))
# calculate predictions
predictions = model.predict(X)
model.evaluate() #compute loss values
model.predict_classes() # compute category outputs 
model.predict_proba() # compute class probabilities


# Regression networks example

DATA_DIR = "data"
AIRQUALITY_FILE = os.path.join(DATA_DIR, "AirQualityUCI.csv")
aqdf = pd.read_csv(AIRQUALITY_FILE, sep=";", decimal=",", header=0)
# remove columsn. data, time and last two columns
del aqdf["Date"]
del aqdf["Time"]
del aqdf["Unnamed: 15"]
del aqdf["Unnamed: 16"]
# fill NaNs with the mean value
aqdf = aqdf.fillna(aqdf.mean())
Xorig = aqdf.as_matrix()

# scale the data
scaler = StandardScaler()
Xscaled = scaler.fit_transform(Xorig)
# store the meand and std to be used after for porediction
Xmeans = scaler.mean_
Xstds = scaler.scale_
# the target variable is the fourthn columun
y= Xscaled[:,3]
# delete the target variable from the input (training data)
X = np.delete(Xscaled, 3, axis=1)
# split training data inot 70 training and 30 testing
train_size = int(0.7*X.shape[0])
Xtrain, Xtest, ytrain, ytest = X[0:train_size], X[train_size:],y[0:train_size], y[train_size:] 
# define the network, a 2 layer dense netweork takes the 12 features and outputs ascaled prediction
# the hidden layer has 8 neurons, initialization, loss function (mse) and optimizer (adam)
readings = Input(shape=(12,))
x = Dense(8, activation="relu", kernel_initializer="glorot_uniform")(readings)
benzene = Dense(1, kernel_initializer="glorot_uniform")(x)
model = Model(inputs=[readings], output=[benzene])
model.compile(loss="mse", optimizer="adam")
# train the model with EPOCS and BATCH_SIZE
NUM_EPOCHS = 20
BATCH_SIZE = 10
history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2)
# check our model predictions
ytest_ = model.predict(Xtest).flatten()
for i in range(ytest.shape[0]):
	label = (ytest[i]*Xstds[3]) + Xmeans[3]
	prediction = (ytest_[i]*Xstds[3]) + Xmeans[3]
	print("Target benzene expected:{:.3f}, predicted:{:.3f}".format( label, prediction))
# Plot the results, actual values against the predictions
plt.plot(np.arange(ytest.shape[0]),(ytest * Xstds[3]) / Xmeans[3], color = "b", label="actual" )
plt.plot(np.arange(ytest_.shape[0]),(ytest_ * Xstds[3]) / Xmeans[3], color = "r", label="predicted" )
plt.xlabel("time")
plt.ylabel("Benzene conc")
plt.legend(loc="best")
plt.show()





