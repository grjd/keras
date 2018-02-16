# -*- coding: utf-8 -*-
from __future__ import print_function
import pdb
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from random import randint
import seaborn as sns

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical




(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Preprocessing data
train_images = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[1]))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[1]))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)	
# define network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
# compile the network
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# train the network
network.fit(train_images, train_labels, epochs=5, batch_size=128)
# check how thew network predicts
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("test_loss:",test_loss, ", test_acc:",test_acc)


# display a digit
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# train_images[4].shape = (28,28) .ndim =2
print(train_images.shape)
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# IMDB 
from keras.datasets import imdb
# discard rare words only use the 10k more common. reviews are sequence of integers, each integer stands for a word in the dictionary
# the review are encded as sequence of words
# the labels are just 0,1
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# max is 999 cause have only 10000 words
max([max(sequence) for sequence in train_data])
# decode the number that corresponds with the word
ixreview = 0
word_index = imdb.get_word_index()  
reverse_word_index = dict( [(value, key) for (key, value) in word_index.items()])   
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[ixreview]]) 
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# validation, set apart 10,000
# In order to monitor during training the accuracy of the model on data it has never seen before, you’ll create a validation set by setting apart 10,000 samples from the original training data.
# a model that performs better on the training data isn’t necessarily a model that will do better on data it has never seen before
# this is to get the number of epochs right

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


model.predict(x_test)

# pg 218. Model Evaluation. 
# Hold out validationThe pb is that n is small validation and test sets may be too small
# you can identify this proble: if different random shuffling rounds of the data before splitting
# end up yielding very different measures of model performance, then you’re having this issue
print "Hold out validation"
num_validation_samples = 10000
# Shuffle data
np.random.shuffle(data)
# define validation set
validation_data = data[:num_validation_samples]   
data = data[num_validation_samples:]
# define training set
training_data = data[:]   
# trains model on training data and evaluates on validation
model = get_model()  
model.train(training_data)  
validation_score = model.evaluate(validation_data)  
# Tune your model, retrain it, evaluate on and on
# once  tuned the hyperparameters from scratch on all non data available
model = get_model()
model.train(np.concatenate([training_data,validation_data]))   
test_score = model.evaluate(test_data)  

# k-fold validation avoids the rpevious problem.  split your data into K partitions of equal size
# and for each partition train model on remaining K-1 partitions and evaluate on i
# Your final score is then the averages of the K scores obtained.
# eg 3 folds: [val, tra, tra], [t,v,t],[t,t,v], each folds return a validations core, ewe do the avg
print "k-fold validation, Good if you have relatively little data available and you need to evaluate your model as precisely as possible"
k = 4
num_validation_samples = len(data) 
np.random.shuffle(data)
for fold in range(k):
	validation_data = data[num_validation_samples*fold:num_validation_samples * (fold + 1)]
	training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):] 
	model = get_model() 
	model.train(training_data)
	validation_score = model.evaluate(validation_data)
	validation_scores.append(validation_score)
validation_score = np.average(validation_scores)   
# Trains the final model on all non-test data available
model = get_model()  
model.train(data)      
test_score = model.evaluate(test_data)  


# Dealing with Overfitting
# The general workflow to find an appropriate model size is to start with relatively few layers and parameters, 
# and increase the size of the layers or add new layers until you see diminishing returns with regard to 
# validation loss.
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# replace previous network for a smaller one with lower capacity
model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# The more capacity the network has, the more quickly it can model 
# the training data (resulting in a low training loss), but the more susceptible 
# it is to overfitting (resulting in a large difference between the training and validation loss).









