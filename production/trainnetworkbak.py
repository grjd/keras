
#https://www.datacamp.com/community/tutorials/deep-learning-python
#DB_WEB = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
#DB_WEB = "/Users/jaime/vallecas/data/surrogate_bcpa/postmortem-JG-AR-17122017.csv"
#dataset = pd.read_csv(DB_WEB) #, sep=';')

# Train and test sets
# Imbalanced data typically refers to a problem with classification problems where the classes 
# are not represented equally, otherwise you are favouring one class versus the other
# Import the train_test_split from sklearn.
# Specify the input data, note that wine_w is type pandas not ndarray
# X and y are ndarray
# X = dataset.ix[:,0:-2] # all columns except last one
# Select features specifically
X = dataset[['pH', 'lactate', 'rin']]
# Select features specifically by isolating data
# X = wines.drop('quality', axis=1) 
#ravel return a contiguous flattened array. eg [[1,2], [3,4]] returns [1,2,3,4]
y = np.ravel(dataset['ipm']) 
# Split the data up in train and test sets, random_state is the seed used by the random number generator
#  default test_size value is set to 0.25. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Standarize : Standardization is a way to deal with these values that lie so far apart
# import the StandardScaler module from sklearn.preprocessing 
# Define the scaler 
scaler = StandardScaler().fit(X_train)
# Scale the train 
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
model.add(Dense(12, activation='relu', input_shape=(X_train.shape[1],)))
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
X = dataset[['pH', 'lactate', 'rin']]
X = StandardScaler().fit_transform(X)
y = np.ravel(dataset['ipm']) 

seed = 7
np.random.seed(seed)
# setup for k fold validation
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
# learning rate of optimizer
rmsprop = RMSprop(lr=0.0001)
sgd=SGD(lr=0.1)
for train, test in kfold.split(X, y):
	# Initialize the model
    model_c = Sequential()
    # input layer
    model_c.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    # add additional hidden layers
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # Add output layer, no activation typical setup for scalar regression (predict a single continuous value)
    model_c.add(Dense(1))
    model_c.compile(optimizer='sgd', loss='mse', metrics=['mae'])
    model_c.fit(X[train], y[train], epochs=10, verbose=1)

y_pred = model_c.predict(X[test])
# evaluate: Scalar test loss. The attribute model.metrics_names will give you the display labels for the scalar outputs.
mse_value, mae_value = model_c.evaluate(X[test], y[test], verbose=1)

print ("Evaluating the model for mean squared deviation mse=",mse_value)
print ("Evaluating the model for mean absolute error mae=",mae_value)
# 1.0 best score
print ("R2 score y_test vs y_pred (best 1.0) =",r2_score(y[test], y_pred))


# Model Fine-Tuning
