# -*- coding: utf-8 -*-
"""
========================
Data Pipeline:
	Import data (csv or xlsx)
	Data Preparation
	Descriptive analytics
	Feature Engineering
	Dimensionality Reduction
	Modeling
	Explainability
========================
"""
from __future__ import print_function
import pdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras import regularizers
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import Callback
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import preprocessing, linear_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from random import randint
import seaborn as sns
# regex
import re
from patsy import dmatrices
import itertools
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss, accuracy_score, confusion_matrix
import warnings
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
import xgboost as xgb
from sklearn.manifold import TSNE
from sklearn import metrics
import importlib


def main():
	plt.close('all')
	# get the data in csv format
	dataset = run_load_csv()
	# 3. Data preparation: 
	# 	3.1.  Variable Selection 
	#	3.2.  Transformation (scaling, discretize continuous variables, expand categorical variables)
	#	3.3. Detect Multicollinearity

	# (3.1) Variable Selection : cosmetic name changing and select input and output 
	# cleanup_column_names(df,rename_dict={},do_inplace=True) cometic cleanup lowercase, remove blanks
	cleanup_column_names(dataset, {}, True)
	run_print_dataset(dataset)
	features_list = dataset.columns.values.tolist()
	print(features_list)
	# Select subset of explanatory variables from prior information
	explanatory_features = ['sexo', 'visita_1_mmse','preocupacion_v1','nivel_educativo', 'anos_escolaridad', 'apoe','scd_v1','visita_1_fcsrtlibdem', 'visita_1_fcsrttotdem', 'visita_1_p']	
	#explanatory_features = None. # If None explanatory_features assigned to features_list
	target_variable = None # if none assigned to 'conversion'. target_variable = ['visita_1_EQ5DMOV']
	print("Calling to run_variable_selection(dataset, explanatory_features= {}, target_variable={})".format(explanatory_features, target_variable))
	dataset = run_variable_selection(dataset,explanatory_features,target_variable)
	# (3.2) Transformation (scaling, discretize continuous variables, expand categorical variables)
	dataset, X_imputed = run_imputations(dataset)
	print("X_imputed:\n{}".format(X_imputed))
	# If necessay, run_binarization_features and run_encoding_categorical_features
	X_imputed_scaled = run_transformations(X_imputed) # standarize to minmaxscale or others make input normal
	print("X_imputed_scaled:\n{}".format(X_imputed_scaled))
	# (3.3) detect multicollinearity: Plot correlation and graphs of variables
	#convert ndarray to pandas DataFrame
	Xdf_imputed_scaled = pd.DataFrame(X_imputed_scaled, columns=explanatory_features)
	#corr_matrix = run_correlation_matrix(Xdf_imputed_scaled,explanatory_features)
	# run_correlation_matrix(Xdf_imputed_scaled, explanatory_features[0:3])
	corr_df = run_correlation_matrix(Xdf_imputed_scaled)
	#corr_matrix = corr_df.as_matrix()
	build_graph_correlation_matrix(corr_df, threshold = np.mean(corr_df.as_matrix()))
	pdb.set_trace()

	# scatter_plot_target_cond(dataset)
	X_prep, X_train, X_test, y_train, y_test = run_feature_engineering(dataset) # sacar regression ocuparse solo de Scale y transform y llamsr antes q multicoll

	## Build model and fit to data
	model = ['LogisticRegression','RandomForestClassifier', 'XGBooster'] 
	run_fitmodel(model[0], X_train, X_test, y_train, y_test)
	run_fitmodel(model[1], X_train, X_test, y_train, y_test)
	run_fitmodel(model[2], X_train, X_test, y_train, y_test)

	model, model2, activations = run_Keras_DN(None, X_train, X_test, y_train, y_test)
	#activations.shape = X_test.shape[0], number of weights
	samples = run_tSNE_analysis(activations, y_test)
	run_TDA_with_Kepler(samples, activations)
	pdb.set_trace()
	


	
	
	
	# run descriptive analytics for longitudinal variables
	#longit_pattern = re.compile("^Visita_[0-9]_FCSRTlibinm+$")
	run_longitudinal_analytics(dataset)
	#run_feature_engineering(dataset)

	feature_label = run_histogram_and_scatter(dataset)
	run_correlation_matrix(dataset,feature_label)
	run_descriptive_stats(dataset,feature_label)
	run_logistic_regression(dataset, feature_label)
	#run_svm()
	#run_networks()

def run_print_dataset(dataset):
	""" run_print_dataset: print information about the dataset, type of features etc
	Agrs: Pandas dataset
	Output: None"""

	print("dtypes of the Pandas dataframe :\n\n{}".format(dataset.dtypes))
	print("\n\n value_counts of dataframe :\n")
	for colix in range(dataset.shape[1]):
		print(dataset.ix[:,colix].value_counts())

def select_featuresindataset(dataset, explanatory_features, dropna=None):
	"""select_featuresindataset: Selects a list of features from the original dataset. Called from select_expl_and_target_features
	Arg: dataset, explanatory_features,dropna. dropna False by default if True delete NAN rows  
	"""
	if dropna is None:
		dropna = False
	if dropna is True:
		subsetdataset = dataset[explanatory_features].dropna()
	else:
		subsetdataset = dataset[explanatory_features]
	return subsetdataset	
	#for f in range(len(explanatory_features)):
		# drop na rows
		#dataset[explanatory_features[f]].dropna(inplace=True)
		# change str for numeric
		#if isinstance(dataset[explanatory_features[f]].values[0], basestring):
		#	print("str column!!",feature_labels_str[f], "\n" )
		#	pd.to_numeric(dataset[feature_labels_str[f]])
		#print("scaling for: ",explanatory_features[f], " ...")	
		# select features that need to be preprocessed
		#dataset[explanatory_features[f]] = preprocessing.scale(dataset[explanatory_features[f]])	
	#return dataset
	
def run_variable_selection(dataset, explanatory_features=None,target_variable=None):
	"""run_variable_selection: select features: explanatory and target.
	Args: dataset, explanatory_features : list of explanatory variables if None assigned all the features dataset.keys()
	target_variable: target feature, if None is assigned inside the function 
	""" 
	if target_variable is None:
		target_variable = ['conversion']
	target_feature = target_variable
	if explanatory_features is None:
		explanatory_features = dataset.keys()
	else:
		print("Original dataframe features:  {}".format(len(dataset.columns)-1))
		dataset = select_featuresindataset(dataset, explanatory_features, dropna=False)
			
	print("Number of Observations: {}".format(dataset.shape[0]))
	print("Number of selected dataframe explanatory features:  {}".format(len(dataset.columns)-1))
	print("Target variable:       '{}' -> '{}'".format('conversion', 'target'))
	print(" explanatory features:  {}".format(dataset.keys()))
	return dataset 

def cleanup_column_names(df,rename_dict={},do_inplace=True):
    """cleanup_column_names: renames columns of a pandas dataframe. It converts column names to snake case if rename_dict is not passed. 
    Args: rename_dict (dict): keys represent old column names and values point to newer ones
        do_inplace (bool): flag to update existing dataframe or return a new one
    Returns: pandas dataframe if do_inplace is set to False, None otherwise
    """
    if not rename_dict:
        return df.rename(columns={col: col.replace('/','').lower().replace(' ','_') 
                    for col in df.columns.values.tolist()}, inplace=do_inplace)
    else:
        return df.rename(columns=rename_dict,inplace=do_inplace)
    #to drop coulumns
    #df = df.drop('ferature_name', axis=1)

def run_longitudinal_analytics(df, longit_pattern=None):
	""" descriptive analytics for longitudinal features e.g. test results for each visit"""
	if longit_pattern is None:
		longit_pattern = re.compile("^visita_[0-9]_fcsrtlibinm+$") 
		#longit_pattern = re.compile("^scd_v[0-9]+$") 
		#(Visita_1_SP) sobrepeso, Visita_1_DEPRE,Visita_1_ANSI,Visita_1_TCE (traumatismo cabeza)
		#Visita_1_SUE_DIA (duerme dia) Visita_1_SUE_NOC (duerme noche) Visita_1_IMC (indice masa corp) Visita_1_COR(corazon)
		#Visita_1_TABAC (fumador) Visita_1_VALFELC(felicidad)
	longit_status_columns = [ x for x in df.columns if (longit_pattern.match(x))]
	df[longit_status_columns].head(10)
	# plot histogram for longit pattern
	fig, ax = plt.subplots(2,2)
	fig.set_size_inches(15,5)
	fig.suptitle('Distribution of scd 4 visits')
	for i in range(len(longit_status_columns)):
		row,col = int(i/2), i%2
		d  = df[longit_status_columns[i]].value_counts()
		#n, bins, patches = ax[row,col].hist(d, 50, normed=1, facecolor='green', alpha=0.75)
    	# kernel density estimation
    	#kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(d.values.reshape(-1, 1))
    	#x_grid = np.linspace(d.min(), d.max(), 1000)
    	#log_pdf = kde.score_samples(x_grid.reshape(-1, 1))
    	# add the density line
    	#ax[row,col].plot(x_grid, np.exp(log_pdf), color='blue', alpha=0.5, lw=3)
    	#ax[row,col].set_title(longit_status_columns[i])
		ax[row,col].bar(d.index, d, align='center', color='g')
		ax[row,col].set_title(longit_status_columns[i])
	plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=1.0)
	plt.show()
	# histograms group by
	df.anos_escolaridad.describe()
	fig = plt.figure()
	fig.set_size_inches(20,5)
	ax = fig.add_subplot(111)
	grd = df.groupby(['anos_escolaridad']).size()
	ax.set_yscale("log")
	ax.set_xticks(np.arange(len(grd)))
	fig.suptitle('anos_escolaridad')
	#ax.set_xticklabels(['%d'  %i for i in d.index], rotation='vertical')
	p = ax.bar(np.arange(len(grd)), grd, color='purple')
	# plot histogram target feature
	fig = plt.figure()
	fig.set_size_inches(5,5)
	grd2 = df.groupby(['conversion']).size()
	print("conversion subjects are {}% out of {} observations".format(100* grd2[1]/(grd2[1]+grd2[0]), grd2[1]+grd2[0]))
	p = grd2.plot(kind='barh', color='orange')
	fig.suptitle('number of conversors vs non conversors')
	# target related to categorical features
	categorical_features = ['sexo', 'nivel_educativo', 'apoe']
	df['sexo'] = df['sexo'].astype('category').cat.rename_categories(['M', 'F'])
	df['nivel_educativo'] = df['nivel_educativo'].astype('category').cat.rename_categories(['~Pr', 'Pr', 'Se', 'Su'])
	df['apoe'] = df['apoe'].astype('category').cat.rename_categories(['No', 'Het', 'Hom'])
	df['visita_1_edad_cat'] = pd.cut(df['visita_1_edad'], range(0, 100, 10), right=False)
	#in absolute numbers
	fig, ax = plt.subplots(1,4)
	fig.set_size_inches(20,5)
	fig.suptitle('Conversion by absolute numbers, for various demographics')
	d = df.groupby(['conversion', 'sexo']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0])
	d = df.groupby(['conversion', 'nivel_educativo']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[1])
	d = df.groupby(['conversion', 'apoe']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[2])
	d = df.groupby(['conversion', 'visita_1_edad_cat']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[3])
	#in relative numbers
	fig, ax = plt.subplots(1,4)
	fig.set_size_inches(20,5)
	fig.suptitle('Conversion by relative numbers, for various demographics')
	d = df.groupby(['conversion', 'sexo']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0])
	d = df.groupby(['conversion', 'nivel_educativo']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1])
	d = df.groupby(['conversion', 'apoe']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[2])
	d = df.groupby(['conversion', 'visita_1_edad_cat']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[3])


def selcols(prefix, a=1, b=4):
	""" selcols: return list of str of longitudinal variables"""
	return [prefix+str(i) for i in np.arange(a,b+1)]

def run_imputations(dataset):
	""" run_imputations: datasets with missign values are incompatible with scikit-learn 
	estimators which assume that all values in an array are numerical, and that all have and hold meaning.
	http://scikit-learn.org/stable/modules/preprocessing.html
	 """
	from sklearn.preprocessing import Imputer
	print( "Number of rows in the dataframe:{}".format(dataset.shape[0]))
	print ("Features containing NAN values:\n {}".format(dataset.isnull().any()))
	print(dataset) 
	imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
	imp.fit(dataset)
	X_train_imputed = imp.transform(dataset)
	print( "Number of rows in Imputed dataframe:{}".format(np.count_nonzero(~np.isnan(X_train_imputed))))
	return dataset, X_train_imputed

def run_binarization_features(dataset):
	"""run_binarizations thresholding numerical features to get boolean values.
	Useful for downstream probabilistic estimators that make assumption that the input data is distributed according to a multi-variate Bernoulli distribution
	Args: dataset ndarray 
	Output: binary matrix, 1 if value larger than threshold"""
	threshold = np.mean(dataset)
	binarizer = preprocessing.Binarizer(threshold=threshold)  # fit does nothing
	X_binarized = binarizer.transform(dataset)
	return X_binarized

def run_encoding_categorical_features(dataset):
	"""run_encoding_categorical_features: Often features are not given as continuous values but categorical. They could be
	efficiently coded as integers, eg male, female -> 0,1 BUT this cannot be used directly with scikit-learn estimators, 
	as these expect continuous input, and would interpret the categories as being ordered.
	OneHotEncoder. is an estimator transforms each categorical feature with m possible values into m binary features, with only one active.
	http://pbpython.com/categorical-encoding.html"""
	enc = preprocessing.OneHotEncoder()
	enc.fit(dataset)

def run_transformations(dataset):
	""" run_transformations performs scaling discretization and categorization. Estimators may behave badly of data are not normally distributed:
	Gaussian with zero mean and unit variance.In practice we often ignore the shape of the distribution and just transform the data to 
	center it by removing the mean value of each feature, then scale it by dividing non-constant features by their standard deviation.
	IMPORTANT : RBF kernel of Support Vector Machines or the l1 and l2 regularizers of linear models) assume that all features are centered around zero and 
	have variance in the same order. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective 
	function and make the estimator unable to learn from other features correctly as expected.
	Args: dataset:ndarray """
	# feature scaling scaling individual samples to have unit norm
	# the quick way to normalize : X_scaled = preprocessing.scale(X_train), fit_transform is practical if we are going to train models 
	# To scale [-1,1] use MaxAbsScaler(). MinMaxScaler formula is std*(max-min) + min
	scaler = preprocessing.MinMaxScaler()
	#The same instance of the transformer can then be applied to some new test data unseen during the fit call:
	# the same scaling and shifting operations will be applied to be consistent with the transformation performed on the train data
	X_train_minmax = scaler.fit_transform(dataset)
	print("Orignal ndarray \n {}".format(dataset))
	#print("X_train_minmax \n {}".format(X_train_minmax))
	return X_train_minmax


def run_feature_engineering(df):
	""" run_feature_engineering(X) : builds the design matrix also feature selection
	return X_prep, X_train, X_test, y_train, y_test """
	# Construct a single design matrix given a formula_ y ~ X
	formula = 'conversion ~ '
	# original features
	formula += 'C(sexo) + C(nivel_educativo) +  C(apoe) '
	#### engineered / normalized features
	# categorical age
	#formula += '+' + 'C(visita_1_edad_cat)'
	#longitudinal
	formula += '+' + '+'.join(selcols('scd_v'))
	print("The formula is:", formula)
	# patsy dmatrices, construct a single design matrix given a formula_like and data.
	y, X = dmatrices(formula, data=df, return_type='dataframe')
	y = y.iloc[:, 0]
	# feature scaling
	scaler = preprocessing.MinMaxScaler()
	scaler.fit(X)
	""" select top features and find top indices from the formula  """
	nboffeats = 6
	warnings.simplefilter(action='ignore', category=(UserWarning,RuntimeWarning))
	# sklearn.feature_selection.SelectKBest, select k features according to the highest scores
	# SelectKBest(score_func=<function f_classif>, k=10)
	# f_classif:ANOVA F-value between label/feature for classification tasks.
	# mutual_info_classif: Mutual information for a discrete target.
	# chi2: Chi-squared stats of non-negative features for classification tasks.
	function_f_classif = ['f_classif', 'mutual_info_classif', 'chi2', 'f_regression', 'mutual_info_regression']
	selector = SelectKBest(chi2, nboffeats)
	#Run score function on (X, y) and get the appropriate features.
	selector.fit(X, y)
	# scores_ : array-like, shape=(n_features,) pvalues_ : array-like, shape=(n_features,)
	top_indices = np.nan_to_num(selector.scores_).argsort()[-nboffeats:][::-1]
	print("Selector scores:",selector.scores_[top_indices])
	print("Top features:\n", X.columns[top_indices])
	# Pipeline of transforms with a final estimator.Sequentially apply a list of transforms and a final estimator.
	# sklearn.pipeline.Pipeline
	preprocess = Pipeline([('anova', selector), ('scale', scaler)])
	print("Estimator parameters:", preprocess.get_params())
	# Fit the model and transform with the final estimator. X =data to predict on
	preprocess.fit(X,y)
	# transform: return the transformed sample: array-like, shape = [n_samples, n_transformed_features]
	X_prep = preprocess.transform(X)
	# model selection
	X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.3, random_state=42)
	return X_prep, X_train, X_test, y_train, y_test

def run_fitmodel(model, X_train, X_test, y_train, y_test):
	""" fit the model (LR, random forest others) and plot the confusion matrix and RUC """
	if model == 'LogisticRegression':
		# Create logistic regression object
		regr = linear_model.LogisticRegression()
		# Train the model using the training sets
		regr.fit(X_train, y_train)
		# model prediction
		y_train_pred = regr.predict_proba(X_train)[:,1]
		y_test_pred = regr.predict_proba(X_test)[:,1]
		threshold = 0.5

		fig,ax = plt.subplots(1,3)
		fig.set_size_inches(15,5)
		plot_cm(ax[0],  y_train, y_train_pred, [0,1], 'LogisticRegression Confusion matrix (TRAIN)', threshold)
		plot_cm(ax[1],  y_test, y_test_pred,   [0,1], 'LogisticRegression Confusion matrix (TEST)', threshold)
		plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, threshold)
		plt.tight_layout()
		plt.show()
	if model == 'RandomForestClassifier':
		rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5)
		rf.fit(X_train,y_train)
		threshold = 0.5
		y_train_pred = rf.predict_proba(X_train)[:,1]
		y_test_pred = rf.predict_proba(X_test)[:,1]
		fig,ax = plt.subplots(1,3)
		fig.set_size_inches(15,5)
		plot_cm(ax[0],  y_train, y_train_pred, [0,1], 'RandomForestClassifier Confusion matrix (TRAIN)', threshold)
		plot_cm(ax[1],  y_test, y_test_pred,   [0,1], 'RandomForestClassifier Confusion matrix (TEST)', threshold)
		plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, threshold)
		plt.tight_layout()
		plt.show()
	if model == 'XGBooster':
		# gradient boosted decision trees  for speed and performance
		#https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
		# 2 versions with DMatrix and vanilla classifier
		dtrain = xgb.DMatrix(X_train, label=y_train)
		dtest = xgb.DMatrix(X_test, label=y_test)
		num_round = 5
		evallist  = [(dtest,'eval'), (dtrain,'train')]
		param = {'objective':'binary:logistic', 'silent':1, 'eval_metric': ['error', 'logloss']}
		bst = xgb.train( param, dtrain, num_round, evallist)
		threshold = 0.5
		y_train_pred = bst.predict(dtrain)
		y_test_pred = bst.predict(dtest)
		fig,ax = plt.subplots(1,3)
		fig.set_size_inches(15,5)
		plot_cm(ax[0],  y_train, y_train_pred, [0,1], 'XGBooster Confusion matrix (TRAIN)', threshold)
		plot_cm(ax[1],  y_test, y_test_pred,   [0,1], 'XGBooster Confusion matrix (TEST)', threshold)
		plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, threshold)
		plt.tight_layout()
		# vanilla XGBooster classifie
		XGBmodel = XGBClassifier()
		XGBmodel.fit(X_train, y_train)
		# make predictions for test data
		y_pred = XGBmodel.predict(X_test)
		predictions = [round(value) for value in y_pred]
		# evaluate predictions
		accuracy = accuracy_score(y_test, predictions)
		print("Accuracy XGBooster classifier: %.2f%%" % (accuracy * 100.0))
		plt.show()

def scatter_plot_target_cond(df):	
	# average and standard deviation of longitudinal status
	df['scd_avg'] = df[selcols('scd_v')].mean(axis=1)
	df['scd_std'] = df[selcols('scd_v')].std(axis=1)
	# sobre peso
	suffix = '_sp'
	suffix = '_stai'
	prefix = selcols('visita_')
	for idx, val in enumerate(prefix):
		val += suffix
		prefix[idx] = val
	df['stai_avg'] = df[prefix].mean(axis=1)
	df['stai_std'] = df[prefix].std(axis=1)
	#scatter plot 2 averages and conversion non conversion
	def_no = df[df['conversion']==0]
	def_yes = df[df['conversion']==1]
	fig,ax = plt.subplots(2,2)
	fig.set_size_inches(15,10)

	x_lab = 'scd_avg'
	y_lab = 'stai_avg'
	ax[0,0].set_title('conversion')
	ax[0,0].set_ylabel(y_lab)
	ax[0,0].set_xlabel(x_lab)
	p = ax[0,0].semilogy(def_yes['scd_avg'].dropna(inplace=False), def_yes['stai_avg'].dropna(inplace=False), 'bo', markersize=5, alpha=0.1)

	ax[0,1].set_title('non conversors')
	ax[0,1].set_ylabel(y_lab)
	ax[0,1].set_xlabel(x_lab)
	p = ax[0,1].semilogy(def_no['scd_avg'].dropna(inplace=False), def_no['scd_avg'].dropna(inplace=False), 'ro', markersize=5, alpha=0.1)
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	plt.show()


def plot_cm(ax, y_true, y_pred, classes, title, th=0.5, cmap=plt.cm.Blues):
    y_pred_labels = (y_pred>th).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_labels)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

def plot_auc(ax, y_train, y_train_pred, y_test, y_test_pred, th=0.5):

    y_train_pred_labels = (y_train_pred>th).astype(int)
    y_test_pred_labels  = (y_test_pred>th).astype(int)

    fpr_train, tpr_train, _ = roc_curve(y_train,y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    acc_train = accuracy_score(y_train, y_train_pred_labels)

    fpr_test, tpr_test, _ = roc_curve(y_test,y_test_pred)
    roc_auc_test = auc(fpr_test, tpr_test)
    acc_test = accuracy_score(y_test, y_test_pred_labels)

    ax.plot(fpr_train, tpr_train)
    ax.plot(fpr_test, tpr_test)

    ax.plot([0, 1], [0, 1], 'k--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    
    train_text = 'train acc = {:.3f}, auc = {:.2f}'.format(acc_train, roc_auc_train)
    test_text = 'test acc = {:.3f}, auc = {:.2f}'.format(acc_test, roc_auc_test)
    ax.legend([train_text, test_text])

#####
class BatchLogger(Callback):
    def on_train_begin(self, epoch, logs={}):
        self.log_values = {}
        for k in self.params['metrics']:
            self.log_values[k] = []

    def on_epoch_end(self, batch, logs={}):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values[k].append(logs[k])
    
    def get_values(self, metric_name, window):
        d =  pd.Series(self.log_values[metric_name])
        return d.rolling(window,center=False).mean()

def run_Keras_DN(dataset, X_train, X_test, y_train, y_test):
	""" deep network classifier using keras
	Remember to activate the virtual environment source ~/git...code/tensorflow/bin/activate"""

	if dataset is not None: 
		features = ['Visita_1_MMSE', 'years_school', 'SCD_v1', 'Visita_1_P', 'Visita_1_STAI', 'Visita_1_GDS','Visita_1_CN']
		df = dataset.fillna(method='ffill')
		X_all = df[features]
		y_all = df['Conversion']
		# split data into training and test sets
		print("Data set set dimensions: X_all=", X_all.shape, " y_all=", y_all.shape)
		cutfortraining = int(X_all.shape[0]*0.8)
		X_train = X_all.values[:cutfortraining, :]
		y_train = y_all.values[:cutfortraining]
		print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
		X_test = X_all.values[cutfortraining:,:]
		y_test = y_all.values[cutfortraining:]
		print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)

		model = Sequential()
		model.add(Dense(15, input_dim=len(features), activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
		model.fit(X_train, y_train, epochs=20, batch_size=50)

		predictions = model.predict_classes(X_test)
		print('Accuracy:', metrics.accuracy_score(y_true=y_test, y_pred=predictions))
		print(metrics.classification_report(y_true=y_test, y_pred=predictions))
	else:
		####  run_keras_dn(dataset, X_train, X_test, y_train, y_test):
		input_dim = X_train.shape[1]
		model = Sequential()
		model.add(Dense(16, input_shape=(input_dim,), activation='relu'))
		model.add(Dense(1,  activation='sigmoid'))
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		bl = BatchLogger()
		history = model.fit(
	              np.array(X_train), np.array(y_train),
	              batch_size=25, epochs=15, verbose=1, callbacks=[bl],
	              validation_data=(np.array(X_test), np.array(y_test)))
		score = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
		print('Test log loss:', score[0])
		print('Test accuracy:', score[1])
		plt.figure(figsize=(15,5))
		plt.subplot(1, 2, 1)
		plt.title('loss, per batch')
		plt.legend(['train', 'test'])
		plt.plot(bl.get_values('loss',1), 'b-', label='train');
		plt.plot(bl.get_values('val_loss',1), 'r-', label='test');
		plt.subplot(1, 2, 2)
		plt.title('accuracy, per batch')
		plt.plot(bl.get_values('acc',1), 'b-', label='train');
		plt.plot(bl.get_values('val_acc',1), 'r-', label='test');
		plt.show()
		#
		y_train_pred = model.predict_on_batch(np.array(X_train))[:,0]
		y_test_pred = model.predict_on_batch(np.array(X_test))[:,0]
		fig,ax = plt.subplots(1,3)
		fig.set_size_inches(15,5)
		plot_cm(ax[0], y_train, y_train_pred, [0,1], 'DN Confusion matrix (TRAIN)')
		plot_cm(ax[1], y_test, y_test_pred, [0,1], 'DN Confusion matrix (TEST)')
		plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred)   
		plt.tight_layout()
		plt.show()
		# we build a new model with the activations of the old model
		# this model is truncated before the last layer
		model2 = Sequential()
		model2.add(Dense(16, input_shape=(input_dim,), activation='relu', weights=model.layers[0].get_weights()))
		activations = model2.predict_on_batch(np.array(X_test))
		ax[0].legend('conversion')
		ax[1].legend('no conversion')
		return model, model2, activations

def run_TDA_with_Kepler(samples, activations):
	""" """
	import keplermap as km #https://github.com/natbusa/deepcredit/blob/master/km.py
	from sklearn.cluster import DBSCAN
	# Initialize
	mapper = km.KeplerMapper(verbose=1)
	# Fit to and transform the data
	projected_data = mapper.fit_transform(samples, projection=[0,1]) #list(range(activations.shape[1])))

	projected_data.shape
	# Create dictionary called 'complex' with nodes, edges and meta-information
	complex = mapper.map(projected_data, activations, nr_cubes=4,clusterer=DBSCAN(eps=0.01, min_samples=1), overlap_perc=0.1)
	# Visualize it
	mapper.visualize(complex, path_html="output.html", show_title=False, 
                 show_meta=False, bg_color_css="#FFF", graph_gravity=0.4)

	from IPython.display import IFrame
	IFrame('output.html', width='100%', height=700)
	pdb.set_trace()


def run_tSNE_analysis(activations, y_test):
	tsne = TSNE(n_components=2, perplexity=25, verbose=0, n_iter=500, random_state=1337)
	samples = tsne.fit_transform(activations)
	fig,ax = plt.subplots(1,2)
	fig.set_size_inches(15,5)
	ax[0].scatter(*samples[y_test==0].T,color='b', alpha=0.5, label='conversion: NO')
	ax[1].scatter(*samples[y_test==1].T,color='r', alpha=0.5, label='conversion: YES')
	plt.tight_layout()
	plt.show()
	return samples

def run_logistic_regression(dataset, features=None):
	""" logistic regression, answer two points: what is the baseline prediction of disease progression and 
	which independent variables are important facors for predicting disease progression"""
	#diabetes dataset
	#from sklearn import datasets
	#diabetes = datasets.load_diabetes()
	#X = diabetes.data
	#y = diabetes.target
	#feature_names=['age', 'sex', 'bmi', 'bp','s1', 's2', 's3', 's4', 's5', 's6']
	# Lasso normal linear regression with L1 regularization (minimize the number of features or predictors int he model )
	from sklearn.linear_model import Lasso
	from sklearn import linear_model
	from sklearn.model_selection import GridSearchCV
	features =['years_school', 'SCD_v1']
	#features = ['Visita_1_MMSE', 'years_school', 'SCD_v1', 'Visita_1_P', 'Visita_1_STAI', 'Visita_1_GDS','Visita_1_CN']
	df = dataset.fillna(method='ffill')
	X_all = df[features]
	y_all = df['Conversion']
	# split data into training and test sets
	print("Data set set dimensions: X_all=", X_all.shape, " y_all=", y_all.shape)
	cutfortraining = int(X_all.shape[0]*0.8)
	X_train = X_all.values[:cutfortraining, :]
	y_train = y_all.values[:cutfortraining]
	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	X_test = X_all.values[cutfortraining:,:]
	y_test = y_all.values[cutfortraining:]
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	# define the model and the hyperparamer alpha (controls the stricteness of the regularization)
	lasso = Lasso(random_state=0)
	alphas = np.logspace(-6, -0.5, 10)
	# estimator with our model, in this case a grid search of model of Lasso type
	estimator = GridSearchCV(lasso, dict(alpha=alphas))
	# take train set and learn a group of Lasso models by varying the value of the alpha hyperparameter.
	estimator.fit(X_train, y_train)
	estimator.best_score_
	estimator.best_estimator_
	estimator.predict(X_test)
	# print results and ITERATE: making changes to the data transformation, Machine Learning algorithm, 
	#tuning hyperparameters of the algorithm etc.
	#Best possible score is 1.0, lower values are worse. Unlike most other scores
	#The score method of a LassoCV instance returns the R-Squared score, which can be negative, means performing poorly
	estimator.score(X_test,y_test)
	
def run_naive_Bayes(dataset, features=None):
	from plot_learning_curve import plot_learning_curve
	from sklearn.naive_bayes import GaussianNB
	title = "Learning Curves (Naive Bayes)"
	# Cross validation with 100 iterations to get smoother mean test and train
	# score curves, each time with 20% data randomly selected as a validation set.
	features =['years_school', 'SCD_v1']
	df = dataset.fillna(method='ffill')
	X_all = df[features]
	y_all = df['Conversion']
	print("Data set set dimensions: X_all=", X_all.shape, " y_all=", y_all.shape)
	cutfortraining = int(X_all.shape[0]*0.8)
	X = X_all.values[:cutfortraining, :]
	y = y_all.values[:cutfortraining]
	print("Training set set dimensions: X=", X.shape, " y=", y.shape)
	X_test = X_all.values[cutfortraining:,:]
	y_test = y_all.values[cutfortraining:]
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	
	estimator = GaussianNB()
	estimator.fit(X,y)
	# test
	predictions = [int(a) for a in estimator.predict(X_test)]
	num_correct = sum(int(a == ye) for a, ye in zip(predictions, y_test))
	print("Baseline classifier using an Naive Bayes.")
	print("%s of %s values correct." % (num_correct, len(y_test)))
	# plot learning curve
	cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
	plot_learning_curve(estimator, title, X_all.values, y_all.values, cv=cv, n_jobs=4)

def run_svm(dataset, features=None):
	""" svm classifier"""
	from sklearn.datasets.mldata import fetch_mldata
	from sklearn.svm import SVC
	from sklearn.model_selection import GridSearchCV
	from sklearn.model_selection import StratifiedKFold
	from plot_learning_curve import plot_learning_curve
	#from sklearn.grid_search import GridSearchCV
	# Fill NA/NaN values using the specified method: ffill: propagate last valid observation forward to next valid backfill 
	features =['Visita_1_EQ5DMOV', 'Visita_1_EQ5DCP', 'years_school', 'SCD_v1']
	df = dataset.fillna(method='ffill')
	#df = df.shift()
	X_all = df.loc[:, df.columns != 'Conversion']
	X_all = df[features]
	y_all = df['Conversion'] #.dropna()
	print("Data set set dimensions: X_all=", X_all.shape, " y_all=", y_all.shape)
	cutfortraining = int(X_all.shape[0]*0.8)
	X = X_all.values[:cutfortraining, :]
	y = y_all.values[:cutfortraining]
	print("Training set set dimensions: X=", X.shape, " y=", y.shape)
	X_test = X_all.values[cutfortraining:,:]
	y_test = y_all.values[cutfortraining:]
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	# SVM model, cost and gamma parameters for RBF kernel. out of the box
	svm = SVC(cache_size=1000, kernel='rbf')
	svm.fit(X, y)
	# test
	predictions = [int(a) for a in svm.predict(X_test)]
	num_correct = sum(int(a == ye) for a, ye in zip(predictions, y_test))
	print("Baseline classifier using an SVM.")
	print("%s of %s values correct." % (num_correct, len(y_test)))
	# plot learning curves
	kfolds = StratifiedKFold(5)
	cv = kfolds.split(X,y)
	title =  "Learning Curves (SVM)"
	plot_learning_curve(svm, title, X_all.values, y_all.values, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

	# Exhaustive search for SVM parameters to improve the out-of-the-box performance of vanilla SVM.
	parameters = {'C':10. ** np.arange(5,10), 'gamma':2. ** np.arange(-5, -1)}
	grid = GridSearchCV(svm, parameters, cv=cv, verbose=3, n_jobs=2)
	grid.fit(X, y)
	print("predicting")
	print("score: ", grid.score(X_test, y_test))
	print(grid.best_estimator_)


def run_load_csv(csv_path = None):
	""" load csv database, print summary of data"""
	if csv_path is None:
		csv_path = "/Users/jaime/vallecas/data/scc/sccplus-24012018.csv"
		csv_path = "/Users/jaime/vallecas/data/scc/SCDPlus_IM-MA-29012018-6years.csv"
	dataset = pd.read_csv(csv_path) #, sep=';')
	#summary of data
	print("Number f Rows=", dataset.shape[0])
	print("Name of the features:",dataset.columns.values)
	print("\n\nColumn data types:",dataset.dtypes)
	print("Number of Features=", dataset.shape[1])
	print("Columns with missing values::", dataset.columns[dataset.isnull().any()].tolist())
	print("Number of rows with missing values::", len(pd.isnull(dataset).any(1).nonzero()[0].tolist()))
	print("Sample Indices with missing data::",pd.isnull(dataset).any(1).nonzero()[0].tolist())
	print("General Stats::")
	print(dataset.info())
	print("Summary Stats::" )
	print(dataset.describe())
	return dataset
	# filtering data
	#mydict = {'ID/': 'myid'}
	muydict = {}
	# rename colum names as in mydict and remove blanks in colum names
	cleanup_column_names(dataset, mydict, True)
	#typecasting
	#dataset['date'] = pd.to_datetime(dataset.date)
	# transform categorical columns
	# input missing values: dataframe with rows without any missing dates
	#print("Drop Rows with missing dates::" )
	df_dropped = df.dropna(subset=['date'])
	dataset.dropna(subset=['SCD_v1'], thresh=0.5)
	#print("Shape::",df_dropped.shape)
	# less expensive is to replace missing values with a central tendency measure like mean or median
	print("Fill Missing Price values with mean price::" )
	df_dropped['price'].fillna(value=np.round(df.price.mean(),decimals=2),inplace=True)
	print("Fill Missing user_type values with value from  previous row (forward fill) ::" )
	df_dropped['user_type'].fillna(method='ffill',inplace=True)
	print("Fill Missing user_type values with value from next row (backward fill) ::" )
	df_dropped['user_type'].fillna(method='bfill',inplace=True)
	#handling duplicates
	dataset[dataset.duplicated(subset=['ID'])]
	dataset.drop_duplicates(subset=['ID'],inplace=True)
	# categorical data using get_dummies to one hot encode
	dataset_categorical = pd.get_dummies(dataset,columns=['Visita_1_ALEMB'])
	print(dataset_categorical.head())
	# normalizing values or feature scaling
	min_max_scaler = preprocessing.MinMaxScaler()
	np_scaled = min_max_scaler.fit_transform(dataset['EdadInicio_v1'].reshape(-1,1))
	dataset['normalized_EdadInicio_v1'] = np_scaled.reshape(-1,1)
	#data seummarization
	print(dataset['Visita_1_RELAFAMI'][dataset['EdadInicio_v1'] >70].mean())
	print(dataset['Visita_1_RELAFAMI'][dataset['Visita_1_ALPESZUL'] == 0].mean())
	print(dataset['Visita_1_RELAFAMI'][dataset['EdadInicio_v1'] >80].value_counts())
	print(df.groupby(['Visita_1_ALPESZUL'])['EdadInicio_v1'].sum())
	print(dataset['EdadInicio_v1'].value_counts())



	return dataset


def run_descriptive_stats(dataset,feature_label):
	#chi_square_of_df_cols(dataset, feature_label[0], feature_label[1])
	anova_test(dataset, [feature_label[1], feature_label[3]])

def run_datavisualization(dataset, feature_label=None):
	""" line charts, box plots"""
	# histogram gouped by visita 1 =0,1,2,3
	dataset[['Visita_1_ALPESZUL','EdadInicio_v1']].hist(by='Visita_1_ALPESZUL' ,sharex=True)

def run_histogram_and_scatter(dataset, feature_label=None):
	"""" plotting histogram and scatter """
	nb_of_bins = 15
	fig, ax = plt.subplots(1, 2)
	feature_label = ['Visita_1_MMSE', 'years_school', 'SCD_v1', 'Visita_1_P', 'Visita_1_STAI', 'Visita_1_GDS','Visita_1_CN']
	# features = [dataset.Visita_1_MMSE, dataset.years_school ]
	features = [dataset[feature_label[f]] for f in range(len(feature_label))]
	#features = [dataset[feature_label[0]], dataset[feature_label[1]]] 
	ax[0].hist(features[0].dropna(), nb_of_bins, facecolor='red', alpha=0.5, label="scc")
	ax[1].hist(features[1].dropna(), nb_of_bins, facecolor='red', alpha=0.5, label="scc")
	#fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
	#ax[0].set_ylim([0, dataset.ipm.shape[0]])
	#bubble_df.plot.scatter(x='purchase_week',y='price',c=bubble_df['enc_uclass'],s=bubble_df['total_transactions']*10)
	#plt.title('Purchase Week Vs Price Per User Class Based on Tx')
	ax[0].set_xlabel(feature_label[0])
	ax[0].set_ylabel("Frequency")
	ax[1].set_xlabel(feature_label[1])
	ax[1].set_ylabel("Frequency")
	msgsuptitle = feature_label[0] + " and " + feature_label[1]
	fig.suptitle(msgsuptitle)
	plt.show()
	fig, ax = plt.subplots(1,2, figsize=(8,4))
	ax[0].scatter(features[4], features[5], color="red")
	ax[1].scatter(features[2], features[3], color="blue")
	ax[0].set_title(feature_label[4] + " - " + feature_label[5])
	ax[1].set_title(feature_label[2] + " - " + feature_label[3])
	ax[0].set_xlabel(feature_label[4])
	ax[1].set_xlabel(feature_label[2])
	ax[0].set_ylabel(feature_label[5])
	ax[1].set_ylabel(feature_label[3])
	#ax[0].set_xlim([0,10])
	#ax[1].set_ylim([0,14])
	#fig.subplots_adjust(wspace=0.5)
	#fig.suptitle("Scatter plots")
	plt.show()
	return feature_label

def build_graph_correlation_matrix(corr_df, threshold=None):
	""" build_graph_correlation_matrix: requires package pip install pygraphviz
	Args:A is the dataframe correlation matrix
	Output:
	"""
	import networkx as nx
	import string
	# extract corr matrix fro the dataframe
	A_df = corr_df.as_matrix()

	node_names = corr_df.keys().tolist()
	if threshold is None:
		threshold = mean(corr_matrix)
	A = np.abs(A_df) > threshold
	fig, ax = plt.subplots()
	
	labels = {}
	for idx,val in enumerate(node_names):
		labels[idx] = val
	G = nx.from_numpy_matrix(A)
	pos=nx.spring_layout(G)
	nx.draw_networkx_nodes(G, pos)
	nx.draw_networkx_edges(G, pos)
	nx.draw_networkx_labels(G, pos, labels, font_size=9)
	plt.title('Binary Graph from correlation matrix{}'.format(node_names))
	plt.title('Binary Graph, threshold={0:.3g}'.format(threshold))
	pdb.set_trace()
	#G.node_attr.update(color="red", style="filled")
	#G.edge_attr.update(color="blue", width="2.0")

	#G.draw('/tmp/out.png', format='png', prog='neato')

def run_correlation_matrix(dataset,feature_label=None):
	""" run_correlation_matrix: calculate the correlation matrix and plot sns map with correlation matrix
	Args: dataset pandas dataframe, feature_label list of features of interest, if None calculate corr with all features in te dataframe
	Output: DataFrame containing the correlation matrix """
	fig, ax = plt.subplots(1,1)
	ax.xaxis.set_tick_params(which='both')
	ax.tick_params(direction='out', length=6, width=2, colors='b')
	cmethod = ['pearson', 'kendall', 'spearman']
	cmethod = cmethod[1] #method : {‘pearson’, ‘kendall’, ‘spearman’}
	if feature_label is None:
		corr = dataset.corr(method=cmethod) 
		feature_label = dataset.columns
	else:
		corr = dataset[feature_label].corr(method=cmethod) 
	g = sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values, vmin =-1, vmax=1, center=0,annot=True)
	g.set(xticklabels=[])
	#ax.set_xlabel("Features")
	#ax.set_ylabel("Features")
	ax.set_title(cmethod + " correlation")
	g.set(xticklabels=[])
	g.set_yticklabels(feature_label, rotation=30)
	plt.show()
	return corr

def chi_square_of_df_cols(df, col1, col2):
	df_col1, df_col2 = df[col1], df[col2]
	obs = np.array(df_col1, df_col2 ).T
	pdb.set_trace()

def anova_test(df, features=None):
	""" The one-way analysis of variance (ANOVA) is used to determine whether there are any statistically significant differences between the means of three or more independent (unrelated) groups.
	ANOVA test for pandas data set and features.  one-way ANOVA is an omnibus test statistic and cannot tell you which specific groups were statistically significantly different from each other, only that at least two groups were
	Example: anova_test(dataset, features = [dataset.keys()[1], dataset.keys()[2]]) 
	MSwithin = SSwithin/DFwithin"""
	#from scipy import stats
	#F, p = stats.f_oneway(dataset[features[0]], dataset[features[1]], dataset[features[1]])
	#print("ANOVA test groups", features, ". F =", F, " p =", p)
	import scipy.stats as ss
	import statsmodels.api as sm
	from statsmodels.formula.api import ols
	ctrl = df[features[0]].where(df['Conversion']==0).dropna()
	sccs = df[features[0]].where(df['Conversion']==1).dropna()
	print("Number of control subjects=",len(ctrl.notnull()))
	print("Number of scc subjects=",len(sccs.notnull()))
	for idx, val in enumerate(features):
		expl_control_var = features[idx] + ' ~ Conversion'
		mod = ols(expl_control_var, data=df).fit()
		aov_table = sm.stats.anova_lm(mod, typ=2)
		print("ANOVA table feature:", val) 
		print(aov_table)
		#Create a boxplot
		df.boxplot(features[idx], by='Conversion', figsize=(12, 8))

def ancova_test(dataset, features=None):
	""" ACNOVA test controlling for features that may have a relationship with the dependent variable"""
	print("Calculating ANCOVA for features:\n",df.ix[0])

def t_test(dataset, features=None):
	""" t test"""
	print("Calculating t test for features:\n",df.ix[0])



if __name__ == "__name__":
	main()