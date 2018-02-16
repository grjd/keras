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
	# 	3.1. Transformation (scaling, discretize continuous variables, expand categorical variables )
	#	3.2. Variable Selection
	#	3.3. Detect Multicollinearity
	# cleanup_column_names(df,rename_dict={},do_inplace=True)
	#cleanup_column_names(dataset, {}, True)
	# run_data_wrangling: (3.1) cleaning, transforming, and mapping data
	[explanatory_features, target_feature] = select_expl_and_target_features(dataset)
	run_data_wrangling(dataset, explanatory_features)
	# 3.2 filter the features relevant
	# (3.3) detect multicollinearity: Plot correlation and graphs of variables
	explanatory_features = ['visita_1_mmse', 'visita_1_fcsrttotdem','anos_escolaridad','apoe','preocupacion_v1','nlg_v1', 'visita_1_cdrtot']
	explanatory_features = ['visita_1_eqm01', 'visita_1_eqm02', 'visita_1_eqm03', 'visita_1_eqm04']
	corr_matrix = run_correlation_matrix(dataset,explanatory_features)
	pdb.set_trace()
	run_detect_multicollinearity(corr_matrix, threshold = np.mean(corr_matrix))

	# scatter_plot_target_cond(dataset)
	X_prep, X_train, X_test, y_train, y_test = run_feature_engineering(dataset)

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

def run_data_wrangling(dataset, explanatory_features):
	"""cleaning, transforming, and mapping data from one form to another ready for 
	analytics, summarization, reporting, visualization etc."""
	# clean up

	for f in range(len(explanatory_features)):
		# drop na rows
		dataset[explanatory_features[f]].dropna(inplace=True)
		# change str for numeric
		#if isinstance(dataset[explanatory_features[f]].values[0], basestring):
		#	print("str column!!",feature_labels_str[f], "\n" )
		#	pd.to_numeric(dataset[feature_labels_str[f]])
		print("scaling for: ",explanatory_features[f], " ...")	
		# select features that need to be preprocessed
		#dataset[explanatory_features[f]] = preprocessing.scale(dataset[explanatory_features[f]])	
	return dataset
	
def select_expl_and_target_features(dataset):
	""" select_expl_and_target_features: select features: explanatory and target.
	This function is called AFTER cleanup_column_names is called because it changes the feature names
	""" 
	#cleanup_column_names(df,rename_dict={},do_inplace=True)
	cleanup_column_names(dataset, {}, True)
	explanatory_features = dataset.keys()
	print("Explanatory variables:  {}".format(len(dataset.columns)-1))
	# Select subset of explanatory variables from prior information
	#explanatory_features = [ 'visita_1_EQ5DMOV', 'visita_1_EQ5DCP', 'visita_1_EQ5DACT', 'Visita_1_EQ5DDOL', 'Visita_1_EQ5DANS', 'Visita_1_EQ5DSALUD', 'Visita_1_EQ5DEVA', 'Visita_1_ALFRUT', 'Visita_1_ALCAR', 'Visita_1_ALPESBLAN', 'Visita_1_ALPESZUL', 'Visita_1_ALAVES', 'Visita_1_ALACEIT', 'Visita_1_ALPAST', 'Visita_1_ALPAN', 'Visita_1_ALVERD', 'Visita_1_ALLEG', 'Visita_1_ALEMB', 'Visita_1_ALLACT', 'Visita_1_ALHUEV', 'Visita_1_ALDULC', 'Visita_1_HSNOCT', 'Visita_1_RELAFAMI', 'Visita_1_RELAAMIGO', 'Visita_1_RELAOCIO', 'Visita_1_RSOLED', 'Visita_1_A01', 'Visita_1_A02', 'Visita_1_A03', 'Visita_1_A04', 'Visita_1_A05', 'Visita_1_A06', 'Visita_1_A07', 'Visita_1_A08', 'Visita_1_A09', 'Visita_1_A10', 'Visita_1_A11', 'Visita_1_A12', 'Visita_1_A13', 'Visita_1_A14', 'Visita_1_EJFRE', 'Visita_1_EJMINUT', 'Visita_1_VALCVIDA', 'Visita_1_VALSATVID', 'Visita_1_VALFELC', 'Visita_1_SDESTCIV', 'Visita_1_SDHIJOS', 'Visita_1_NUMHIJ', 'Visita_1_SDVIVE', 'Visita_1_SDECONOM', 'Visita_1_SDRESID', 'Visita_1_SDTRABAJA', 'Visita_1_SDOCUPAC', 'Visita_1_SDATRB', 'Visita_1_HTA', 'Visita_1_HTA_INI', 'Visita_1_GLU', 'Visita_1_LIPID', 'Visita_1_LIPID_INI', 'Visita_1_TABAC', 'Visita_1_TABAC_INI', 'Visita_1_TABAC_FIN', 'Visita_1_SP', 'Visita_1_COR', 'Visita_1_COR_INI', 'Visita_1_ARRI', 'Visita_1_CARD', 'Visita_1_CARD_INI', 'Visita_1_TIR', 'Visita_1_ICTUS',  'Visita_1_ICTUS_INI', 'Visita_1_ICTUS_SECU', 'Visita_1_DEPRE', 'Visita_1_DEPRE_INI', 'Visita_1_DEPRE_NUM', 'Visita_1_ANSI', 'Visita_1_ANSI_NUM', 'Visita_1_ANSI_TRAT', 'Visita_1_TCE', 'Visita_1_TCE_NUM', 'Visita_1_TCE_INI', 'Visita_1_TCE_CON', 'Visita_1_SUE_DIA',  'Visita_1_SUE_CON', 'Visita_1_SUE_MAN', 'Visita_1_SUE_SUF', 'Visita_1_SUE_PRO', 'Visita_1_SUE_RON', 'Visita_1_SUE_MOV', 'Visita_1_SUE_RUI', 'Visita_1_SUE_HOR', 'Visita_1_SUE_DEA', 'Visita_1_SUE_REC', 'Visita_1_EDEMMAD','Visita_1_EDEMPAD', 'Visita_1_PABD', 'Visita_1_PESO', 'Visita_1_TALLA', 'Visita_1_AUDI', 'Visita_1_VISU', 'Visita_1_IMC','Visita_1_GLU_INI','Visita_1_TABAC_CANT','Visita_1_ARRI_INI', 'Visita_1_ICTUS_NUM', 'Visita_1_DEPRE_TRAT','Visita_1_ANSI_INI', 'Visita_1_TCE_SECU', 'Visita_1_SUE_NOC']
	#print("Explanatory variables:  {}".format(len(explanatory_features)))
	target_feature = ['conversion']
	print("Number of Observations: {}".format(dataset.shape[0]))
	print("Target variable:       '{}' -> '{}'".format('conversion', 'target'))
	return explanatory_features, target_feature 

def cleanup_column_names(df,rename_dict={},do_inplace=True):
    """This function renames columns of a pandas dataframe
       It converts column names to snake case if rename_dict is not passed. 
    Args:
        rename_dict (dict): keys represent old column names and values point to 
                            newer ones
        do_inplace (bool): flag to update existing dataframe or return a new one
    Returns:
        pandas dataframe if do_inplace is set to False, None otherwise
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

def run_correlation_matrix(dataset,feature_label=None):
	""" plot sns map with correlation matrix"""
	fig, ax = plt.subplots(1,1)
	ax.xaxis.set_tick_params(which='both')
	ax.tick_params(direction='out', length=6, width=2, colors='b')
	cmethod = ['pearson', 'kendall', 'spearman']
	cmethod = cmethod[-1]
	corr = dataset[feature_label].corr(method=cmethod) #method : {‘pearson’, ‘kendall’, ‘spearman’}
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