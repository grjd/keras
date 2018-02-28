# -*- coding: utf-8 -*-
"""
========================
Data Pipeline:
	Import data (csv or xlsx)
	3. Data Preparation
	4. Descriptive analytics
	5. Dimensionality Reduction
	6. Feature Engineering
	
	7. Modeling
	8. Explainability
========================
"""
from __future__ import print_function
import os, sys, pdb, operator
import numpy as np
import pandas as pd
import importlib
from random import randint
import matplotlib.pyplot as plt
from plot_learning_curve import plot_learning_curve
import seaborn as sns
import re # regex
from patsy import dmatrices
import itertools
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, ShuffleSplit
from sklearn.metrics import r2_score, roc_curve, auc, roc_auc_score, log_loss, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score,accuracy_score, classification_report, matthews_corrcoef
from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, linear_model, metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.datasets.mldata import fetch_mldata
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from plot_learning_curve import plot_learning_curve
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import XGBClassifier
	
from keras import regularizers
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import Callback

import networkx as nx




def main():
	plt.close('all')
	# get the data in csv format
	dataset = run_load_csv()
	# 3. Data preparation: 
	# 	3.1. Variable Selection 
	#	3.2. Transformation (scaling, discretize continuous variables, expand categorical variables)
	#	3.3. Detect Multicollinearity

	# (3.1) Variable Selection : cosmetic name changing and select input and output 
	# cleanup_column_names(df,rename_dict={},do_inplace=True) cometic cleanup lowercase, remove blanks
	cleanup_column_names(dataset, {}, True)
	dataset_orig = dataset
	run_print_dataset(dataset)
	features_list = dataset.columns.values.tolist()
	print(features_list)
	# Select subset of explanatory variables from prior information MUST include the target_variable

	features_static = ['sexo', 'nivel_educativo', 'apoe',  'conversion', 'tiempo']
	features_year1 = [s for s in dataset_orig.keys().tolist()  if "visita1" in s]; features_year1.remove('fecha_visita1')
	features_year2 = [s for s in dataset_orig.keys().tolist()  if "visita2" in s]; features_year2.remove('fecha_visita2')
	features_year3 = [s for s in dataset_orig.keys().tolist()  if "visita3" in s]; features_year3.remove('fecha_visita3'); features_year3.remove('act_prax_visita3'), features_year3.remove('act_comp_visita3')
	features_year4 = [s for s in dataset_orig.keys().tolist()  if "visita4" in s]; features_year4.remove('fecha_visita4'); features_year4.remove('act_prax_visita4'), features_year4.remove('act_comp_visita4')
	features_year5 = [s for s in dataset_orig.keys().tolist()  if "visita5" in s]; features_year5.remove('fecha_visita5'); features_year5.remove('act_prax_visita5'), features_year5.remove('act_comp_visita5')
	explanatory_features = features_static + features_year1
	#explanatory_features = None # If None explanatory_features assigned to features_list
	target_variable = 'conversion' # if none assigned to 'conversion'. target_variable = ['visita_1_EQ5DMOV']
	print("Calling to run_variable_selection(dataset, explanatory_features= {}, target_variable={})".format(explanatory_features, target_variable))
	dataset, explanatory_features = run_variable_selection(dataset, explanatory_features, target_variable)
	# dataset with all features including the target and removed NaN
	print ("Features containing NAN values:\n {}".format(dataset.isnull().any()))
	print( "Number of NaN cells in original dataframe:{} / {}, total rows:{}".format(pd.isnull(dataset.values).sum(axis=1).sum(), dataset.size, dataset.shape[0]))
	#ss = dataset.isnull().sum(axis=1)
	#print(" Number of cells with NaNs per Row:\n{}".format(ss[ss==0]))

	dataset.dropna(axis=0, how='any', inplace=True)
	print( "Number of NaN cells in the imputed dataframe: {} / {}, total rows:{}".format(pd.isnull(dataset.values).sum(axis=1).sum(), dataset.size, dataset.shape[0]))
	# (3.2) Transformation (scaling, discretize continuous variables, expand categorical variables)
	# to imput missing values uncomment these 2 lines
	#dataset, Xy_imputed = run_imputations(dataset, type_imput='zero')
	#print("Xy_imputed:\n{}".format(Xy_imputed))	
	# If necessay, run_binarization_features and run_encoding_categorical_features
	# remove duplicated feature names, NOTE conver to set rearrange the order of the features
	#explanatory_features = list(set(explanatory_features))
	unique_explanatory = []
	[unique_explanatory.append(item) for item in explanatory_features if item not in unique_explanatory]
	explanatory_features = []
	explanatory_features = unique_explanatory

	if explanatory_features.index(target_variable) > 0:
		# this is already checked in run_variable_selection
		print("target variable:{} in position:{}".format(target_variable,explanatory_features.index(target_variable)))
	#dataset=dataset.T.drop_duplicates().T
	Xy = dataset[explanatory_features].values
	Xy_scaled, scaler = run_transformations(Xy) # standarize to minmaxscale or others make input normal
	print("Xy scaled dimensions:{} \n {}".format(Xy_scaled.shape, Xy_scaled))
	# (3.3) detect multicollinearity: Plot correlation and graphs of variables
	#convert ndarray to pandas DataFrame
	Xy_df_scaled = pd.DataFrame(Xy_scaled, columns=unique_explanatory)
	X_df_scaled = Xy_df_scaled.drop('conversion',1)
	# corr_X_df = run_correlation_matrix(X_df_scaled, explanatory_features[0:30]) # correlation matrix of features
	corr_Xy_df = run_correlation_matrix(Xy_df_scaled, explanatory_features) # correlation matrix of features and target
	#corr_matrix = corr_df.as_matrix()
	corr_with_target = corr_Xy_df[target_variable]
	#corr_with_target = calculate_correlation_with_target(Xdf_imputed_scaled, target_values) # correlation array of features with the target
	threshold = np.mean(np.abs(corr_Xy_df.as_matrix())) + 1*np.std(np.abs(corr_Xy_df.as_matrix()))
	graph = build_graph_correlation_matrix(corr_Xy_df, threshold, corr_with_target)
	graph_metrics = calculate_network_metrics(graph)
	# print sumary network metrics
	print_summary_network(graph_metrics, nodes=corr_Xy_df.keys().tolist(), corrtarget=corr_with_target)
	
	#(4) Descriptive analytics: plot scatter and histograms
	longit_xy_scatter = ['scd_visita', 'fcsrtlibdem_visita'] #it works for longitudinal
	plot_scatter_target_cond(Xy_df_scaled,longit_xy_scatter, target_variable)
	features_to_plot = ['scd_visita1', 'fcsrtlibdem_visita1'] 
	plot_histogram_pair_variables(dataset, features_to_plot)
	#sp_visita (sobrepeso), depre_(depresion),ansi_,tce_(traumatismo), sue_dia_(duerme dia), sue_noc_(duerme noche), imc_(imc), cor_(corazon)
	#tabac_(fuma), valfelc_(felicidad) 
	longit_pattern = re.compile("^fcsrtlibdem_+visita[1-5]+$") 
	longit_pattern = re.compile("^mmse_+visita[1-5]+$") 
	# plot N histograms one each each variable_visitai
	plot_histograma_one_longitudinal(dataset_orig, longit_pattern)
	#plot 1 histogram by grouping vlalues of one continuous feature 
	plot_histograma_bygroup(dataset_orig, 'mmse_visita1')
	# plot one histogram grouping by the value of the target variable
	plot_histograma_bygroup_target(dataset_orig, 'conversion')
	# plot some categorical features hardcoded inside the function gropued by target
	# categorical_features = ['sexo','nivel_educativo', 'apoe', 'edad']
	plot_histograma_bygroup_categorical(dataset_orig, target_variable)
	# perform statistical tests: ANOVA
	features_to_test = ['scd_visita1']
	target_anova_variable = 'valsatvid_visita1'#'conversion' nivel_educativo' #tabac_visita1 depre_visita1
	run_statistical_tests(Xy_df_scaled,features_to_test, target_anova_variable)
	
	# (5) Dimensionality Reduction
	pca, projected_data = run_PCA_for_visualization(Xy_df_scaled,target_variable, explained_variance=0.7)
	print("The variance ratio by the {} principal compments is:{}, singular values:{}".format(pca.n_components_, pca.explained_variance_ratio_,pca.singular_values_ ))
	
	# (6) Feature Engineering
	formula= build_formula()
	# build design matrix(patsy.dmatrix) and rank the features in the formula y ~ X 
	X_prep = run_feature_ranking(Xy_df_scaled, scaler, formula)
	#Split dataset into train and test
	y = Xy_df_scaled[target_variable].values
	X_features = explanatory_features
	if target_variable in explanatory_features:
		X_features.remove(target_variable)
	X = Xy_df_scaled[X_features].values
	X_train, X_test, y_train, y_test = run_split_dataset_in_train_test(X, y, test_size=0.2)
	######
	naive_bayes_estimator = run_naive_Bayes(X_train, y_train, X_test, y_test)
	compare_against_dummy_estimators(naive_bayes_estimator, X_train, y_train, X_test, y_test)
	pdb.set_trace()
	#####
	# (7) Modelling
	# (7.1) Linear Classifiers
	# 7.1.1 Regression with Lasso normalization. NOT good method for binary classification
	# 7.1.2  (vanilla) Logistic Regression, SVM
	# 7.1.3 Logistic Regression, SVM with SGD training setting the SGD loss parameter to 'log' for Logistic Regression or 'hinge' for SVM SGD
	model = ['LogisticRegression','RandomForestClassifier', 'XGBooster']
	thres_bin = 0.5
	
	# (7.1.1)
	lasso_estimator = regression_Lasso(X_train, y_train, X_test, y_test)
	print("Regression Lasso best score={}".format(lasso_estimator.best_score_))
	calculate_top_features_contributing_class(lasso_estimator, X_features, 10)
	# (7.1.2)a vanilla logistic regression
	y_pred = run_fitmodel(model[0], X_train, y_train, X_test, y_test, thres_bin)
	
	#run_model_evaluation(y_test,y_pred)
	
	# (7.1.2)b LinearSVM
	lsvm_estimator_vanilla, lsvm_estimator_grid = linear_svm_classifier(X_train, y_train, X_test, y_test, X_features)
	#calculate_top_features_contributing_class(lsvm_estimator_vanilla, X_features, 10) error
	# (7.1.3) SGD classifier better for binary classification and can go both Log Reg and SVM
	sgd_estimator = SGD_classifier(X_train, y_train, X_test, y_test, X_features)
	calculate_top_features_contributing_class(sgd_estimator, X_features, 10)
	
	
	# (7.2) NON Linear Classifiers RandomForest and XGBooster http://xgboost.readthedocs.io/en/latest/tutorials/index.html
	## (7.2.1) RandomForestClassifier
	y_pred = run_fitmodel(model[1], X_train, y_train, X_test, y_test)
	#run_model_evaluation(y_test,y_pred)
	#how to evaluate random forest??? 
	# (7.2.2) XGBoost is an implementation of gradient boosted decision trees designed for speed and performance
	y_pred = run_fitmodel(model[2], X_train, y_train, X_test, y_test)
	run_model_evaluation(y_test,y_pred)
	# (7.2.3) Kneighbors classifier
	knn = kneighbors_classifier(X_train, y_train, X_test, y_test)
	compare_against_dummy_estimators(knn, X_train, y_train, X_test, y_test)



	######
	# kneighbors
	#run_naive_Bayes()
	#run_Keras_DN(dataset, X_train, y_train, X_test, y_test)

	#########
	# algebraic topology
	#activations.shape = X_test.shape[0], number of weights
	samples = run_tSNE_analysis(activations, y_test)
	run_TDA_with_Kepler(samples, activations)
	pdb.set_trace()

	
def run_print_dataset(dataset):
	""" run_print_dataset: print information about the dataset, type of features etc
	Args: Pandas dataset
	Output: None"""

	print("dtypes of the Pandas dataframe :\n\n{}".format(dataset.dtypes))
	print(" Number of cells with NaNs per Column:\n{}".format(dataset.isnull().sum()))
	ss = dataset.isnull().sum(axis=1)
	print(" Number of cells with NaNs per Row:\n{}".format(ss[ss==0 ]))
	print("List of rows that contain some NaNs:{}".format(ss[ss>0].index[:].tolist()))
	#for colix in range(dataset.shape[1]):
	#	print(dataset.ix[:,colix].value_counts())
	
def run_variable_selection(dataframe, explanatory_features=None,target_variable=None):
	"""run_variable_selection: select features: explanatory and target. check if target var is in explanatory if not EXIT
	Args: dataset, explanatory_features, target_variable : list of explanatory variables if None assigned all the features dataset.keys()
	target_variable: target feature, if None is assigned inside the function 
	Output: dataframe containing the selected explanatory and target variables
	Example: run_variable_selection(dataset, ['', ''...], 'conversion')
	run_variable_selection(dataset, ['', ''...])
	run_variable_selection(dataset)

	""" 
	if target_variable is None:
		target_variable = 'conversion'
	target_feature = target_variable
	if explanatory_features is None:
		explanatory_features = dataframe.keys().tolist()
	print("Original dataframe number of features: {}".format(len(dataframe.columns)-1))
	#explanatory_features.append(target_variable)
	dataframe = dataframe[explanatory_features]
	if target_variable not in explanatory_features:
		sys.exit('Error! You need to add the target variable in the list of features!!')	
	# remove the object type fields, eg dates and other crappy features
	dataframe  = dataframe.select_dtypes(exclude=['object'])
	print("Dataframe shape after removing object type features:{}".format(dataframe.keys().shape[0]))
	excludefeatures = ['dx_visita1','dx_visita2', 'dx_visita3', 'dx_visita4','dx_visita5']
	if set(excludefeatures).issubset(dataframe.keys().tolist()) is True:
		print("Dropping features:{} \n".format(excludefeatures))
		dataframe.drop(excludefeatures, axis=1, inplace=True)
	print("Selected features after removing excludefeatures: {}".format(dataframe.keys().shape[0]))
	print("Number of Observations: {}".format(dataframe.shape[0]))
	print("Number of selected dataframe explanatory features + target: {}".format(len(dataframe.columns)))
	print("Target variable:       '{}' -> '{}'".format('conversion', 'target'))
	print(" explanatory features:  {}".format(dataframe.keys()))
	return dataframe, explanatory_features 

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

def plot_histograma_one_longitudinal(df, longit_pattern=None):
	""" plot_histogram_pair_variables: histograma 1 for each year 
	Args: Pandas dataframe , regular expression pattern eg mmse_visita """

	longit_status_columns = [ x for x in df.columns if (longit_pattern.match(x))]
	df[longit_status_columns].head(10)
	# plot histogram for longitudinal pattern
	fig, ax = plt.subplots(2,3)
	fig.set_size_inches(15,5)
	fig.suptitle('Distribution of scd 5 visits')
	for i in range(len(longit_status_columns)):
		row,col = int(i/3), i%3
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

def plot_histograma_bygroup(df, label=None):
	""" plot_histograma_bygroup: plot one histogram grouping by values of numeric feature label"""
	# plot_histograma_bygroup: histogram group by label
	print("Plotting histogram in log scale grouping by:{}".format(label))
	df[label].describe()
	fig = plt.figure()
	fig.set_size_inches(20,5)
	ax = fig.add_subplot(111)
	grd = df.groupby([label]).size()
	ax.set_yscale("log")
	ax.set_xticks(np.arange(len(grd)))
	ax.set_xlabel(label + ': values ')
	fig.suptitle(label)
	#ax.set_xticklabels(['%d'  %i for i in d.index], rotation='vertical')
	p = ax.bar(np.arange(len(grd)), grd, color='orange')
	plt.show()

def plot_histograma_bygroup_target(df, target_label=None):
	""" plot_histograma_bygroup_target : histogram grouped by the target (binary) """
	#plot histogram target feature
	fig = plt.figure()
	fig.set_size_inches(5,5)
	ax = fig.add_subplot(111)
	ax.set_xlabel('number of subjects ')
	ax.set_ylabel('conversion y/n')
	grd2 = df.groupby([target_label]).size()
	print("conversion subjects are {}% out of {} observations".format(100* grd2[1]/(grd2[1]+grd2[0]), grd2[1]+grd2[0]))
	p = grd2.plot(kind='barh', color='orange')
	fig.suptitle('number of conversors vs non conversors')
	plt.show()

def plot_histograma_bygroup_categorical(df, target_variable=None):
	# target related to categorical features
	#categorical_features = ['sexo','nivel_educativo', 'apoe', 'edad']
	df['sexo'] = df['sexo'].astype('category').cat.rename_categories(['M', 'F'])
	df['nivel_educativo'] = df['nivel_educativo'].astype('category').cat.rename_categories(['~Pr', 'Pr', 'Se', 'Su'])
	df['apoe'] = df['apoe'].astype('category').cat.rename_categories(['No', 'Het', 'Hom'])
	df['edad_visita1_cat'] = pd.cut(df['edad_visita1'], range(0, 100, 10), right=False)
	#in absolute numbers
	fig, ax = plt.subplots(1,4)
	fig.set_size_inches(20,5)
	fig.suptitle('Conversion by absolute numbers, for various demographics')
	d = df.groupby([target_variable, 'sexo']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0])
	d = df.groupby([target_variable, 'nivel_educativo']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[1])
	d = df.groupby([target_variable, 'apoe']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[2])
	d = df.groupby([target_variable, 'edad_visita1_cat']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[3])
	#in relative numbers
	fig, ax = plt.subplots(1,4)
	fig.set_size_inches(20,5)
	fig.suptitle('Conversion by relative numbers, for various demographics')
	d = df.groupby([target_variable, 'sexo']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0])
	d = df.groupby([target_variable, 'nivel_educativo']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1])
	d = df.groupby([target_variable, 'apoe']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[2])
	d = df.groupby([target_variable, 'edad_visita1_cat']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[3])
	plt.show()

def build_formula():
	""" build formula to be used for  run_feature_slection. 'C' for categorical features
	Args: None
	Outputs: formula"""
	#formula = 'conversion ~ '; formula += 'C(sexo) + C(nivel_educativo) + C(apoe)'; 
	formula = 'conversion ~ '; formula += 'sexo + nivel_educativo + apoe'; 
	formula += ' + ' + ' + '.join(selcols('scd_visita',1,1)); formula += ' + ' + ' + '.join(selcols('gds_visita',1,1)); formula += ' + ' + ' + '.join(selcols('eqm10_visita',1,1));formula += ' + ' + ' + '.join(selcols('eqm09_visita',1,1));
	formula += ' + ' + ' + '.join(selcols('hta_visita',1,1)); formula += ' + ' + ' + '.join(selcols('sue_con_visita',1,1));formula += ' + ' + ' + '.join(selcols('sue_rec_visita',1,1))
	formula += ' + ' + ' + '.join(selcols('lipid_visita',1,1));
	formula += ' + ' + ' + '.join(selcols('sdvive_visita',1,1));formula += ' + ' + ' + '.join(selcols('sdhijos_visita',1,1));formula += ' + ' + ' + '.join(selcols('valsatvid_visita',1,1))
	formula += ' + ' + ' + '.join(selcols('ejfre_visita',1,1));formula += ' + ' + ' + '.join(selcols('a13_visita',1,1));formula += ' + ' + ' + '.join(selcols('a10_visita',1,1));
	formula += ' + ' + ' + '.join(selcols('rsoled_visita',1,1));formula += ' + ' + ' + '.join(selcols('relaocio_visita',1,1));formula += ' + ' + ' + '.join(selcols('preocupacion_visita',1,1))
	formula += ' + ' + ' + '.join(selcols('act_depre_visita',1,1));formula += ' + ' + ' + '.join(selcols('act_ansi_visita',1,1));formula += ' + ' + ' + '.join(selcols('act_orie_visita',1,1));
	formula += ' + ' + ' + '.join(selcols('card_visita',1,1));formula += ' + ' + ' + '.join(selcols('sp_visita',1,1));formula += ' + ' + ' + '.join(selcols('tabac_visita',1,1));
	formula += ' + ' + ' + '.join(selcols('imc_visita',1,1));
	formula += ' + ' + ' + '.join(selcols('fcsrtrl1_visita',1,1));formula += ' + ' + ' + '.join(selcols('fcsrtrl3_visita',1,1));formula += ' + ' + ' + '.join(selcols('fcsrtlibdem_visita',1,1));
	return formula


def selcols(prefix, a=1, b=5):
	""" selcols: return list of str of longitudinal variables
	Args:prefix name of the variable
	a: initial index year, b last year
	Output: list of feature names
	Example: selcols('scd_visita',1,3) returns [scd_visita,scd_visita2,scd_visita3] """
	return [prefix+str(i) for i in np.arange(a,b+1)]

def run_imputations(dataset, type_imput=None):
	""" run_imputations: datasets with missign values are incompatible with scikit-learn 
	estimators which assume that all values in an array are numerical, and that all have and hold meaning.
	Args: dataset is a Pandas dataframe, type_imput='zero', 'mean', 'median', 'most_frequent'
	http://scikit-learn.org/stable/modules/preprocessing.html
	Output:dataset datframe modified for 'zero' strategy , nmnodified for nonzero strategy and Xy_train_imputed : ndarray modief with the imputed method
	"""
	from sklearn.preprocessing import Imputer
	print( "Number of rows in the dataframe:{}".format(dataset.shape[0]))
	print ("Features containing NAN values:\n {}".format(dataset.isnull().any()))
	print( "Number of NaN cells in original dataframe:{}".format(np.count_nonzero(np.isnan(dataset.values))))
	if type_imput is None or type_imput is 'zero':
		print("Imputations replacing NaNs for 0")
		dataset.fillna(0, inplace=True)
		Xy_train_imputed = dataset.values
	else:
		#print(dataset) 
		imp = Imputer(missing_values='NaN', strategy=type_imput, axis=1)
		imp.fit(dataset)
		Xy_train_imputed = imp.transform(dataset)
		print( "Number of NaN cells in imputed dataframe, strategy:{}, {}".format(type_imput, np.count_nonzero(np.isnan(Xy_train_imputed))))
	return dataset, Xy_train_imputed

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
	Args: dataset:ndarray
	Output: X_train_minmax transformed by scaling ndarray. 
	Note: If NaN values doesnt work """

	# feature scaling scaling individual samples to have unit norm
	# the quick way to normalize : X_scaled = preprocessing.scale(X_train), fit_transform is practical if we are going to train models 
	# To scale [-1,1] use MaxAbsScaler(). MinMaxScaler formula is std*(max-min) + min
	scaler = preprocessing.MinMaxScaler()
	#The same instance of the transformer can then be applied to some new test data unseen during the fit call:
	# the same scaling and shifting operations will be applied to be consistent with the transformation performed on the train data
	X_train_minmax = scaler.fit_transform(dataset)
	print("Orignal ndarray \n {}".format(dataset))
	#print("X_train_minmax \n {}".format(X_train_minmax))
	return X_train_minmax, scaler


def run_split_dataset_in_train_test(X,y,test_size=None):
	""" run_split_dataset_in_train_test """
	if test_size is None:
		test_size = 0.2
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
	return X_train, X_test, y_train, y_test

def run_feature_ranking(df, scaler, formula=None):
	""" run_feature_ranking(X) : builds the design matrix for feature ranking (selection)
	Args: panas dataframe scaled and normalized
	Outputs: design matrix for given formula"""
	# Construct a single design matrix given a formula_ y ~ X
	if formula is None:
		formula = 'conversion ~ '
		# original features
		#formula += 'C(sexo) + C(nivel_educativo) +  C(apoe) '
		formula += 'sexo + nivel_educativo + apoe '
		formula += '+' + '+'.join(selcols('scd_visita',1,1))
	print("The formula is:", formula)
	# patsy dmatrices, construct a single design matrix given a formula_like and data.
	y, X = dmatrices(formula, data=df, return_type='dataframe')
	y = y.iloc[:, 0]
	# feature scaling
	if scaler is None:
		scaler = preprocessing.MinMaxScaler()
		scaler.fit(X)
	""" select top features and find top indices from the formula  """
	nboffeats = 12
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
	return X_prep
	# model selection
	#X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.3, random_state=42)
	#return X_prep, X_train, X_test, y_train, y_test

def run_PCA_for_visualization(Xy_df, target_label, explained_variance=None):
	""" run_PCA_for_visualization Linear dimensionality reduction using Singular Value Decomposition 
	of the data to project it to a lower dimensional space. 
	http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html"""
	from sklearn.decomposition import PCA
	from mpl_toolkits.mplot3d import Axes3D
	# Choose number of components
	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(1,1,1)
	pca = PCA().fit(Xy_df.values)
	cumvar = np.cumsum(pca.explained_variance_ratio_)
	if explained_variance is None:
		explained_variance = 0.7
	optimal_comp = np.where(cumvar >explained_variance )[0][0]
	print("With at least {} components we explain {}'%'' of the {} dimensional input data".format(optimal_comp,explained_variance,Xy_df.values.shape[1] ))
	plt.plot(cumvar)
	plt.xlabel('number of components')
	plt.ylabel('cumulative explained variance')
	
	#Visualize
	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(1,1,1)
	#optimal_comp = 2
	if optimal_comp > 2:
		ax = Axes3D(fig)
	pca = PCA(n_components=optimal_comp)
	projected = pca.fit_transform(Xy_df.values)
	print("Dimensions original input:{} \n".format(Xy_df.values.shape))
	print("Dimensions projected input:{} \n".format(projected.shape))
	print("PCA {} components_:{} \n".format(pca.n_components_, pca.components_))
	print("PCA explained variance ratio:{} \n".format(pca.explained_variance_ratio_))
	# plot the principal components
	targets = Xy_df[target_label].unique().astype(int)
	if optimal_comp > 2:
		plt.scatter(projected[:, 0], projected[:, 1],projected[:, 2], c=Xy_df[target_label].astype(int), edgecolor='none', alpha=0.6)
	else:
		plt.scatter(projected[:, 0], projected[:, 1], c=Xy_df[target_label].astype(int), edgecolor='none', alpha=0.6)
	ax.grid()
	ax.set_xlabel('PC 1', fontsize = 10)
	ax.set_ylabel('PC 2', fontsize = 10)
	if optimal_comp > 2:
		ax.set_zlabel('PC 3', fontsize = 10)
	msgtitle = str(pca.n_components_) + ' components PCA'
	ax.set_title(msgtitle, fontsize = 10)
	#Noise filtering https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
	return pca, projected

def compare_against_dummy_estimators(estimator1, X_train, y_train, X_test, y_test):
	""" compare_against_dummy_estimators: When doing supervised learning, a simple sanity check consists of comparing one's
	estimator against simple rules of thumb. DummyClassifier implements such strategies(stratified
	most_frequent, prior, uniform, constant). Used for imbalanced datasets
	Args: fitted estimator1 eg clf = SVC(kernel='linear',..).fit(X_train, y_train)
	X_train, X_test, y_train, y_test
	Outputs: """
	print("Estimator {}\n".format(estimator1))
	print("Score ={} \n".format(estimator1.score(X_test, y_test)))
	dummy_strategy = 'uniform'# 'most_frequent'  'constant'. constant=1
	estimator_dummy = DummyClassifier(strategy=dummy_strategy, random_state=0)
	estimator_dummy.fit(X_train, y_train)
	# see if estimator_1 does much better than the dummy, if it doesnt we can change for example the kernel
	#if estimator_1 is SVC and see of the score compreed to the dummy is better
	print("Score of Dummy estimator={}".format(estimator_dummy.score(X_test, y_test)))
	# comapre scores estimator_1 vs estimator_dummy
	estimator_dummy = DummyClassifier(strategy='constant', random_state=0, constant=1)
	estimator_dummy.fit(X_train, y_train)
	print("Score of Dummy constant estimator (always 1)={}".format(estimator_dummy.score(X_test, y_test)))
	estimator_dummy = DummyClassifier(strategy='constant', random_state=0, constant=0)
	estimator_dummy.fit(X_train, y_train)
	print("Score of Dummy constant estimator (always 0)={}".format(estimator_dummy.score(X_test, y_test)))
	

def run_model_evaluation(y_true, y_pred):
	""" """
	print("Accuracy score={}".format(accuracy_score(y_true, y_pred))) 
	print("Cohen kappa score={} (expected kcohen < accuracy)".format(cohen_kappa_score(y_true, y_pred))) 
	print("Precision (not to label + when is - (do not miss sick patients) score={}".format(precision_score(y_true, y_pred))) 
	print("Recall (find all the positives) score={}".format(recall_score(y_true, y_pred))) 
	print("F1(weighted harmonic mean of precision and recall) score={}".format(f1_score(y_true, y_pred))) 
	target_names = ['class 0', 'class 1']
	print(classification_report(y_true, y_pred, target_names=target_names))
	#matthews_corrcoef a balance measure useful even if the classes are of very different sizes.
	print("The matthews_corrcoef(+1 is perfect prediction , 0 average random prediction and -1 inverse prediction)={}. \n ".format(matthews_corrcoef(y_true, y_pred))) 

def run_fitmodel(model, X_train, y_train, X_test, y_test, threshold=None):
	""" fit the model (LR, random forest others) and plot the confusion matrix and RUC 
	Args:model:string,X_train, X_test, y_train, y_test , threshold for binarize prediction
	Outputs: predictions (y_test_pred)"""
	if threshold is None:
		threshold = 0.5
	if model == 'LogisticRegression':
		# Create logistic regression object
		regr = linear_model.LogisticRegression()
		# Train the model using the training sets
		regr.fit(X_train, y_train)
		# model prediction
		y_train_pred = regr.predict_proba(X_train)[:,1]
		y_test_pred = regr.predict_proba(X_test)[:,1]
		
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
		y_test_pred = [round(value) for value in y_pred]
		# evaluate predictions
		accuracy = accuracy_score(y_test, y_test_pred)
		print("Accuracy XGBooster classifier: %.2f%%" % (accuracy * 100.0))
		plt.show()
	# return ground truth and predictions	
	return y_test_pred

def plot_scatter_target_cond(df, preffix_longit_xandy, target_variable=None):	
	"""scatter_plot_pair_variables_target: scatter dataframe features and coloured based on target variable values
	Args:df: Pandas dataframe, preffix_longitx:preffix of longitudinal variable to plot in X axiseg dx_visita, 
	preffix_longit_y:preffix of longitudinal variable to plot in Y axis eg audi_visita, target_variable: target feature contained in dataframe 
	Example: scatter_plot_target_cond(df, 'conversion')"""
	
	# scatter_plot_target_cond: average and standard deviation of longitudinal status
	
	#selcols(prefix,year ini,year end), if we dont have longitudinal set to 1,1
	df['longit_x_avg'] = df[selcols(preffix_longit_xandy[0],1,1)].mean(axis=1)
	df['longit_y_avg'] = df[selcols(preffix_longit_xandy[1],1,1)].mean(axis=1)
	#df['longit_x_std'] = df[selcols(preffix_longit_x,1,1)].std(axis=1)
	def_no = df[df[target_variable]==0]
	def_yes = df[df[target_variable]==1]
	print("Rows with 0 target={} and with 1 target={}".format(def_no.shape[0],def_yes.shape[0] ))
	#target_noes = df[target_variable].where(df[target_variable] == 0).dropna()
	#target_yes = df[target_variable].where(df[target_variable] == 1).dropna()
	fig,ax = plt.subplots(1,2)
	#ax.set_xlim([min(def_no['longit_y_avg'],def_yes['longit_y_avg']),max(def_no['longit_y_avg'],def_yes['longit_y_avg'])])
	fig.set_size_inches(10,10)
	x_lab = preffix_longit_xandy[0]
	y_lab = preffix_longit_xandy[1]
	ax[0].set_title(target_variable)
	ax[0].set_ylabel(y_lab)
	ax[0].set_ylimit = (0,1)
	ax[0].set_xlabel(x_lab)
	ax[0].scatter(def_yes['longit_x_avg'], def_yes['longit_y_avg'],color='r',alpha=0.1)
	ax[1].set_title('no ' + target_variable)
	ax[1].set_ylimit = (0,1)
	ax[1].set_xlabel(x_lab)
	ax[1].scatter(def_no['longit_x_avg'], def_no['longit_y_avg'],color='b',alpha=0.1)
	plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
	plt.show()
	#p = ax[0,0].semilogy(def_yes['scd_avg'].dropna(inplace=False), def_yes['stai_avg'].dropna(inplace=False), 'bo', markersize=5, alpha=0.1)

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

def run_Keras_DN(dataset, X_train, y_train, X_test, y_test):
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

def linear_svm_classifier(X_train, y_train, X_test, y_test, features=None):
	""" linear_svm_classifier: is the linear classifier with the maximum margin"""

	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	# SVM model, cost and gamma parameters for RBF kernel. out of the box
	svm = SVC(cache_size=1000, kernel='rbf')
	svm.fit(X_train, y_train)
	# test
	y_pred = [int(a) for a in svm.predict(X_test)]
	num_correct = sum(int(a == ye) for a, ye in zip(y_pred, y_test))
	print("Vanilla classifier using linear SVM: %s of %s values correct." % (num_correct, len(y_test)))
	# plot learning curves
	kfolds = StratifiedKFold(5)
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	#Generate indices to split data into training and test set.
	cv = kfolds.split(X_all,y_all)
	title ='Learning Curves vanilla linearSVM'
	plot_learning_curve(svm, title, X_all, y_all, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

	print("Exhaustive search for SVM parameters to improve the out-of-the-box performance of vanilla SVM.")
	parameters = {'C':10. ** np.arange(5,10), 'gamma':2. ** np.arange(-5, -1)}
	grid = GridSearchCV(svm, parameters, cv=5, verbose=3, n_jobs=2)
	grid.fit(X_train, y_train)
	print(grid.best_estimator_)
	print("LVSM GridSearchCV. The best alpha is:{}".format(grid.best_params_)) 
	print("Linear SVM accuracy of the given test data and labels={} ", grid.score(X_test, y_test))
	return svm, grid

def run_naive_Bayes(X_train, y_train, X_test, y_test):
	""" run_naive_Bayes"""

	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	kfolds = StratifiedKFold(5)
	cv = kfolds.split(X_all,y_all)
	cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
	estimator = GaussianNB()
	estimator.fit(X_train,y_train)
	# test
	y_pred = [int(a) for a in estimator.predict(X_test)]
	num_correct = sum(int(a == ye) for a, ye in zip(y_pred, y_test))
	print("Baseline classifier using Naive Bayes: %s of %s values correct." % (num_correct, len(y_test)))
	# plot learning curve
	#http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
	plot_learning_curve(estimator, 'Naive Bayes classifier', X_all, y_all, cv=cv, n_jobs=4)	
	return estimator

def SGD_classifier(X_train, y_train, X_test, y_test, features=None):
	"""SGD_classifier: Stochastic Gradient Descent classifier. Performs Log Regression and/or SVM (loss parameter) with SGD training 
	Args:X_train,y_train,X_test,y_test
	Output: SGD fitted estimator"""
	#find an opimum value of 'alpha' by either looping over different values of alpha and evaluating the performance over a validation set
	#use gridsearchcv
	from sklearn.linear_model import SGDClassifier
	tuned_parameters = {'alpha': [10 ** a for a in range(-6, -2)]}
	#class_weight='balanced' addresses the skewness of the dataset in terms of labels
	# loss='hinge' LSVM,  loss='log' gives logistic regression, a probabilistic classifier
	# ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’.
	clf = GridSearchCV(SGDClassifier(loss='log', penalty='elasticnet',l1_ratio=0.15, n_iter=5, shuffle=True, verbose=False, n_jobs=10, average=False, class_weight='balanced')
                  , tuned_parameters, cv=10, scoring='f1_macro')
	#now clf is the best classifier found given the search space
	clf.fit(X_train, y_train)
	print("LVSM GridSearchCV SGDClassifier. The best alpha is:{}".format(clf.best_params_)) 
	print("The classifier accuracy of the given test data and labels={}".format(clf.score(X_train,y_train)))
	return clf

def calculate_top_features_contributing_class(clf, features, numbertop=None):
	""" calculate_top_features_contributing_class: print the n features tcontributing the most to class labels for a fitted estimator.
	Args: estimator, features lits and number of features"""
	if numbertop is None:
		numbertop = 10
	print("The estimator is:{} \n".format(clf))
	for i in range(0, clf.best_estimator_.coef_.shape[0]):
		toplist = np.argsort(clf.best_estimator_.coef_[i])[-numbertop:]
	print("the top {} features indices contributing to the class labels are:{}".format(numbertop, toplist))
	if features is not None:
		print("\tand the top {} features labels contributing to the class labels are:{} \n".format(numbertop, operator.itemgetter(*toplist)(features)))

def kneighbors_classifier(X_train, y_train, X_test, y_test):
	""" kneighbors_classifier : KNN is non-parametric, instance-based and used in a supervised learning setting. Minimal training but expensive testing.
	KNN is used as a benchmark for more complex classifiers such as (ANN) (SVM).
	KNN classifier is also a non parametric ( it makes no explicit assumptions about the functional form of h:X->y) 
	this is protection against for example assume data are Gaussian and they are not.
	and instance-based (algorithm doesnt explicitely learn a model) so it chooses to memorize the training instances 
	which are subsequently used as “knowledge” for the prediction phase, that is, only when a query to our database is made
	(i.e. when we ask it to predict a label given an input), will the algorithm use the training instances to spit out an answer
	https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
	CONS: Note the rigid dichotomy between KNN and the more sophisticated Neural Network which has a lengthy training phase albeit a very fast testing phase.
	Furthermore, KNN can suffer from skewed class distributions. For example, if a certain class is very frequent in the training set,
	it will tend to dominate the majority voting of the new example (large number = more common). 
	Finally, the accuracy of KNN can be severely degraded with high-dimension data because there is little difference between the nearest and farthest neighbor.
	Args:  
	Output: knn estimator"""

	#Randomly dividing the training set into k groups (k and k_hyper are nothinmg to do with each other), or folds, of approximately equal size.
	# The first fold is treated as a validation set, and the method is fit on the remaining k−1 folds.
	#The misclassification rate is then computed on the observations in the held-out fold. 
	#This procedure is repeated k times; each time, a different group of observations is treated as a validation set. 
	#This process results in k estimates of the test error which are then averaged out
	#performing a 10-fold cross validation on our dataset using a generated list of odd K’s ranging from 1 to 50.
	# creating odd list of K for KNN
	myList = list(range(1,50)) #odd to avoid tie of points
	# subsetting just the odd ones
	neighbors = filter(lambda x: x % 2 != 0, myList)
	# empty list that will hold cv scores
	# perform 10-fold cross validation
	cv_scores = [] # list with x validation scores
	for k in neighbors:
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
		cv_scores.append(scores.mean())
	MSE = [1 - x for x in cv_scores]
	# determining best k
	optimal_k = neighbors[MSE.index(min(MSE))]
	# instantiate learning model
	knn = KNeighborsClassifier(n_neighbors=optimal_k)
	knn.fit(X_train, y_train)
	# predict the response
	y_pred = knn.predict(X_test)
	# evaluate accuracy
	print("KNN classifier with optimal k={}, accuracy={}".format(optimal_k, accuracy_score(y_test, y_pred)))
	fig, ax = plt.subplots(1, 1, figsize=(6,9))
	# plot misclassification error vs k
	ax.plot(neighbors, MSE)
	ax.set_xlabel('Number of Neighbors K')
	ax.set_ylabel('Misclassification Error')
	ax.set_title('KNN classifier')
	plt.show()
	return knn

def regression_Lasso(X_train, y_train, X_test, y_test):
	""" logistic regression, answer two points: what is the baseline prediction of disease progression and 
	which independent variables are important facors for predicting disease progression.
	VERY SUBOPTIMAL method for binary classification (scores 0.1~0.2)
	Args:(X_train,y_train,X_test,y_test, features, target_label
	Output: GridSearchCV Lasso estimator """
	# Lasso normal linear regression with L1 regularization (minimize the number of features or predictors int he model)
	from sklearn.linear_model import Lasso
	from sklearn import linear_model

	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	# define the model and the hyperparameter alpha (controls the stricteness of the regularization)
	lasso = Lasso(random_state=0)
	alphas = np.logspace(-6, -0.5, 10)
	# estimator with our model, in this case a grid search of model of Lasso type
	# GridSearchCV does exhaustive search over specified parameter values for an estimator
	# fit of an estimator on a parameter grid and chooses the parameters to maximize the cross-validation score
	estimator = GridSearchCV(lasso, dict(alpha=alphas))
	# take train set and learn a group of Lasso models by varying the value of the alpha hyperparameter.
	#Best possible score is 1.0, lower values are worse. Unlike most other scores
	#The score method of a LassoCV instance returns the R-Squared score, which can be negative, means performing poorly
	#Estimator score method is a default evaluation criterion for the problem they are designed to solve
	#By default, the GridSearchCV uses a 3-fold cross-validation
	estimator.fit(X_train, y_train)
	print("Lasso estimator cv results:{}".format(sorted(estimator.cv_results_.keys())))
	print("Mean cross-validated score of the best estimator:{}".format(estimator.best_score_))
	print("Estimator was chosen by the search(highest score):{} ".format(estimator.best_estimator_))
	print("Calling to estimator.predict with the best estimator parameters...")
	y_pred = estimator.predict(X_test)
	
	scores = estimator.score(X_test,y_test)
	print("Scores:{}".format(scores))
	return estimator
	


def run_load_csv(csv_path = None):
	""" load csv database, print summary of data"""
	if csv_path is None:
		csv_path = "/Users/jaime/vallecas/data/scc/sccplus-24012018.csv"
		csv_path = "/Users/jaime/vallecas/data/scc/SCDPlus_IM_27022018.csv"
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

def run_statistical_tests(dataset, feature_label, target_label):
	#chi_square_of_df_cols(dataset, feature_label[0], feature_label[1])
	anova_test(dataset, feature_label, target_label)



def plot_histogram_pair_variables(dataset, feature_label=None):
	"""" histogram_plot_pair_variables: plot 2 hisotgram one for each variables
	Args: dataset Pandas dataframe and festure_label list of variables with at least two contained in the dataset
	Example: histogram_plot_pair_variables(dataset, ['var1, 'var2']) """
	
	nb_of_bins = 15
	fig, ax = plt.subplots(1, 2)
	# features = [dataset.Visita_1_MMSE, dataset.years_school ]
	features = [dataset[feature_label[f]] for f in range(len(feature_label))]
	#plot two first features
	ax[0].hist(features[0].dropna(), nb_of_bins, facecolor='red', alpha=0.5, label="scc")
	ax[1].hist(features[1].dropna(), nb_of_bins, facecolor='red', alpha=0.5, label="scc")
	#fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
	#ax[0].set_ylim([0, dataset.ipm.shape[0]])
	#bubble_df.plot.scatter(x='purchase_week',y='price',c=bubble_df['enc_uclass'],s=bubble_df['total_transactions']*10)
	#plt.title('Purchase Week Vs Price Per User Class Based on Tx')
	ax[0].set_xlabel(feature_label[0])
	ax[0].set_ylabel("Frequency")
	ax[1].set_xlabel(feature_label[1])
	#ax[1].set_ylabel("Frequency")
	msgsuptitle = feature_label[0] + " and " + feature_label[1]
	fig.suptitle(msgsuptitle)
	plt.show()

def build_graph_correlation_matrix(corr_df, threshold=None, corr_target=None):
	""" build_graph_correlation_matrix: requires package pip install pygraphviz
	Args:A is the dataframe correlation matrix
	Output:None
	"""
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
		# plot label and correlation with the target 
		if corr_target is not None:
			labels[idx] = val+ ' ' + `'{0:.2g}'.format(corr_target[idx])`

	G = nx.from_numpy_matrix(A)
	pos=nx.spring_layout(G)
	nx.draw_networkx_nodes(G, pos)
	nx.draw_networkx_edges(G, pos)
	nx.draw_networkx_labels(G, pos, labels, font_size=9)
	# plt.title('Binary Graph from correlation matrix{}'.format(node_names))
	plt.title('Binary Graph, threshold={0:.3g}'.format(threshold))
	return G

def print_summary_network(Gmetrics, nodes=None, corrtarget=None):
	"""print_summary_network: print summary of the graph metrics (clustering, centrality )
	Args: Gmetrics: dictionar keys() are the metrics and values the values, nodes: list of node labels, corrtarget:corrleation value with the target of each node
	Output: None """
	
	nbitems = 6 # top nbitems nodes
	ignored_list = ['communicability','degree', 'density','is_connected','degree_assortativity_coefficient','degree_histogram','estrada_index','number_connected_components','transitivity']
	ignored =set(ignored_list)
	for key, value in Gmetrics.iteritems():
		if key not in ignored:
			print("Summary for: {} : best {} nodes\n".format(key, nbitems))
			sorted_value = sorted(value.items(), key=operator.itemgetter(1))
			if type(Gmetrics[key]) is dict:
				for labeli in sorted_value[-nbitems:]:
					nodeindex = labeli[0]
					metricvalue = labeli[1]
					labelname =  nodes[nodeindex]
					print("		{} at node: {} = {}, correlation with target={}".format(key, labelname, metricvalue,corrtarget[labelname]))
			else:
				print("{} = {} \n".format(key, Gmetrics[key]))
		else:
			print("Skipping {} \n".format(key))
	print("Density ={} ".format(Gmetrics['density']))
	print("Is connected ={} ".format(Gmetrics['is_connected']))
	print("degree_assortativity_coefficient ={}".format(Gmetrics['degree_assortativity_coefficient']))
	print("estrada_index ={} ".format(Gmetrics['estrada_index']))
	print("number_connected_components ={}, Normalized ncc/totalnodes={}".format(Gmetrics['number_connected_components'], Gmetrics['number_connected_components']/float(len(nodes))))
	print("transitivity ={} ".format(Gmetrics['transitivity']))
		

def calculate_network_metrics(G):
	""" calculate_network_metrics: study the netwok properties
	Args:G networkx graph object
	Output: dictionary with network metrics"""

	G_metrics = {}
	#info(G[, n]) 	Print short summary of information for the graph G or the node n.
	print("Graph directed:{}, characteristics:\n{}".format(nx.is_directed(G), nx.info(G)))
	# Degree
	degree_G = nx.degree(G); G_metrics["degree"] = degree_G
	#a list of the frequency of each degree value.
	freq_degree = nx.degree_histogram(G); G_metrics["degree_histogram"] = freq_degree
	density_G = nx.density(G); G_metrics["density"] = density_G
	# Clustering
	nb_triangles = nx.triangles(G); G_metrics["triangles"] = nb_triangles
	#fraction of all possible triangles present in G.
	frac_triangles = nx.transitivity(G); G_metrics["transitivity"] = frac_triangles
	#clustering coefficient for nodes.
	clustering_coeff = nx.clustering(G); G_metrics["clustering"] = clustering_coeff
	#clustering_avg = nx.average_clustering(G); G_metrics["average_clustering"] = clustering_avg
	#clique_G = nx.max_clique(G) ; G_metrics["max_clique"] = clique_G
	# Assortativity degree_assortativity_coefficient(G[, x, y, ...]) 	Compute degree assortativity of graph.
	degassor_coeff = nx.degree_assortativity_coefficient(G);  G_metrics["degree_assortativity_coefficient"] = degassor_coeff
	#average degree connectivity of graph
	degconn_avg = nx.average_degree_connectivity(G); G_metrics["average_degree_connectivity"] = degconn_avg
	degconn_k_avg = nx.k_nearest_neighbors(G); G_metrics["k_nearest_neighbors"] = degconn_k_avg
	# Centrality
	centrality_deg = nx.degree_centrality(G); G_metrics["degree_centrality"] = centrality_deg
	# Compute the shortest-path betweenness centrality for nodes.
	centrality_clo = nx.closeness_centrality(G); G_metrics["closeness_centrality"] = centrality_clo
	# shortest-path betweenness centrality for nodes. 	
	centrality_btw = nx.betweenness_centrality(G); G_metrics["betweenness_centrality"] = centrality_btw
	# communicability between all pairs of nodes in G.
	comunica = nx.communicability(G); G_metrics["communicability"] = comunica
	#In chemical graph theory, is a topological index of protein folding
	estrada_idx = nx.estrada_index(G); G_metrics["estrada_index"] = estrada_idx
	#dispersion(G[, u, v, normalized, alpha, b, c]) 	Calculate dispersion between u and v in G.
	
	# Connectivity
	is_conn_G = nx.is_connected(G); G_metrics["is_connected"] = is_conn_G
	# Distance measures
	if is_conn_G is True:
		#eccentricity(G[, v, sp]) 	Return the eccentricity of nodes in G.
		ecc_G = nx.eccentricity(G); G_metrics["eccentricity"] = ecc_G
		#center(G[, e]) 	Return the center of the graph G.
		center_G = nx.center(G); G_metrics["center"] = center_G
		#diameter(G[, e]) 	Return the diameter of the graph G.
		diam_G = nx.diameter(G); G_metrics["diameter"] = diam_G
		#periphery(G[, e]) 	Return the periphery of the graph G.
		peri_G = nx.periphery(G); G_metrics["periphery"] = peri_G
		#radius(G[, e]) 	Return the radius of the graph G.
		rad_G = nx.radius(G); G_metrics["radius"] = rad_G
	ncc_G = nx.number_connected_components(G); G_metrics["number_connected_components"] = ncc_G
	# these are only for directed graph
	#is_strongly_conn_G = nx.is_strongly_connected(G); G_metrics["is_strongly_connected"] = is_strongly_conn_G 
	#ncc_strongly_G = nx.number_strongly_connected_components(G); G_metrics["number_strongly_connected_components"] = ncc_strongly_G
	#is_weakly_conn_G = nx.is_weakly_connected(G); G_metrics["is_weakly_connected"] = is_weakly_conn_G
	#ncc_weakly_G = nx.number_weakly_connected_components(G); G_metrics["number_weakly_connected_components"] = ncc_weakly_G

	return G_metrics

def run_correlation_matrix(dataset,feature_label=None):
	""" run_correlation_matrix: calculate the correlation matrix and plot sns map with correlation matrix. feature_label MUST be a subset of the dataset.keys()
	Args: dataset pandas dataframe, feature_label list of features of interest, if None calculate corr with all features in te dataframe
	Output: DataFrame containing the correlation matrix 
	Example: run_correlation_matrix(dataset) return matrix all with all in dataset
	run_correlation_matrix(dataset, ['a','b']) return 2x2 matrix both labels must be in dataset.keys()"""
	# if set(feature_label) > set(dataset.keys().tolist()):
	# 	warnings.warn('The list of features is not contained in the dataframe, trying to fix this by removing the target variable')
	# 	feature_label.remove('conversion')
	# 	if set(feature_label) <= set(dataset.keys().tolist()):
	# 		print("Removed 'conversion' in the features list, the function run_correlation_matrix can continue now...")
	# 	#sys.exit('Error! features list need to be included in the dataset!! Check that the target variable is present/absent in both')
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
	g = sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values, vmin =-1, vmax=1, center=0,annot=False)
	#ax.set_xlabel("Features")
	#ax.set_ylabel("Features")
	ax.set_title(cmethod + " correlation")
	#g.set(xticklabels=[])
	g.set_xticklabels(feature_label, fontsize=4)
	g.set_yticklabels(feature_label, rotation=30, fontsize=9)
	plt.show()
	return corr

def chi_square_of_df_cols(df, col1, col2):
	df_col1, df_col2 = df[col1], df[col2]
	obs = np.array(df_col1, df_col2 ).T

def anova_test(df, feature=None, target_label=None):
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
	ctrl = df[feature[0]].where(df[target_label]==0).dropna()
	sccs = df[feature[0]].where(df[target_label]==1).dropna()
	print("Number of control subjects=",len(ctrl.notnull()))
	print("Number of scc subjects=",len(sccs.notnull()))
	for idx, val in enumerate(feature):
		#expl_control_var = feature[idx] + ' ~ conversion'
		expl_control_var = "{} ~ {}".format(feature[idx],target_label)
		mod = ols(expl_control_var, data=df).fit()
		aov_table = sm.stats.anova_lm(mod, typ=2)
		print("ANOVA table feature:", val) 
		print(aov_table)
		#Create a boxplot
		df.boxplot(feature[idx], by=target_label, figsize=(12, 8))

def ancova_test(dataset, features=None):
	""" ACNOVA test controlling for features that may have a relationship with the dependent variable"""
	print("Calculating ANCOVA for features:\n",df.ix[0])

def t_test(dataset, features=None):
	""" t test"""
	print("Calculating t test for features:\n",df.ix[0])



if __name__ == "__name__":
	main()