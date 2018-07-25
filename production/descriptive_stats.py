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

	# 1. Exploratory Data analysis EDA: 
	#	1.0 EDA plot
	#   1.1. Detect Multicollinearity
	# 	1.2. Variable Selection (leakage, combine features)
	#	1.3. Transformation (scaling, discretize continuous variables, expand categorical variables)
	# 2. Prediction
	# 3. Causation
	# 4. Deployment	
========================
"""
from __future__ import print_function
import os, sys, pdb, operator
import datetime
import time
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
from copy import deepcopy
import statsmodels.api as sm
from sklearn.externals import joblib

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import SGDClassifier, LogisticRegression, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, ShuffleSplit, KFold
from sklearn.metrics import r2_score, roc_curve, auc, roc_auc_score, log_loss, accuracy_score, confusion_matrix, classification_report, precision_score, matthews_corrcoef, recall_score, f1_score, cohen_kappa_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets.mldata import fetch_mldata

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.manifold.t_sne import (_joint_probabilities,_kl_divergence)
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from xgboost import XGBClassifier
from keras import regularizers
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils, plot_model
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import Callback

import networkx as nx
from adspy_shared_utilities import plot_class_regions_for_classifier, plot_decision_tree, plot_feature_importances, plot_class_regions_for_classifier_subplot

def main():
	plt.close('all')
	#Particular EDAs
	#df = meritxell_eda()
	
	
	#pdb.set_trace()
	#df = socioeconomics_infographics()
	#pdb.set_trace()
	# get the data in csv format
	dataframe = run_load_csv()	
	# Feature Selection : cosmetic name changing and select input and output 
	print('Calling for cosmetic cleanup (all lowercase, /, remove blanks) e.g. cleanup_column_names(df,rename_dict={},do_inplace=True)') 
	cleanup_column_names(dataframe, {}, True)
	#copy dataframe with the cosmetic changes e.g. Tiempo is now tiempo
	dataframe_orig = dataframe.copy()
	#combine physical exercise and diet features
	dataframe = combine_features(dataframe)
	#income_charts(dataframe)

	###########################################################################################
	##################  3.0. EDA plot ##########################################################
	target_variable = 'conversionmci'
	edaplot = False
	if edaplot is True:
		print('Plot longitudinal features')
		longit_pattern = re.compile("^scd_+visita[1-5]+$"); 
		longit_pattern2 = re.compile("^stai_+visita[1-5]+$") 
		longit_pattern3 = re.compile("^gds_+visita[1-5]+$")
		longit_pattern4 = re.compile("^fcsrtrl1_visita[1-5]+$") 
		longit_pattern5 = re.compile("^preocupacion_visita[1-5]+$") 
		longit_pattern6 = re.compile("^mmse_visita[1-5]+$") 
		
		# longit_pattern = re.compile("^mmse_+visita[1-5]+$") 
		# # plot N histograms one each each variable_visitai
		plot_histograma_one_longitudinal(dataframe, longit_pattern)
		plot_histograma_one_longitudinal(dataframe, longit_pattern2)
		plot_histograma_one_longitudinal(dataframe, longit_pattern3)
		plot_histograma_one_longitudinal(dataframe, longit_pattern4)
		plot_histograma_one_longitudinal(dataframe, longit_pattern5)
		plot_histograma_one_longitudinal(dataframe, longit_pattern6)
		# YS: this is hardcode fix
		plot_histograma_bygroup_categorical(dataframe, target_variable=target_variable)
		#(4) Descriptive analytics: plot scatter and histograms
		#longit_xy_scatter = ['scd_visita', 'gds_visita'] #it works for longitudinal
		plot_scatter_target_cond(dataframe, ['scd_visita', 'gds_visita'], target_variable=target_variable)
		#features_to_plot = ['scd_visita1', 'gds_visita1'] 
		plot_histogram_pair_variables(dataframe, ['scd_visita2', 'gds_visita2'] )
		# #plot 1 histogram by grouping values of one continuous feature 
		plot_histograma_bygroup(dataframe, 'glu')
		plot_histograma_bygroup(dataframe, 'conversionmci')
		# # plot one histogram grouping by the value of the target variable
		plot_histograma_bygroup_target(dataframe, 'conversionmci')
	
	################################################################################################
	##################  1.1.Detect Multicollinearity  ##############################################

	gaussian_test = False
	if gaussian_test is True:
		print('Test for Gaussian using Shapiro and Kolmogorov tests...\n')
		#visual ispection better than p-valñue, if some ouliers p ~0 reject 
		#null hyp is gaussian but it fits the line so it is gaussian
		colname = ['imc', 'a13', 'scd_visita1', 'a08', 'renta']
		for colix in colname:
			normal_gaussian_test(dataframe[colix],colix, method=None)
	multicollin = False
	if multicollin is True:
		feature_x = 'scd_visitan'
		feature_y = target_variable
		dfjoints = dataframe_orig[[feature_x, feature_y]].dropna()
		#plot_jointdistributions(dfjoints, feature_x, feature_y)
		# To plot scatter feature_x and feature_x2 uncomment
		# #feature_x2 = 'alcar' 
		# #dfjoints = dataframe[[feature_x,feature_y, feature_x2]].dropna() 
		# #plot_jointdistributions(dfjoints, feature_x, feature_y, feature_x2)
		# dfjoints = dataframe[[feature_x,feature_y]].dropna()
		plot_jointdistributions(dfjoints, feature_x, feature_y)

		#Detect multicollinearities
		#cols_list = [['scd_visita1', 'gds_visita1']] to multicollinearity of a list
		#cols_list = [['scd_visita1', 'edadinicio_visita1', 'tpoevol_visita1', 'peorotros_visita1', 'preocupacion_visita1', 'eqm06_visita1', 'eqm07_visita1', 'eqm81_visita1', 'eqm82_visita1', 'eqm83_visita1', 'eqm84_visita1', 'eqm85_visita1', 'eqm86_visita1', 'eqm09_visita1', 'eqm10_visita1', 'act_aten_visita1', 'act_orie_visita1', 'act_mrec_visita1', 'act_memt_visita1', 'act_visu_visita1', 'act_expr_visita1', 'act_comp_visita1', 'act_ejec_visita1', 'act_prax_visita1', 'act_depre_visita1', 'act_ansi_visita1', 'act_apat_visita1', 'gds_visita1', 'stai_visita1', 'eq5dmov_visita1', 'eq5dcp_visita1', 'eq5dact_visita1', 'eq5ddol_visita1', 'eq5dans_visita1', 'eq5dsalud_visita1', 'eq5deva_visita1', 'relafami_visita1', 'relaamigo_visita1', 'relaocio_visita1', 'rsoled_visita1', 'valcvida_visita1', 'valsatvid_visita1', 'valfelc_visita1']]
		cols_list = [['scd_visita1', 'gds_visita1', 'educrenta', 'nivelrenta', 'apoe']]
		#cols_list = [['scd_visitan', 'gds_visitan', 'educrenta', 'nivelrenta', 'apoe', 'valsatvid2_visitan', 'a13', 'sue_rec','imc']]
		cols_list = [['scd_visita1', 'gds_visita1', 'educrenta', 'nivelrenta', 'apoe', 'valsatvid2_visitan', 'a13', 'sue_rec','imc']]

		#for cols in cols_list:
		#for cols in dict_features.keys():
		for cols in cols_list:
		 	print("Calculating multicolinearities for Group feature: ", cols)
		 	#features = dict_features[cols]
		 	features = cols
		 	#detect_multicollinearities calls to plot_jointdistributions
		 	detect_multicollinearities(dataframe, target_variable, features)

	################################################################################################
	##################  END 1.1.Detect Multicollinearity t ##########################################

	######################################################################################################
	##################  1.2. Variable Selection ##########################################################
	# Leakage data and remove unnecessary features
	#Remove cognitive performance features for data leakage. This can be done in colstoremove
	# features_to_remove = ['mmse_visita1','reloj_visita1','faq_visita1','fcsrtrl1_visita1','fcsrtrl2_visita1','fcsrtrl3_visita1','fcsrtlibdem_visita1','p_visita1','animales_visita1','cn_visita1','cdrsum_visita1']
	# features_to_remove = features_to_remove + [ 'edadinicio_visita1', 'tpoevol_visita1', 'peorotros_visita1', 'eqm06_visita1', 'eqm07_visita1', 'eqm81_visita1', 'eqm82_visita1', 'eqm83_visita1', 'eqm84_visita1', 'eqm85_visita1', 'eqm86_visita1', 'eqm09_visita1', 'eqm10_visita1','act_memt_visita1', 'act_ejec_visita1', 'act_prax_visita1', 'act_depre_visita1', 'act_ansi_visita1', 'eq5dmov_visita1', 'eq5dcp_visita1', 'eq5dact_visita1', 'eq5ddol_visita1', 'eq5dans_visita1', 'relaocio_visita1', 'rsoled_visita1']
	# colstoremove = ['tiempomci','tpo1.2','tpo1.3','tpo1.4','tpo1.5','dx_visita1','fecha_visita1','fecha_nacimiento','id', 'ultimodx','conversiondementia', 'tiempodementia']
	# colstoremove = colstoremove + features_to_remove
	# #colstoremove = ['dx_visita1','fecha_nacimiento','id']
	# print('Calling to leakage_data to remove features:', colstoremove, ' about the target \n')
	# dataframe = leakage_data(dataframe, colstoremove)
	# print('Removed ', dataframe_orig.shape[1] - dataframe.shape[1], ' columns in the dataframe \n' )
	# features_list = dataframe.columns.values.tolist()
	# #combine physical exercise and diet features
	# #dataframe = combine_features(dataframe)
	
	dict_features = split_features_in_groups()
	print("Dictionary of static features: ", dict_features)
	run_print_dataset(dataframe)
	
	# Select subset of explanatory variables from prior information MUST include the target_variable
	features_static = dict_features['vanilla'] + dict_features['sleep'] + dict_features['anthropometric'] + \
	dict_features['intellectual'] + dict_features['offspring'] + \
	dict_features['professional'] +  dict_features['cardiovascular'] + dict_features['ictus'] + \
	dict_features['diet'] + dict_features['wealth'] + dict_features['physical_exercise'] + dict_features['familiar_ad'] 
	
	#Remove cognitive performance features for data leakage. This can be done in colstoremove
	#features_to_remove = ['mmse_visita1','reloj_visita1','faq_visita1','fcsrtrl1_visita1','fcsrtrl2_visita1','fcsrtrl3_visita1','fcsrtlibdem_visita1','p_visita1','animales_visita1','cn_visita1','cdrsum_visita1']
	#features_to_remove = features_to_remove + [ 'edadinicio_visita1', 'tpoevol_visita1', 'peorotros_visita1', 'eqm06_visita1', 'eqm07_visita1', 'eqm81_visita1', 'eqm82_visita1', 'eqm83_visita1', 'eqm84_visita1', 'eqm85_visita1', 'eqm86_visita1', 'eqm09_visita1', 'eqm10_visita1','act_memt_visita1', 'act_ejec_visita1', 'act_prax_visita1', 'act_depre_visita1', 'act_ansi_visita1', 'eq5dmov_visita1', 'eq5dcp_visita1', 'eq5dact_visita1', 'eq5ddol_visita1', 'eq5dans_visita1', 'relaocio_visita1', 'rsoled_visita1']
	# selected_features = [x for x in all_features if x not in features_to_remove]
	# features_year1 = [s for s in selected_features if "visita1" in s ];
	#features_year2 = [s for s in dataset.keys().tolist()  if "visita2" in s]; features_year2.remove('fecha_visita2')
	#features_year3 = [s for s in dataset.keys().tolist()  if "visita3" in s]; features_year3.remove('fecha_visita3'); features_year3.remove('act_prax_visita3'), features_year3.remove('act_comp_visita3')
	#features_year4 = [s for s in dataset.keys().tolist()  if "visita4" in s]; features_year4.remove('fecha_visita4'); features_year4.remove('act_prax_visita4'), features_year4.remove('act_comp_visita4')
	#features_year5 = [s for s in dataset.keys().tolist()  if "visita5" in s]; features_year5.remove('fecha_visita5'); features_year5.remove('act_prax_visita5'), features_year5.remove('act_comp_visita5')
	#features_year = ['scd_visitan', 'preocupacion_visitan', 'act_aten_visitan', 'act_orie_visitan', \
	#'act_mrec_visitan', 'act_visu_visitan', 'act_expr_visitan', 'act_comp_visitan', 'act_apat_visitan', 'gds_visitan',\
	#'stai_visitan', 'eq5dsalud_visitan', 'eq5deva_visitan', 'valcvida2_visitan', 'valsatvid2_visitan', 'valfelc2_visitan']
	features_year = ['scd_visita1', 'preocupacion_visita1', 'act_aten_visita1', 'act_orie_visita1', \
	'act_mrec_visita1', 'act_visu_visita1', 'act_expr_visita1', 'act_comp_visita1', 'act_apat_visita1', 'gds_visita1',\
	'stai_visita1', 'eq5dsalud_visita1', 'eq5deva_visita1', 'valcvida2_visita1', 'valsatvid2_visita1', 'valfelc2_visita1']
	#explanatory_features = features_static + features_year1
	explanatory_features = features_static + features_year
	#explanatory_features = ['my favorite list of features']
	#target_variable = 'conversion' # if none assigned to 'conversion'. target_variable = ['visita_1_EQ5DMOV']
	print("Calling to run_variable_selection(dataset, explanatory_features= {}, target_variable={})".format(explanatory_features, target_variable))
	#dataset, explanatory_features = run_variable_selection(dataset, explanatory_features, target_variable)
	# dataset with all features including the target and removed NaN
	#dataframe.dropna(axis=0, how='any', inplace=True)

	explanatory_and_target_features = deepcopy(explanatory_features)
	explanatory_and_target_features.append(target_variable)
	dataframe = dataframe[explanatory_and_target_features]
	all_features = dataframe_orig.keys().tolist()
	print(' Features dataframe_orig=', all_features,' Features dataframe post =', dataframe.shape[1])	
	print ("Features containing NAN values:\n {}".format(dataframe.isnull().any()))
	print( "Number of NaN cells in original dataframe:{} / {}, total rows:{}".format(pd.isnull(dataframe.values).sum(axis=1).sum(), dataframe.size, dataframe.shape[0]))
	
	coluswithnans = []
	for colu in dataframe.columns:
		nanscount = np.sum(pd.isnull(dataframe[colu]))
		if nanscount > 0:
			coluswithnans.append(colu)
			print("Number of NaNs for column", colu, " = ", nanscount)
	
	dataframe.dropna(axis=0, how='any', inplace=True)
	print( "Number of NaN cells in the imputed dataframe: {} / {}, total rows:{}".format(pd.isnull(dataframe.values).sum(axis=1).sum(), dataframe.size, dataframe.shape[0]))
	######################################################################################################
	##################  END 1.2. Variable Selection ##########################################################

	######################################################################################################
	##################   1.3. Transformation (Scaling) ###################################################
	Xy = dataframe[explanatory_and_target_features].values
	X = Xy[:,:-1]
	y = Xy[:,-1]
	X_scaled_list, scaler = run_transformations(X) # Normalize to minmaxscale or others make input normal
	#run_feature_ranking requires all positive 	X_scaled_list[0]. Use standard otherwise
	X_scaled = 	X_scaled_list[0] #0 MinMax (0,1), 1 standard (mean 0 , std 1), 2 robust
	print("X Standard (mu=0, std=1) scaled dimensions:{} \n ".format(X_scaled.shape))

	#construct a LogisticRegression model to choose the best performing features 
	nbofRFEfeatures = 0 # how many best features you want to find
	if nbofRFEfeatures > 0:
		print('Running RFE algorithm to select the ', nbofRFEfeatures, ' most important features for Logistic Regression...\n')
		best_logreg_features = recursive_feature_elimination(X_scaled, y, nbofRFEfeatures, explanatory_and_target_features[:-1])
		print('The top features are:', best_logreg_features)
		best_logreg_features.append(target_variable)
		dataframe = dataframe[best_logreg_features]
	
	# (3.3) detect multicollinearity: Plot correlation and graphs of variables
	#convert ndarray to pandas DataFrame
	X_df_scaled = pd.DataFrame(X_scaled, columns=explanatory_features)
	Xy_df_scaled = X_df_scaled.copy()
	Xy_df_scaled[target_variable] = Xy[:,-1]

	########################################################################################################
	##################   1.3. END Transformation (Scaling)###################################################

	######################################################################################################
	##################  1.4. Network analysis ############################################################
	#corr_X_df = run_correlation_matrix(X_df_scaled, explanatory_features) #[0:30] correlation matrix of features
	networkanalysis = False
	if networkanalysis is True:
		corr_Xy_df = run_correlation_matrix(Xy_df_scaled, explanatory_and_target_features) # correlation matrix of features and target
		#corr_matrix = corr_df.as_matrix()
		corr_with_target = corr_Xy_df[target_variable]
		print("Correlations with the target:\n{}".format(corr_with_target.sort_values()))
		#corr_with_target = calculate_correlation_with_target(Xdf_imputed_scaled, target_values) # correlation array of features with the target
		threshold = np.mean(np.abs(corr_Xy_df.as_matrix())) + 3*np.std(np.abs(corr_Xy_df.as_matrix()))
		graph = build_connectedGraph_correlation_matrix(corr_Xy_df, threshold, corr_with_target)
		graph_metrics = calculate_network_metrics(graph)
		# # print sumary network metrics
		print_summary_network(graph_metrics, nodes=corr_Xy_df.keys().tolist(), corrtarget=corr_with_target)
	######################################################################################################
	##################  END 1.4. Network analysis ############################################################

	######################################################################################################
	##################  1.4. Statistical tests ############################################################
	# # # perform statistical tests: ANOVA
	statstest = False
	if statstest is True:
		feature_to_test = ['familiar_ad', 'nivel_educativo', 'tabac_cant', 'apoe'] #['scd_visita1']
		feature_to_test = ['renta', 'hta', 'glu', 'lipid', 'tabac_cant', 'cor', 'arri', 'card', 'ictus', 'tce', 'imc','valcvida_visita1', 'physical_exercise', 'dietaketo', 'dietaglucemica', 'dietasaludable','sue_noc', 'sue_rec', 'imc']
		feature_to_test = explanatory_features
		target_anova_variable = target_variable # nivel_educativo' #tabac_visita1 depre_visita1
		tests_result = run_statistical_tests(Xy_df_scaled,feature_to_test, target_anova_variable)
		#tests_result.keys() 
	
	######################################################################################################
	##################  END 1.4. Statistical tests ########################################################

	######################################################################################################
	##################  1.5. Dimensionality Reduction ############################################################
	# # # (5) Dimensionality Reduction
	dimreduction = False
	if dimreduction is True:
		pca, projected_data = run_PCA_for_visualization(Xy_df_scaled,target_variable, explained_variance=0.7)
		print("The variance ratio by the {} principal compoments is:{}, singular values:{}".format(pca.n_components_, pca.explained_variance_ratio_,pca.singular_values_ ))
	#Manifold learning
	######################################################################################################
	##################  END 1.5. Dimensionality Reduction ############################################################
	
	######################################################################################################
	################## 2. PREDICTION ############################################################

	formula = build_formula(explanatory_features)
	
	# build design matrix(patsy.dmatrix) and rank the features in the formula y ~ X 
	#Xy_df_scaled must be MinMax [0,1]
	feature_ranking = 1
	if feature_ranking > 0:
		#run_feature_ranking requires all positive 	X_scaled_list[0]. Use standard otherwise
		run_feature_ranking(Xy_df_scaled, formula, nboffeats=12)
		#plot ranking of best features
	
	#Split dataset into train and test
	y = Xy_df_scaled[target_variable].values
	X_features = explanatory_features
	if target_variable in explanatory_features:
		X_features.remove(target_variable)
	X = Xy_df_scaled[X_features].values
	X_train, X_test, y_train, y_test = run_split_dataset_in_train_test(X, y, test_size=0.2)

	######################################################################################################
	##### resampling data with SMOTE #####################################################################
	######################################################################################################
	from sklearn.utils import shuffle
	smote_algo = False
	if smote_algo is True:
		X_resampled_train, y_resampled_train = resampling_SMOTE(X_train, y_train)
		X_shuf, Y_shuf = shuffle(X_resampled_train, y_resampled_train)
		X_resampled_test, y_resampled_test = resampling_SMOTE(X_test, y_test)
		X_shuf_test, Y_shuf_test = shuffle(X_resampled_test, y_resampled_test)
		X_train = X_resampled_train; y_train = y_resampled_train; X_test=X_resampled_test;y_test=y_resampled_test
		X_train = X_shuf; y_train = Y_shuf; X_test=X_shuf_test; y_test=Y_shuf_test

	################## Build List of Classifers/Learners ############################################################
	#Run Classifiers
	learners = {}; dict_learners = {}

	#TEST
	#SVM for both linear and non linear kernel
	#print('Building a Sigmoid (Linear) svm_estimator.....\n')
	#learners['svm_sigmoid_estimator'] = run_svm_classifier(X_train, y_train, X_test, y_test, kernel='sigmoid')
	#dict_learners["svm_sigmoid"]=learners['svm_sigmoid_estimator']
	#pdb.set_trace()
	print('Building a RBF svm_estimator.....\n')
	learners['svm_rbf_estimator'] = run_svm_classifier(X_train, y_train, X_test, y_test, kernel='rbf')
	dict_learners["svm_rbf"]=learners['svm_rbf_estimator']
	pdb.set_trace()

	boost_type = 'XGBClassifier'
	learners['xgbcboost_estimator'] = run_gradient_boosting_classifier(X_train, y_train, X_test, y_test, boost_type)
	print_feature_importances(learners['xgbcboost_estimator'], explanatory_features)
	#only for XGBClassifier learner
	plot_feature_importance(learners['xgbcboost_estimator'], explanatory_features)
	dict_learners["xgbc"]=learners['xgbcboost_estimator']
	pdb.set_trace()

	print('Building an unconstrained Logistic Regression Classifier.....\n')
	learners['lr_estimator'] = run_logistic_regression(X_train, y_train, X_test, y_test, 0.5, explanatory_and_target_features[:-1])
	dict_learners["lr"]=learners['lr_estimator']

	print('Building a k-neighbors Classifier.....\n')
	learners['knn_estimator'] = run_kneighbors(X_train, y_train, X_test, y_test)
	dict_learners["knn"]=learners['knn_estimator']

	#SVM for both linear and non linear kernel
	#print('Building a Sigmoid (Linear) svm_estimator.....\n')
	#learners['svm_sigmoid_estimator'] = run_svm_classifier(X_train, y_train, X_test, y_test, kernel='sigmoid')
	#dict_learners["svm_sigmoid"]=learners['svm_sigmoid_estimator']
	print('Building a RBF svm_estimator.....\n')
	learners['svm_rbf_estimator'] = run_svm_classifier(X_train, y_train, X_test, y_test, kernel='rbf')
	dict_learners["svm_rbf"]=learners['svm_rbf_estimator']


	print('Building a naive_bayes_estimator.....\n')
	learners['naive_bayes_estimator'] = run_naive_Bayes(X_train, y_train, X_test, y_test)
	dict_learners["nb"]=learners['naive_bayes_estimator']
	
	#compare estimators against dummy estimators
	dummies_score = build_dummy_scores(X_train, y_train, X_test, y_test)
	#listofestimators = [learners['svm_estimator'],learners['lr_estimator'],learners['dt_estimator'],learners['rf_estimator']]
	#estimatorlabels = ['svm', 'lr', 'dt', 'rf']
	#compare_against_dummy_estimators(listofestimators, estimatorlabels, X_test, y_test, dummies_score)
	compare_against_dummy_estimators(learners, learners.keys(), X_test, y_test, dummies_score)
	#del learners['lasso_estimator']
	#selct some lements of the dict
	#learners_metrics = {'dt_estimator':learners['dt_estimator'], 'voting_estimator':learners['voting_estimator'], 'rfgrid_estimator':learners['rfgrid_estimator'],'adaboost_estimator':learners['adaboost_estimator'], 'xgbcboost_estimator':learners['xgbcboost_estimator']}
	all_results = evaluate_learners_metrics(learners, X_train, y_train, X_test, y_test)
	

	print('Building a random_decision_tree.....\n')
	learners['dt_estimator'] = run_decision_tree(X_train, y_train, X_test, y_test, X_features, target_variable)
	dict_learners["dt"]=learners['dt_estimator']
	
	print('Calling to Ensemble Learner with majority vote...')
	learners['voting_estimator'] = run_votingclassifier(dict_learners, X_train, y_train, X_test, y_test)
	dict_learners["voting"]=learners['voting_estimator']

	print('Building a (Bagging) randomforest estimator.....\n')
	#learners['rf_estimator'], learners['rfgrid_estimator'] = run_randomforest(X_train, y_train, X_test, y_test, X_features)
	learners['rfgrid_estimator'] = run_randomforest(X_train, y_train, X_test, y_test, X_features)[1]
	#print_feature_importances ONLY for RF no for GridseearchCV
	#print_feature_importances(learners['rf_estimator'], explanatory_features)
	#dict_learners["rf"]=learners['rf_estimator']
	dict_learners["rfgrid"]=learners['rfgrid_estimator']
	print('Best parameters for RF is:{}', dict_learners["rfgrid"].best_params_)
	pdb.set_trace()

	print('Building a (Boosting) run_boosting_classifier:AdaBoostClassifier,GradientBoostingClassifier,XGBClassifier.....\n')
	boost_type = 'AdaBoostClassifier' #AdaBoostClassifier,GradientBoostingClassifier,XGBClassifier
	learners['adaboost_estimator'] = run_gradient_boosting_classifier(X_train, y_train, X_test, y_test, boost_type)
	print_feature_importances(learners['adaboost_estimator'], explanatory_features)
	dict_learners["ada"]=learners['adaboost_estimator']

	boost_type = 'XGBClassifier'
	learners['xgbcboost_estimator'] = run_gradient_boosting_classifier(X_train, y_train, X_test, y_test, boost_type)
	print_feature_importances(learners['xgbcboost_estimator'], explanatory_features)
	#only for XGBClassifier learner
	plot_feature_importance(learners['xgbcboost_estimator'], explanatory_features)
	dict_learners["xgbc"]=learners['xgbcboost_estimator']
	pdb.set_trace()
	#compare estimators against dummy estimators
	dummies_score = build_dummy_scores(X_train, y_train, X_test, y_test)
	#listofestimators = [learners['svm_estimator'],learners['lr_estimator'],learners['dt_estimator'],learners['rf_estimator']]
	#estimatorlabels = ['svm', 'lr', 'dt', 'rf']
	#compare_against_dummy_estimators(listofestimators, estimatorlabels, X_test, y_test, dummies_score)
	compare_against_dummy_estimators(learners, learners.keys(), X_test, y_test, dummies_score)
	#del learners['lasso_estimator']
	#selct some lements of the dict
	learners_metrics = {'dt_estimator':learners['dt_estimator'], 'voting_estimator':learners['voting_estimator'], 'rfgrid_estimator':learners['rfgrid_estimator'],'adaboost_estimator':learners['adaboost_estimator'], 'xgbcboost_estimator':learners['xgbcboost_estimator']}
	all_results = evaluate_learners_metrics(learners_metrics, X_train, y_train, X_test, y_test)
	pdb.set_trace()
	
	##### Deep Networks #####################################################################
	#########################################################################################
	print('Building a MLP_classifier estimator.....\n')
	learners['mlp_estimator'] = run_multi_layer_perceptron(X_train, y_train, X_test, y_test)
	dict_learners["mlp"]=learners['mlp_estimator']

	print('Building a Keras_DL_classifier estimator.....\n')
	learners['deepnetwork_res'] = run_keras_deep_learning(X_train, y_train, X_test, y_test)
	dict_learners["deep"]=learners['deepnetwork_res']

	# print('Building a lasso_estimator.....\n')
	# learners['lasso_estimator'] = run_logreg_Lasso(X_train, y_train, X_test, y_test,10)
	# dict_learners["log_reg_lasso"]=learners['lr_estimator']
	
	# print('Building a sgd.....\n')
	# learners['sgd_estimator'] = run_sgd_classifier(X_train, y_train, X_test, y_test,'hinge',10) #loss = log|hinge
	# dict_learners["og_reg_sgd"]=learners['sgd_estimator']
	################## Save the List of Classifers/Learners ############################################################
	save_models = False
	if save_models is True:
		for learner in learners.keys():
			print('Saving the model {}.....\n', learner)
			save_the_model(learners[learner], learner, '/Users/jaime/github/code/tensorflow/production/ML_models/')	
	######################################################################################################
	################## MODEL EVALUATION ############################################################

	#Model interpretation of the estimators, exclude SVM 
	skater_interpretation = False
	if skater_interpretation is True:
		for learner in learners.keys():
			if learner is 'svm_estimator' or 'svm_sigmoid' or 'svm_rbf' or 'lasso_estimator' or 'knn_estimator':
				print('Skipping SVM has not predict_proba...\n')
			else: 
				print('\n ***** Running model {} interpretation with Skater..... **** \n', learner)
				time.sleep(3)
				model_interpretation(learners[learner], X_train, X_test, explanatory_features)	

	#compare estimators against dummy estimators
	dummies_score = build_dummy_scores(X_train, y_train, X_test, y_test)
	#listofestimators = [learners['svm_estimator'],learners['lr_estimator'],learners['dt_estimator'],learners['rf_estimator']]
	#estimatorlabels = ['svm', 'lr', 'dt', 'rf']
	#compare_against_dummy_estimators(listofestimators, estimatorlabels, X_test, y_test, dummies_score)
	compare_against_dummy_estimators(learners, learners.keys(), X_test, y_test, dummies_score)
	#del learners['lasso_estimator']
	all_results = evaluate_learners_metrics(learners, X_train, y_train, X_test, y_test)
	pdb.set_trace()
	
	######################################################################################################
	##### Unsupervised Learning ##########################################################################
	######################################################################################################
	clustering_analysis = False
	if clustering_analysis is True:
		tSNE_reduced  = run_tSNE_manifold_learning(X_train, y_train, X_test, y_test)
		svd_reduced = run_truncatedSVD(X_train, y_train, X_test, y_test)
		hierclust = run_hierarchical_clustering(np.concatenate((X_train, X_test), axis=0), explanatory_features)

	TDA_analysis = False
	if TDA_analysis is True:
		#Algebraic topology
		run_TDA_with_Kepler(samples, activations)
	
	# #
	# calculate_top_features_contributing_class(sgd_estimator, X_features, 10)
	# #compare estimators against dummy estimators
	# dummies_score = build_dummy_scores(X_train, y_train, X_test, y_test)
	# listofestimators = [knn, naive_bayes_estimator,lr_estimator,dectree_estimator,rf_estimator,gbm_estimator,xgbm_estimator]
	# estimatorlabels = ['knn', 'nb', 'lr', 'dt', 'rf','gbm','xgbm']
	# compare_against_dummy_estimators(listofestimators, estimatorlabels, X_test, y_test, dummies_score)
	# #Evaluate a score comparing y_pred=estimator().fit(X_train)predict(X_test) from y_test
	# metrics_estimator = compute_metrics_estimator(knn,X_test,y_test)
	# #Evaluate a score by cross-validation
	# metrics_estimator_with_cv = compute_metrics_estimator_with_cv(knn,X_test,y_test,5)
	
	# # QUICK Model selection accuracy 0 for Train test, >0 for the number of folds
	# grid_values = {'gamma': [0.001, 0.01, 0.1, 1, 10]}
	# #print_model_selection_metrics(X_train, y_train, X_test, y_test,0) -train/test; print_model_selection_metrics(X_train, y_train, X_test, y_test,10) KFold
	# print_model_selection_metrics(X_train, y_train, X_test, y_test, grid_values)
	


		

	print('Program Finished!! Yay!! \n\n\n')

	
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
	"""run_variable_selection: select features: explanatory and target. check if target var is in
	explanatory if not EXIT
	Args: dataset, explanatory_features, target_variable : list of explanatory variables if None assigned all the features dataset.keys()
	target_variable: target feature, if None is assigned inside the function 
	Output: dataframe containing the selected explanatory and target variables
	Example: run_variable_selection(dataset, ['', ''...], 'conversion')
	run_variable_selection(dataset, ['', ''...])
	run_variable_selection(dataset)
	%https://www.coursera.org/learn/python-machine-learning/lecture/meBKr/model-selection-optimizing-classifiers-for-different-evaluation-metrics
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

def combine_features(dataset):
	""" combine_features: combine features and remove redundant 
	Args:dataset
	Output: detaset"""
	#diet_med = ['alfrut','alaves', 'alaceit', 'alpast', 'alpan', 'alverd','allact','alpesblan', 'alpeszul']
	import math
	phys_exercise = ['a01', 'ejfre', 'ejminut']
	dataset['physical_exercise'] = dataset['ejfre']*dataset['ejminut']
	dataset.drop(['a01', 'ejfre', 'ejminut'], axis=1,  inplace=True)
	dataset['familiar_ad'] = dataset['demmad'] | dataset['dempad']
	dataset.drop(['edemmad', 'edempad'], axis=1,  inplace=True)
	#fillna with mean
	dataset['dietaketo'] = dataset['dietaproteica']*dataset['dietagrasa']
	dataset['dietaketo'].fillna(dataset['dietaketo'].mean(), inplace=True)
	dataset['dietaketo'] = dataset['dietaketo'].astype('int')
	dataset['dietaglucemica'].fillna(dataset['dietaglucemica'].mean(), inplace=True)
	dataset['dietaglucemica'] = dataset['dietaglucemica'].astype('int')
	dataset['dietasaludable'].fillna(dataset['dietasaludable'].mean(), inplace=True)
	dataset['dietasaludable'] = dataset['dietasaludable'].astype('int')
	
	#make last visit column
	make_visit_N = False
	if make_visit_N is True:  
		visitas = ['tpo1.2', 'tpo1.3', 'tpo1.4', 'tpo1.5']
		visitas_scores =['scd_visita1', 'preocupacion_visita1', 'act_aten_visita1', 'act_orie_visita1', 'act_mrec_visita1', 'act_visu_visita1', 'act_expr_visita1', 'act_comp_visita1', 'act_apat_visita1', 'gds_visita1', 'stai_visita1', 'eq5dsalud_visita1', 'eq5deva_visita1', 'valcvida2_visita1', 'valsatvid2_visita1', 'valfelc2_visita1'] #'relafami_visita1', 'relaamigo_visita1', valcvida(2)_visita1, valsatvid(2)_visita1, valfelc2_visita1
		#initialize visitaN to visita1
		for k in visitas_scores:
			score_year = k[:-1] + '1'
			score_N = k[:-1] + 'N'
			dataset[score_N] = dataset[score_year]

		for subject in np.arange(0, dataset.shape[0]):		
			print('Finding last visit for subject:', subject)
			for v in reversed(visitas):
				if math.isnan(dataset[v][subject]) == False:
					print('Found last visit for subject:', subject, ' visit: ', v)
					year = v[-1]
					for k in visitas_scores:
						score_year = k[:-1] + year
						score_N = k[:-1] + 'N'
						dataset[score_N] = ""
						dataset[score_N][subject] = dataset[score_year][subject]
						print('Changed :', score_year, ' to :', score_N)
					break
			print('Out of loop s:', subject, ' v:', v)
	dataset.to_csv('/Users/jaime/vallecas/data/BBDD_vallecas/vallecaswithvisitN.csv')		
	return dataset

def leakage_data(dataset, colstoremove):
	"""leakage_data: remove attributes about the target and others we may not need
	Args: dataset (pd)
	Output: dataset (pd)"""
	#tiempo (time to convert), tpo1.1..5 (time from year 1 to conversion), dx visita1
	for col in colstoremove:
		dataset.drop(col, axis=1,inplace=True)
	return dataset	
	# dataset.drop('tiempo', axis=1,inplace=True)
	# dataset.drop('tpo1.2', axis=1,inplace=True)
	# dataset.drop('tpo1.3', axis=1,inplace=True)
	# dataset.drop('tpo1.4', axis=1,inplace=True)
	# dataset.drop('tpo1.5', axis=1,inplace=True)
	# dataset.drop('dx_visita1', axis=1,inplace=True)
	# #Dummy features to remove: id, fecha nacimiento, fecha_visita
	# dataset.drop('fecha_visita1', axis=1,inplace=True)
	# dataset.drop('fecha_nacimiento', axis=1,inplace=True)
	# dataset.drop('id', axis=1,inplace=True)
	return dataset

def cleanup_column_names(df,rename_dict={},do_inplace=True):
    """cleanup_column_names: renames columns of a pandas dataframe. It converts column names to snake case if rename_dict is not passed. 
    Args: rename_dict (dict): keys represent old column names and values point to newer ones
        do_inplace (bool): flag to update existing dataframe or return a new one
    Returns: pandas dataframe if do_inplace is set to False, None otherwise
    """
    #Rename columns eg df.rename(index=str, columns={"A": "a", "B": "c"})
    df.rename(index=str, columns={"Edad_visita1": "edad"}, inplace=True) 
    #df.drop('lat_visita1', inplace=True)
    if not rename_dict:
        return df.rename(columns={col: col.replace('/','').lower().replace(' ','_') 
                    for col in df.columns.values.tolist()}, inplace=do_inplace)
    else:
        return df.rename(columns=rename_dict,inplace=do_inplace)

def plot_histograma_one_longitudinal(df, longit_pattern=None):
	""" plot_histogram_pair_variables: histograma 1 for each year 
	Args: Pandas dataframe , regular expression pattern eg mmse_visita """

	longit_status_columns = [ x for x in df.columns if (longit_pattern.match(x))]
	df[longit_status_columns].head(10)
	# plot histogram for longitudinal pattern
	fig, ax = plt.subplots(2,3)
	fig.set_size_inches(15,5)
	#fig.suptitle('Distribution in' +  str(len(longit_status_columns)) + ' visits')
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
	fig.set_size_inches(12,4)
	ax = fig.add_subplot(111)
	grd = df.groupby([label]).size()
	ax.set_yscale("log")
	ax.set_xticks(np.arange(len(grd)))
	ax.set_xlabel('Group by values of ' + label )
	fig.suptitle('Histogram of ' + label + ' log scale')
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
	title = 'number of ' + target_label + '  vs non' + target_label
	fig.suptitle(title)
	plt.show()

def plot_histograma_bygroup_categorical(df, target_variable=None):
	# target related to categorical features
	#categorical_features = ['sexo','nivel_educativo', 'apoe', 'edad']
	#df['sexo'] = df['sexo'].astype('category').cat.rename_categories(['M', 'F'])
	df['apoe'] = df['apoe'].astype('category').cat.rename_categories(['No', 'Het', 'Hom'])
	df['nivel_educativo'] = df['nivel_educativo'].astype('category').cat.rename_categories(['~Pr', 'Pr', 'Se', 'Su'])
	df['familiar_ad'] = df['familiar_ad'].astype('category').cat.rename_categories(['NoFam', 'Fam'])
	df['nivelrenta'] = df['nivelrenta'].astype('category').cat.rename_categories(['Baja', 'Media', 'Alta'])
	df['edad'] = pd.cut(df['edad'], range(0, 100, 10), right=False)
	
	# df['alfrut'] = df['alfrut'].astype('category').cat.rename_categories(['0', '1-2', '3-5','6-7'])
	# df['alcar'] = df['alcar'].astype('category').cat.rename_categories(['0', '1-2', '3-5','6-7'])
	# df['aldulc'] = df['aldulc'].astype('category').cat.rename_categories(['0', '1-2', '3-5','6-7'])
	# df['alverd'] = df['alverd'].astype('category').cat.rename_categories(['0', '1-2', '3-5','6-7'])
	#Diet
	nb_of_categories = 4
	#df['dietaproteica_cut'] = pd.cut(df['dietaproteica'],nb_of_categories)
	#df['dietagrasa_cut'] = pd.cut(df['dietagrasa'],nb_of_categories)
	df['dietaketo_cut'] = pd.cut(df['dietaketo'],nb_of_categories)
	df['dietaglucemica_cut'] = pd.cut(df['dietaglucemica'],nb_of_categories)
	df['dietasaludable_cut']= pd.cut(df['dietasaludable'],nb_of_categories)

	#in absolute numbers
	fig, ax = plt.subplots(1,5)
	fig.set_size_inches(20,5)
	fig.suptitle('Conversion by absolute numbers, for various demographics')
	d = df.groupby([target_variable, 'apoe']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0])
	d = df.groupby([target_variable, 'nivel_educativo']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[1])
	d = df.groupby([target_variable, 'familiar_ad']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[2])
	d = df.groupby([target_variable, 'nivelrenta']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[3])
	d = df.groupby([target_variable, 'edad']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[4])

	#in relative numbers
	fig, ax = plt.subplots(1,5)
	fig.set_size_inches(20,5)
	fig.suptitle('Conversion by relative numbers, for various demographics')
	d = df.groupby([target_variable, 'apoe']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0])
	d = df.groupby([target_variable, 'nivel_educativo']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1])
	d = df.groupby([target_variable, 'familiar_ad']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[2])
	d = df.groupby([target_variable, 'nivelrenta']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[3])
	d = df.groupby([target_variable, 'edad']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[4])
	plt.show()
	
	#diet in relative numbers
	fig, ax = plt.subplots(1,3)
	fig.set_size_inches(20,5)
	fig.suptitle('Conversion by relative numbers, for Alimentation')
	d = df.groupby([target_variable, 'dietaketo_cut']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0])
	#d = df.groupby([target_variable, 'dietagrasa_cut']).size().unstack(level=1)
	#d = d / d.sum()
	#p = d.plot(kind='bar', ax=ax[1])
	d = df.groupby([target_variable, 'dietaglucemica_cut']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1])
	d = df.groupby([target_variable, 'dietasaludable_cut']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[2])

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
	print('Normalizing the dataset using MinMaxScaler (values shifted and rescaled to ranging from 0 to 1) \n')
	
	scaler = preprocessing.MinMaxScaler()
	#http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
	scaler_std  = preprocessing.StandardScaler()
	scaler_rob = preprocessing.RobustScaler()
	#Standarization substracts the mean and divide by std so that the new distribution has unit variance.
	#Unlike min-max scaling, standardization does not bound values to a specific range this may be a problem
	#standardization is much less affected by outliers
	X_train_minmax = scaler.fit_transform(dataset)
	X_std = scaler_std.fit_transform(dataset)
	X_rob = scaler_rob.fit_transform(dataset)
	print("Orignal ndarray \n {}".format(dataset))
	#print("X_train_minmax \n {}".format(X_train_minmax))
	return [X_train_minmax, X_std, X_rob], [scaler, scaler_std, scaler_rob]


def run_split_dataset_in_train_test(X,y,test_size=None):
	""" run_split_dataset_in_train_test """
	if test_size is None:
		test_size = 0.2
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
	return X_train, X_test, y_train, y_test

def split_features_in_groups():
	""" split_features_in_groups
	Output: dictionaty 'group name':list of features"""
	dict_features = {}
	vanilla = ['sexo', 'lat_manual', 'edad', 'edad_ultimodx'] #remove apoe , edad_ultimodx
	#vanilla = ['sexo', 'lat_manual', 'nivel_educativo', 'edad']
	#sleep = ['hsnoct' , 'sue_dia' , 'sue_noc' , 'sue_con' , 'sue_man' , 'sue_suf' , 'sue_pro' , 'sue_ron' , 'sue_mov' , 'sue_rui' , 'sue_hor', 'sue_rec']
	sleep = ['sue_noc', 'sue_rec']
	anthropometric = ['imc'] #['pabd' , 'peso' , 'talla' , 'imc']
	#sensory = ['audi', 'visu']
	#intellectual = ['a01' , 'a02' , 'a03' , 'a04' , 'a05' , 'a06' , 'a07' , 'a08' , 'a09' , 'a10' , 'a11' , 'a12' , 'a13' , 'a14'] 
	intellectual = ['a02', 'a03', 'a08', 'a11', 'a12', 'a13', 'a14', 'relafami', 'relaamigo'] #social activities
	#demographics = ['sdhijos' , 'numhij' , 'sdvive' , 'sdeconom' , 'sdresid' , 'sdestciv']
	offspring = ['numhij' , 'sdvive' , 'sdeconom' , 'sdresid' , 'sdestciv']
	#professional = ['sdtrabaja' , 'sdocupac', 'sdatrb']
	professional = [ 'sdatrb']
	#cardiovascular = ['hta', 'hta_ini', 'glu', 'lipid', 'tabac', 'tabac_ini', 'tabac_fin', 'tabac_cant', 'sp', 'cor', 'cor_ini', 'arri', 'arri_ini', 'card', 'card_ini']
	cardiovascular = ['hta', 'glu', 'lipid', 'tabac_cant', 'cor', 'arri',  'card']
	#ictus = ['tir', 'ictus', 'ictus_num', 'ictus_ini', 'ictus_secu', 'tce', 'tce_num', 'tce_ini', 'tce_con', 'tce_secu']
	ictus = ['ictus', 'tce', 'tce_con']
	diet = ['alfrut', 'alcar', 'alpesblan', 'alpeszul', 'alaves', 'alaceit', 'alpast', 'alpan', 'alverd', 'alleg', 'alemb', 'allact', 'alhuev', 'aldulc']
	diet = ['dietaketo', 'dietaglucemica', 'dietasaludable']
	wealth = ['renta', 'nivel_educativo', 'educrenta']
	#physical_exercise = ['a01', 'ejfre', 'ejminut']
	physical_exercise = ['physical_exercise'] #ejfre' * 'ejminut
	#family_history = ['dempad' , 'edempad' , 'demmad' , 'edemmad']
	#family_history = ['dempad' , 'demmad']
	familiar_ad = ['familiar_ad']
	dict_features = {'vanilla':vanilla, 'sleep':sleep,'anthropometric':anthropometric, 'familiar_ad':familiar_ad, \
	'intellectual':intellectual,'offspring':offspring,'professional':professional, \
	'cardiovascular':cardiovascular, 'ictus':ictus, 'diet':diet, 'wealth':wealth, 'familiar_ad':familiar_ad, 'physical_exercise':physical_exercise}

	return dict_features

def build_formula(features):
	""" build formula to be used for  run_feature_slection. 'C' for categorical features
	Args: None
	Outputs: formula"""
	#formula = 'conversion ~ '; formula += 'C(sexo) + C(nivel_educativo) + C(apoe)'; 
	#formula = 'conversion ~ '; formula += 'sexo + lat_manual + nivel_educativo + apoe '; 
	formula = 'conversionmci ~ '; formula += 'sexo + lat_manual + nivel_educativo  + edad_ultimodx + renta'; 
	#sleep
	#formula += '+ hsnoct + sue_dia + sue_noc + sue_con + sue_man + sue_suf + sue_pro +sue_ron+ sue_mov+sue_rui + sue_hor + sue_rec'
	formula += '+ sue_noc + sue_rec'

	#family history
	#formula += '+ dempad + edempad + demmad + edemmad'
	#formula += '+ dempad + demmad '
	formula += '+ familiar_ad '
	#Anthropometric measures
	#formula += '+ pabd + peso + talla + imc'
	formula += '+ imc'
	#sensory disturbances
	#formula += '+ audi + visu'
	#intellectual activities
	#formula += '+ a01 + a02 + a03 + a04 + a05 + a06 + a07 + a08 + a09 + a10 + a11 + a12 + a13 + a14'
	formula += '+ a02 + a03 + a08 + a11 + a12 + a13 + a14' 

	#demographics
	#formula += '+ sdhijos + numhij+ sdvive + sdeconom + sdresid + sdestciv'
	formula += '+ numhij+ sdvive + sdeconom + sdresid + sdestciv + sdatrb'

	#professional life
	#formula += '+ sdtrabaja + sdocupac + sdatrb'
	formula += '+ sdatrb'
	#cardiovascular risk
	#formula += '+ hta + hta_ini + glu + lipid + tabac + tabac_ini + tabac_fin + tabac_cant + sp + cor + cor_ini + arri + arri_ini + card + card_ini'
	formula += '+ hta + glu + lipid + tabac_cant + cor + arri + card'
	#brain conditions that may affect cog performance
	#formula += '+ tir + ictus + ictus_num + ictus_ini + ictus_secu + tce + tce_num + tce_ini + tce_con + tce_secu'
	formula += '+ ictus + tce + tce_con'
	#diet
	#formula += '+ alfrut + alcar + alpesblan + alpeszul + alaves + alaceit + alpast + alpan + alverd + alleg + alemb + allact + alhuev + aldulc'
	formula += '+ dietaketo + dietasaludable + dietaglucemica'
	#physical exercise
	#formula += ' + ' + ' + '.join(selcols('ejfre_visita',1,1));formula += ' + ' + ' + '.join(selcols('ejminut_visita',1,1));
	formula += '+ physical_exercise'
	#scd
	formula += ' + ' + ' + '.join(selcols('scd_visita',1,1)); #formula += ' + ' + ' + '.join(selcols('peorotros_visita',1,1));
	#formula += ' + ' + ' + '.join(selcols('tpoevol_visita',1,1)); formula += ' + ' + ' + '.join(selcols('edadinicio_visita',1,1)); 
	formula += ' + ' + ' + '.join(selcols('preocupacion_visita',1,1)); #formula += ' + ' + ' + '.join(selcols('eqm06_visita',1,1));
	#formula += ' + ' + ' + '.join(selcols('eqm07_visita',1,1));formula += ' + ' + ' + '.join(selcols('eqm09_visita',1,1));
	#formula += ' + ' + ' + '.join(selcols('eqm10_visita',1,1));
	#cognitive complaints
	#formula += ' + ' + ' + '.join(selcols('eqm81_visita',1,1));formula += ' + ' + ' + '.join(selcols('eqm86_visita',1,1));
	#formula += ' + ' + ' + '.join(selcols('eqm82_visita',1,1));formula += ' + ' + ' + '.join(selcols('eqm83_visita',1,1));
	#formula += ' + ' + ' + '.join(selcols('eqm84_visita',1,1));formula += ' + ' + ' + '.join(selcols('eqm85_visita',1,1));
	formula += ' + ' + ' + '.join(selcols('act_aten_visita',1,1));formula += ' + ' + ' + '.join(selcols('act_orie_visita',1,1));
	formula += ' + ' + ' + '.join(selcols('act_mrec_visita',1,1));#formula += ' + ' + ' + '.join(selcols('act_memt_visita',1,1));
	formula += ' + ' + ' + '.join(selcols('act_visu_visita',1,1));formula += ' + ' + ' + '.join(selcols('act_expr_visita',1,1));
	formula += ' + ' + ' + '.join(selcols('act_comp_visita',1,1));#formula += ' + ' + ' + '.join(selcols('act_ejec_visita',1,1));
	#formula += ' + ' + ' + '.join(selcols('act_prax_visita',1,1));
	#cognitive performance
	# formula += ' + ' + ' + '.join(selcols('mmse_visita',1,1));formula += ' + ' + ' + '.join(selcols('reloj_visita',1,1));
	# formula += ' + ' + ' + '.join(selcols('faq_visita',1,1));
	# formula += ' + ' + ' + '.join(selcols('fcsrtrl1_visita',1,1));formula += ' + ' + ' + '.join(selcols('fcsrtrl2_visita',1,1));
	# formula += ' + ' + ' + '.join(selcols('fcsrtrl3_visita',1,1));formula += ' + ' + ' + '.join(selcols('fcsrtlibdem_visita',1,1));
	# formula += ' + ' + ' + '.join(selcols('p_visita',1,1));formula += ' + ' + ' + '.join(selcols('animales_visita',1,1));
	# formula += ' + ' + ' + '.join(selcols('cn_visita',1,1));formula += ' + ' + ' + '.join(selcols('cdrsum_visita',1,1));
	#psychioatric symptomes
	formula += ' + ' + ' + '.join(selcols('gds_visita',1,1));
	formula += ' + ' + ' + '.join(selcols('stai_visita',1,1));
	#formula += ' + ' + ' + '.join(selcols('act_depre_visita',1,1))
	#formula += ' + ' + ' + '.join(selcols('act_ansi_visita',1,1));
	formula += ' + ' + ' + '.join(selcols('act_apat_visita',1,1));
	#social engagement
	#formula += ' + ' + ' + '.join(selcols('relafami_visita',1,1)); formula += ' + ' + ' + '.join(selcols('relaamigo_visita',1,1));
	formula += ' + relaamigo + relafami'
	#formula += ' + ' + ' + '.join(selcols('relaocio_visita',1,1));formula += ' + ' + ' + '.join(selcols('rsoled_visita',1,1));

	#quality of life
	formula += ' + ' + ' + '.join(selcols('valcvida2_visita',1,1));formula += ' + ' + ' + '.join(selcols('valsatvid2_visita',1,1));
	formula += ' + ' + ' + '.join(selcols('valfelc2_visita',1,1));
	formula += ' + ' + ' + '.join(selcols('eq5dsalud_visita',1,1));formula += ' + ' + ' + '.join(selcols('eq5deva_visita',1,1));

	#formula += ' + ' + ' + '.join(selcols('eq5dmov_visita',1,1));
	#formula += ' + ' + ' + '.join(selcols('eq5dcp_visita',1,1));formula += ' + ' + ' + '.join(selcols('eq5dact_visita',1,1));
	#formula += ' + ' + ' + '.join(selcols('eq5ddol_visita',1,1));formula += ' + ' + ' + '.join(selcols('eq5dans_visita',1,1));
	
	#[ 'edadinicio_visita1', 'tpoevol_visita1', 'peorotros_visita1', 'eqm06_visita1', 'eqm07_visita1', 'eqm81_visita1', 'eqm82_visita1', 
	#'eqm83_visita1', 'eqm84_visita1', 'eqm85_visita1', 'eqm86_visita1', 'eqm09_visita1', 'eqm10_visita1','act_memt_visita1', 
	#'act_ejec_visita1', 'act_prax_visita1', 'act_depre_visita1', 'act_ansi_visita1', 'eq5dmov_visita1', 'eq5dcp_visita1', 
	#'eq5dact_visita1', 'eq5ddol_visita1', 'eq5dans_visita1', 'relaocio_visita1', 'rsoled_visita1']
	return formula


def selcols(prefix, a=1, b=5):
	""" selcols: return list of str of longitudinal variables
	Args:prefix name of the variable
	a: initial index year, b last year
	Output: list of feature names
	Example: selcols('scd_visita',1,3) returns [scd_visita,scd_visita2,scd_visita3] """
	return [prefix+str(i) for i in np.arange(a,b+1)]

def run_feature_ranking(df, formula, nboffeats=12, scaler=None):
	""" run_feature_ranking(X) : builds the design matrix for feature ranking (selection)
	Args: pandas dataframe scaled and MinMax. NO NEGATIVE VALUES
	Outputs: design matrix for given formula"""
	# Construct a single design matrix given a formula_ y ~ X
	print("The formula is:", formula)
	# patsy dmatrices, construct a single design matrix given a formula_like and data.

	y, X = dmatrices(formula, data=df, return_type='dataframe')
	#convert dataframe into Series
	y = y.iloc[:, 0]
	# feature scaling
	#if scaler is None:
	#	scaler = preprocessing.MinMaxScaler()
	#	scaler.fit(X)
	""" select top nboffeats features and find top indices from the formula  """
	#nboffeats = 12
	warnings.simplefilter(action='ignore', category=(UserWarning,RuntimeWarning))
	# sklearn.feature_selection.SelectKBest, select k features according to the highest scores
	# SelectKBest(score_func=<function f_classif>, k=10)
	# f_classif:ANOVA F-value between label/feature for classification tasks.
	# mutual_info_classif: Mutual information for a discrete target.
	# chi2: Chi-squared stats of non-negative features for classification tasks.
	function_f_classif = ['f_classif', 'mutual_info_classif', 'chi2', 'f_regression', 'mutual_info_regression']
	#selector = SelectKBest(chi2, nboffeats)
	selector = SelectKBest(f_classif, nboffeats)

	#Run score function on (X, y) and get the appropriate features.

	selector.fit(X, y)
	# scores_ : array-like, shape=(n_features,) pvalues_ : array-like, shape=(n_features,)
	top_indices = np.nan_to_num(selector.scores_).argsort()[-nboffeats:][::-1]
	print("Selector {} scores:",nboffeats, selector.scores_[top_indices])
	print("Top features:\n", X.columns[top_indices])

	# Pipeline of transforms with a final estimator.Sequentially apply a list of transforms and a final estimator.
	# sklearn.pipeline.Pipeline
	#preprocess = Pipeline([('anova', selector), ('scale', scaler)])
	#print("Estimator parameters:", preprocess.get_params())
	# Fit the model and transform with the final estimator. X =data to predict on
	#preprocess.fit(X,y)
	# transform: return the transformed sample: array-like, shape = [n_samples, n_transformed_features]
	#X_prep = preprocess.transform(X)
	#return X_prep
	# model selection
	#X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.3, random_state=42)
	#return X_prep, X_train, X_test, y_train, y_test

def plot_dendogram(Z, explanatory_features=None):
	"""  plot_dendogram: 
	Args: Z linkage matrix
	output:
	https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
	"""
	# calculate full dendrogram
	plt.figure(figsize=(25, 10))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('sample index')
	plt.ylabel('distance')

	dendrogram(
		Z,
		truncate_mode='lastp',  # show only the last p merged clusters
		p=20,  # show only the last p merged clusters
		leaf_rotation=90.,  # rotates the x axis labels
		leaf_font_size=8.,  # font size for the x axis labels
		show_contracted=True,  # to get a distribution impression in truncated branches
		labels=explanatory_features
	)
	plt.show()
	# a huge jump in distance is typically what we're interested in if we want to argue for a certain number of clusters

def run_hierarchical_clustering(X, explanatory_features=None):
	""" run_hierarchical_clustering: hierarchical clustering algorithm and plot dendrogram
	Args:X ndarray X_train and X_test concatenated
	Output:
	Exxample: run_hierarchical_clustering(X) # X is the dataset (without the labels)
	"""
	# generate the linkage matrix using Ward variance minimization algorithm.
	#no matter the method and metric specified, linkage() function will use that method and metric
	#to calculate the distances of the clusters (starting with your n individual samples
	#as singleton clusters)) and in each iteration will merge the two clusters which have
	#the smallest distance according the selected method and metric
	#[idx1, idx2, dist, sample_count] 
	#Plotting a Dendrogram. X=X.T clustering of features, X clustering of subjects
	import pylab
	from sklearn.metrics.pairwise import euclidean_distances
	#labels = ['a','b','c','d']
	X = X.T
	D = euclidean_distances(X, X)
	if sum(np.diagonal(D)) > 0:
		raise SomethingError("The distance matrix is not diagonal!!!!")
	Z = linkage(D, 'centroid') #method='centroid'
	# Cophenetic Correlation Coefficient compares (correlates) the actual 
	# pairwise distances of all your samples to those implied by the hierarchical clustering
	c, coph_dists = cophenet(Z, pdist(X))
	print("Cophenetic Correlation Coefficient ={:.3f} The closer the value is to 1, the better the clustering preserves the original distances".format(c))
	plot_dendogram(Z, explanatory_features)
	# Plot dendogram and distance matrix
	#https://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python
	# Compute and plot first dendrogram.
	fig = pylab.figure(figsize=(8,8))
	ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
	Y = linkage(D,  method='centroid')
	Z1 = dendrogram(Y, orientation='right', labels=explanatory_features)
	print("values passed to leaf_label_func\nleaves : ", Z1["leaves"])
	#temp = {Z1["leaves"][ii]: labels[ii] for ii in range(len(Z1["leaves"]))}
	ax1.set_xticks([])
	#ax1.set_yticks([])
	ax1.set_title('Centroid linkage')
	# Compute and plot second dendrogram.
	ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
	Y = linkage(D, method='ward')
	Z2 = dendrogram(Y, labels=explanatory_features)
	#ax2.set_xticks([])
	#ax2.set_yticks([])
	ax2.set_title('Ward linkage')
	# Plot distance matrix.
	axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
	idx1 = Z1['leaves']
	idx2 = Z2['leaves']
	D = D[idx1,:]
	D = D[:,idx2]
	
	im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
	axmatrix.set_xticks([])
	axmatrix.set_yticks([])
	# Plot colorbar.
	axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
	pylab.colorbar(im, cax=axcolor)
	fig.show()
	fig.savefig('/Users/jaime/github/papers/vallecas5years/figures/dendrogram.png')
	pdb.set_trace()

def run_truncatedSVD(X_train, y_train, X_test, y_test):
	""" run_truncatedSVD: dimensionality reduction for sparse matrices (Single Value decompotitoion)
	truncated because the dimension can be chosen (k)
	Args:X_train, y_train, X_test, y_test
	Output: reduced data points"""
	import matplotlib.patches as mpatches
	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	datadimension = X_train.shape[1]
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	svd = TruncatedSVD(n_components=2)
	X_reduced = svd.fit_transform(X_all)
	# scatter plot of original and reduced data
	fig = plt.figure(figsize=(5, 5))
	plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_all, s=50, edgecolor='k')
	plt.title("truncated SVD reduction (2D) of N="+ str(datadimension) +" dimensions")
	plt.axis('off')
	yellow_patch = mpatches.Patch(color='gold', label='converters')
	brown_patch = mpatches.Patch(color='indigo', label='Non converters')
	plt.legend(handles=[yellow_patch, brown_patch])
	print('svd.explained_variance_ratio_={}', svd.explained_variance_ratio_)  
	print('svd.explained_variance_ratio_.sum()={}', svd.explained_variance_ratio_.sum())  
	print('svd.singular_values_={}', svd.singular_values_) 
	return X_reduced

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
	explained_variance = 0.95
	optimal_comp = np.where(cumvar > explained_variance )[0][0]
	print("\nWith at least {} components we explain {}'%'' of the {} dimensional input data".format(optimal_comp,explained_variance,Xy_df.values.shape[1] ))
	plt.plot(cumvar)     # ditto, but with red plusses)
	horline = np.repeat(explained_variance, len(cumvar))
	plt.plot(np.arange(0,len(cumvar)), horline, 'r--')
	plt.xlabel('number of components')
	plt.ylabel('cumulative explained variance')
	plt.title('PCA explained variance')

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
	#print("PCA {} components_:{} \n".format(pca.n_components_, pca.components_))
	print("PCA explained variance ratio:{} \n".format(pca.explained_variance_ratio_))
	# plot the principal components
	targets = Xy_df[target_label].unique().astype(int)
	if optimal_comp > 2:
		plt.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c=Xy_df[target_label].astype(int), edgecolor='none', alpha=0.6)
	else:
		plt.scatter(projected[:, 0], projected[:, 1], c=Xy_df[target_label].astype(int), edgecolor='none', alpha=0.6)
	ax.grid()
	ax.set_xlabel('PC 1', fontsize = 10)
	ax.set_ylabel('PC 2', fontsize = 10)
	if optimal_comp > 2:
		ax.set_zlabel('PC 3', fontsize = 10)
	msgtitle = str(pca.n_components_) + ' components PCA for variance=' + str(explained_variance*100)+'%'
	ax.set_title(msgtitle, fontsize = 10)
	#Noise filtering https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
	return pca, projected
	

def compute_metrics_estimator(estimator, X_train, y_train, X_test,y_test):
	""" compute_metrics_estimator compute metrics between a pair of arrays y_pred and y_test
	Evaluate a score comparing y_pred=estimator().https://www.kaggle.com/ganakar/using-ensemble-methods-and-smote-samplingfit(X_train)predict(X_test) from y_test
	Args:estimator(object),X_test,y_test
	Output: dictionary label:value
	Example: compute_metrics_estimator(estimator,X_test,y_test) the estimator has been previously fit"""
	print("Computing metrics for estimator {}".format(estimator.__class__.__name__))
	y_pred_train = estimator.predict(X_train)
	y_pred_test = estimator.predict(X_test)
	scores = {'accuracy_test':accuracy_score(y_test, y_pred_test), 'accuracy_train':accuracy_score(y_train, y_pred_train),\
	'matthews_corrcoef_test':matthews_corrcoef(y_test, y_pred_test),'matthews_corrcoef_train':matthews_corrcoef(y_train, y_pred_train),\
	'f1_test':f1_score(y_test, y_pred_test),'f1_train':f1_score(y_train, y_pred_train),\
	'cohen_kappa_test':cohen_kappa_score(y_test, y_pred_test),'cohen_kappa_train':cohen_kappa_score(y_train, y_pred_train), \
	'recall_test': recall_score(y_test, y_pred_test),'recall_train': recall_score(y_train, y_pred_train), \
	'precision_test':precision_score(y_test, y_pred_test),'precision_train':precision_score(y_train, y_pred_train), \
	'roc_auc_test':roc_auc_score(y_test, y_pred_test),'roc_auc_train':roc_auc_score(y_train, y_pred_train),\
	'log_loss_test':log_loss(y_test, y_pred_test),'log_loss_train':log_loss(y_train, y_pred_train),\
	'confusion_matrix_test':confusion_matrix(y_test, y_pred_test).T, 'confusion_matrix_train':confusion_matrix(y_train, y_pred_train).T}
	print('Estimator metrics:{}'.format(scores))
	return scores

def evaluate_learners_metrics(learners, X_train, y_train, X_test,y_test): 
	""" evaluate_learners_metrics
	Args:learners list of learners fit, results array each element is the metric for a learner eg results
	https://www.kaggle.com/ganakar/using-ensemble-methods-and-smote-sampling
	"""
	import random
	import matplotlib.patches as mpatches
	results = {}; appended_results = []
	for key, value in learners.iteritems():
		clf_name = value.__class__.__name__
		results[clf_name] = {}
		results[clf_name] = compute_metrics_estimator(value, X_train, y_train, X_test,y_test)
		#plot most important features
		
	#plot results
	#fig, ax = plt.subplots(2, 8, figsize = (18,12))
	fig, ax = plt.subplots(2, 8, figsize = (12,7))
	tit_label={0:'Training ',1:'Testing '}
	#bar_width = 0.1
	bar_width = 0.2
	plt.tick_params(
	axis='x',          # changes apply to the x-axis
	which='both',      # both major and minor ticks are affected
	bottom='off',      # ticks along the bottom edge are off
	top='off',         # ticks along the top edge are off
	labelbottom='off') # labels along the bottom edge are off
	#len(colors) = len(learners)
	colors = ['#5F9EA0','#6495ED','#90EE90','#9ACD32','#90AC32','#40AC32', '#FFFF00','#800000','#000000','#0000FF']
	#random.shuffle(colors)
	colors = colors[:len(learners)]
	for k, learner in enumerate(results.keys()):
		for j, metric in enumerate(['accuracy_train','matthews_corrcoef_train',\
			'cohen_kappa_train','recall_train','precision_train','f1_train','roc_auc_train','log_loss_train']):
			
			ax[0, j].bar(k*bar_width, results[learner][metric], width = bar_width, color = colors[k])
			ax[0, j].set_xlim((-0.1, .9))
			ax[0,j].set_facecolor('white')
			plt.setp(ax[0,j].get_xticklabels(),visible=False)
			ax[0,j].set_xticks([])
		for j, metric in enumerate(['accuracy_test','matthews_corrcoef_test',\
			'cohen_kappa_test','recall_test','precision_test','f1_test','roc_auc_test','log_loss_test']):
			ax[1, j].bar(k*bar_width, results[learner][metric], width = bar_width, color = colors[k])
			ax[1, j].set_xlim((-0.1, .9))
			ax[1,j].set_facecolor('white')
			ax[1,j].set_xticks([])

	for r in range(2):
		#Add unique y-labels
		ax[r, 0].set_ylabel("Accuracy")
		ax[r, 1].set_ylabel("Matthews")
		ax[r, 2].set_ylabel("Cohen kappa")
		ax[r, 3].set_ylabel("Recall")
		ax[r, 4].set_ylabel("Precision")
		ax[r, 5].set_ylabel("F1")
		ax[r, 6].set_ylabel("ROC AUC")
		ax[r, 7].set_ylabel("Log Loss")
		# Add titles
		# ax[r, 0].set_title(tit_label[r]+"Accuracy ")
		# ax[r, 1].set_title(tit_label[r]+"Matthews ")
		# ax[r, 2].set_title(tit_label[r]+"Cohen ")
		# ax[r, 3].set_title(tit_label[r]+"Recall")
		# ax[r, 4].set_title(tit_label[r]+"Precision")
		# ax[r, 5].set_title(tit_label[r]+"F1 ")
		# ax[r, 6].set_title(tit_label[r]+"ROC AUC")
		# ax[r, 7].set_title(tit_label[r]+"Log Loss")
	# Add titles
	ax[0, 3].set_title("Training metrics")
	ax[1, 3].set_title("Testing metrics")
	# Create patches for the legend
	
	patches = []
	for i, learner in enumerate(results.keys()):
		patches.append(mpatches.Patch(color = colors[i], label = learner))
		#plt.legend(handles = patches, bbox_to_anchor = (-2, 2.4), \
		#	loc = 'upper center', borderaxespad = 0., ncol = 4, fontsize = 'x-large')
	plt.legend(handles = patches,loc = 1 , fontsize = 6, borderaxespad=0.)
	#plt.suptitle("Performance Metrics for Four Supervised Learning Models", fontsize = 8)
	plt.tight_layout()
	plt.show()

def plot_feature_importance(learner, features):
	""" plot_feature_importance use booster in 3.2.4.3.5. sklearn.ensemble.GradientBoostingClassifier """
	from xgboost import plot_importance
	plt.figure(figsize=(6, 6))
	features_imp = pd.DataFrame(learner.feature_importances_, index=features)
	features_imp  = features_imp.sort_values(by=0, ascending=False)
	plot_importance(learner)
	print("The most important features are {}".format(features_imp.head(20)))


def compute_metrics_estimator_with_cv(estimator,X_test,y_test,cv):
	""" compute_metrics_estimator_with_cv : evaluate a score with sklearn.model_selection.cross_val_score for X_test y_test 
	Args:estimator(object) to implement the fit,X_test (data to fit),y_test (target variable in case of supervised learning)
	,cv the number of splits (int) used StratifiedKFold 
	Output: scores_cv (dict)
	Example: compute_metrics_estimator_with_cv(estimator,X_test,y_test,10)"""
	kfold = KFold(n_splits=cv, random_state=0)
	acc_cv = cross_val_score(estimator, X_test, y_test, cv=cv, scoring='accuracy', verbose=1).mean()
	recall_cv = cross_val_score(estimator, X_test,y_test, cv=cv, scoring = 'recall', verbose=0).mean()
	f1_cv = cross_val_score(estimator, X_test,y_test, cv=cv, scoring = 'f1', verbose=0).mean()
	rocauc_cv = cross_val_score(estimator, X_test,y_test, cv=cv, scoring = 'roc_auc', verbose=0).mean()
	precision_cv = cross_val_score(estimator, X_test,y_test, cv=cv, scoring = 'precision', verbose=0).mean()
	scores_cv = {'accuracy_cv':acc_cv,'recall_cv':recall_cv,'f1_cv':f1_cv,'rocauc_cv':rocauc_cv,'precision_cv':precision_cv}
	print('Estimator metrics for cv={} is \n {}'.format(kfold,scores_cv))
	return scores_cv

def print_model_selection_metrics(X_train, y_train, X_test, y_test, modsel=None):
	""" print_model_selection_metrics
	Args: X_train, y_train, X_test, y_test, modsel is int, modsel==0 Train/Test modsel>0 is the number fo folds
	Example: print_model_selection_metrics(X_train, y_train, X_test, y_test, 0) Train/Test split model selection 
	print_model_selection_metrics(X_train, y_train, X_test, y_test, 5) K=5 Fold model seelction 
	"""
	models = []; names = []; scores_acc_list = [];scores_auc_list = [];scores_recall_list=[];scores_matthews_list=[];scores_f1score_list=[];scores_precision_list=[]
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('SVC', SVC()))
	models.append(('LR', LogisticRegression()))
	models.append(('DT', DecisionTreeClassifier()))
	models.append(('GNB', GaussianNB()))
	models.append(('RF', RandomForestClassifier()))
	models.append(('GB', GradientBoostingClassifier()))
	if modsel == 0:
		print("Model selection accuracy based for Train/Test split)")
		for name, model in models:
			model.fit(X_train, y_train)
			y_pred = model.predict(X_test)
			scores_acc_list.append(accuracy_score(y_test, y_pred))
			scores_matthews_list.append(matthews_corrcoef(y_test, y_pred))
			scores_f1score_list.append(f1_score(y_test, y_pred))
			names.append(name)
		pdb.set_trace()
		tr_split = pd.DataFrame({'Names': names, 'Score_acc': scores_acc_list, 'Score_matthews':scores_matthews_list,'Score_f1score':scores_f1score_list})
		data_to_plot = tr_split
		print(tr_split)
	elif type(modsel) is int and modsel >0:
		print("Model selection accuracy based for k={}-Fold Cross Validation".format(modsel))
		for name, model in models:
			kfold = KFold(n_splits=modsel, random_state=0)
			#score_acc = 
			scores_acc_list.append(cross_val_score(model, X_test, y_test, cv=modsel, scoring='accuracy').mean())
			# note if many Folds the y pred may have only one class and auc retund error
			scores_auc_list.append(cross_val_score(model, X_test,y_test, cv=modsel, scoring = 'roc_auc').mean())
			scores_recall_list.append(cross_val_score(model, X_test,y_test, cv=modsel, scoring = 'recall').mean())
			scores_f1score_list.append(cross_val_score(model, X_test,y_test, cv=modsel, scoring = 'f1').mean())
			scores_precision_list.append(cross_val_score(model, X_test,y_test, cv=modsel, scoring = 'precision').mean())
			names.append(name)
		kf_cross_val = pd.DataFrame({'Name': names, 'Score_acc': scores_acc_list, 'Score_auc':scores_auc_list,'Score_recall':scores_recall_list, 'Score_f1':scores_f1score_list, 'Score_precision':scores_precision_list})
		data_to_plot = kf_cross_val
		print(kf_cross_val)
	elif type(modsel) is dict:
		for name, model in models[1:2]:
			# only for SVC we can use gamma parameter
			print("name {} model {}".format(name, model))
			grid_clf_acc = GridSearchCV(model, param_grid = modsel)
			grid_clf_acc.fit(X_train, y_train)
			#optimize for acc
			y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test)
			print('Grid best parameter model:{} (max. accuracy):{}'.format(name, grid_clf_acc.best_params_))
			print('Grid best score model:{} (accuracy):{}'.format(name, grid_clf_acc.best_score_)) 
			#optimize for auc
			grid_clf_auc = GridSearchCV(model, param_grid = modsel, scoring='roc_auc')
			grid_clf_auc.fit(X_train, y_train)
			y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test)
			print('Test set AUC: ', roc_auc_score(y_test,y_decision_fn_scores_auc))
			print('Grid best parameter model:{} (max. auc):{}'.format(name, grid_clf_auc.best_params_))
			print('Grid best score model:{} (auc):{}'.format(name,grid_clf_auc.best_score_))
			#the gamma parameter can be equal but also different depending on the metric used
	# print("Plot the accuracy scores nicely")
	# metric_to_plot = 'Score_matthews'
	# axis = sns.barplot(x = 'Name', y = 'Score', data = data_to_plot[metric_to_plot])
	# axis.set(xlabel='Classifier', ylabel=metric_to_plot)
	# for p in axis.patches:
	# 	height = p.get_height()
	# 	axis.text(p.get_x() + p.get_width()/2, height + 0.005,'{:1.4f}'.format(height), ha="center")
	# plt.show()

def compare_against_dummy_estimators(estimators, estimatorlabels, X_test, y_test, dummies_score):
	""" compare_against_dummy_estimators: compare test score of estimators against dummy estimators
	Args: estimators, estimatorlabels, X_test, y_test, dummies_score
	Output: 
	Example: compare_against_dummy_estimators([knn], ['knn'], X_test, y_test, {'uniform':0,9})
	"""

	scores_to_compare = []
	#for estobj in estimators:
	for key, value in estimators.iteritems():
		if key is not 'deepnetwork_res':
			scores_to_compare.append(value.score(X_test, y_test))
	scores_to_compare = scores_to_compare + dummies_score.values()
	estimatorlabels = estimatorlabels + dummies_score.keys()

	fig, ax = plt.subplots()
	x = np.arange(len(estimators) + len(dummies_score.keys()))
	#histograme with accuracy of estimators compared to dummies
	barlist = plt.bar(x,scores_to_compare)
	#colour red the actual estimators (dummies in blue)
	for i in range(0,len(estimators)):
		barlist[i].set_color('r') 
	#plt.xticks(x, ('knn', 'dummy-uniform', 'dummy-cte0', 'dummy-cte1'))
	plt.xticks(x, estimatorlabels)
	plt.ylabel('score')
	plt.ylabel('X,y test score vs dummy estimators')
	plt.show()

def build_dummy_scores(X_train, y_train, X_test, y_test):
	""" build_dummy_scores: When doing supervised learning, a simple sanity check consists of 
	comparing one's estimator against simple rules of thumb. DummyClassifier implements such strategies(stratified
	most_frequent, prior, uniform, constant). Used for imbalanced datasets
	Args: X_train, X_test, y_train, y_test
	Outputs: dict of dummy estimators"""

	dummy_strategy = ['uniform', 'constant']
	dummies =[]; dummy_scores =[]
	for strat in dummy_strategy:
		if strat is 'constant':
			for const in range(0,2):
				estimator_dummy = DummyClassifier(strategy='constant', random_state=0, constant=const)
				estimator_dummy = estimator_dummy.fit(X_train, y_train)
				dummies.append(estimator_dummy)
				dscore = estimator_dummy.score(X_test, y_test)
				dummy_scores.append(dscore)
				print("Score of Dummy {}={} estimator={}".format(strat, const,dscore ))
		else:
			estimator_dummy = DummyClassifier(strategy=strat, random_state=0)
			estimator_dummy = estimator_dummy.fit(X_train, y_train)
			dummies.append(estimator_dummy)
			dscore = estimator_dummy.score(X_test, y_test)
			dummy_scores.append(dscore)
			print("Score of Dummy {} estimator={}".format(strat, dscore))
	dict_dummy_scores = {'uniform':dummy_scores[0] , 'constant0':dummy_scores[1],'constant1':dummy_scores[2]}
	return dict_dummy_scores

def resampling_SMOTE(X_train, y_train):
	""" resampling_SMOTE: """
	from imblearn.over_sampling import SMOTE
	sm = SMOTE(ratio='minority',random_state=1234)
	X_resampled_train, y_resampled_train = sm.fit_sample(X_train, y_train)
	print('---------------Resampled data statistics---------------')
	converters = sum(y_resampled_train)
	converters_ratio = converters/y_resampled_train.shape[0]
	nonconverters_ratio = 1-converters_ratio
	print('Total number of subjects : {} '.format(len(y_resampled_train)))
	print('Total number of non converters : {}'.format(sum(y_resampled_train==0)))
	print('Total number of converters : {}'.format(sum(y_resampled_train==1)))
	print('Percent of non converters is : {:.4f}%,  converters is : {:.4f}%'.format(nonconverters_ratio*100, converters_ratio*100))
	return X_resampled_train, y_resampled_train
	
	
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

def run_sgd_classifier(X_train, y_train, X_test, y_test, loss,cv):
	"""SGD_classifier: Stochastic Gradient Descent classifier. Performs Log Regression and/or SVM (loss parameter) with SGD training 
	Args:X_train,y_train,X_test,y_test, loss = 'hinge' linear Support Vector Machine,loss="log": logistic regression
	Output: SGD fitted estimator
	Example: run_SGD_classifier(X_train, y_train, X_test, y_test, 'hinge|'log')"""
	#find an opimum value of 'alpha' by either looping over different values of alpha and evaluating the performance over a validation set
	#use gridsearchcv
	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	tuned_parameters = {'alpha': [10 ** a for a in range(-6, -2)]}
	#class_weight='balanced' addresses the skewness of the dataset in terms of labels
	# loss='hinge' LSVM,  loss='log' gives logistic regression, a probabilistic classifier
	# ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’.
	clf = GridSearchCV(SGDClassifier(loss=loss, penalty='elasticnet',l1_ratio=0.15, n_iter=5, shuffle=True, verbose=False, n_jobs=1, \
		average=False, class_weight='balanced',random_state=0),tuned_parameters, cv=cv, scoring='f1_macro').fit(X_train, y_train)
	if loss is 'log':
		y_train_pred = clf.predict_proba(X_train)[:,1]
		y_test_pred = clf.predict_proba(X_test)[:,1]
	else:
		y_train_pred = clf.predict(X_train)
		y_test_pred = clf.predict(X_test)
	# predict the response
	y_pred = [int(a) for a in clf.predict(X_test)]
	num_correct = sum(int(a == ye) for a, ye in zip(y_pred, y_test))
	print("SGDClassifier. The best alpha is:{}".format(clf.best_params_)) 
	# plot learning curve
	title='SGD learning curve'+loss
	plot_learning_curve(clf, title, X_all, y_all, n_jobs=1)
	print('Accuracy of sgd {} alpha={} classifier on training set {:.2f}'.format(clf.best_params_,loss, clf.score(X_train, y_train)))
	print('Accuracy of sgd {} alpha={} classifier on test set {:.2f}'.format(clf.best_params_,loss, clf.score(X_test, y_test)))
	print(classification_report(y_test, y_pred))
	#plot auc
	fig,ax = plt.subplots(1,3)
	fig.set_size_inches(15,5)
	plot_cm(ax[0], y_train, y_train_pred, [0,1], 'sgd Confusion matrix (TRAIN) '+loss, 0.5)
	plot_cm(ax[1], y_test, y_test_pred,   [0,1], 'sgd Confusion matrix (TEST) '+loss, 0.5)
	plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, 0.5)
	plt.tight_layout()
	plt.show()
	return clf

def run_kneighbors(X_train, y_train, X_test, y_test, kneighbors=None):
	""" kneighbors_classifier : KNN is non-parametric, instance-based and used in a supervised learning setting. 
	Minimal training but expensive testing.
	Args:  X_train, y_train, X_test, y_test, kneighbors
	Output: knn estimator"""
	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	myList = list(range(1,50)) #odd number (49) to avoid tie of points
	# subsetting just the odd ones
	neighbors = filter(lambda x: x % 2 != 0, myList)
	# perform 10-fold cross validation
	cv_scores = [] # list with x validation scores
	cv = 5
	for k in neighbors:
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
		cv_scores.append(scores.mean())
	MSE = [1 - x for x in cv_scores]
	# determining best k
	optimal_k = neighbors[MSE.index(min(MSE))]
	print('The optimal for k-NN algorithm cv={} is k-neighbors={}'.format(cv, optimal_k))
	
	# instantiate learning model
	optimal_k = 5
	knn = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance').fit(X_train, y_train)
	y_train_pred = knn.predict_proba(X_train)[:,1]
	y_test_pred = knn.predict_proba(X_test)[:,1]
	y_pred = [int(a) for a in knn.predict(X_test)]
	num_correct = sum(int(a == ye) for a, ye in zip(y_pred, y_test))
	print("Baseline classifier using %s-NN: %s of %s values correct." % (optimal_k, num_correct, len(y_test)))
	# predict the response
	y_pred = knn.predict(X_test)
	# plot learning curve
	plot_learning_curve(knn, 'kNN learning curve', X_all, y_all, cv = cv, n_jobs=1)
	print('Accuracy of kNN classifier on training set {:.2f}'.format(knn.score(X_train, y_train)))
	print('Accuracy of kNN classifier on test set {:.2f}'.format(knn.score(X_test, y_test)))
	print(classification_report(y_test, y_pred))
	
	#plot auc
	fig,ax = plt.subplots(1,3)
	fig.set_size_inches(15,5)
	plot_cm(ax[0],  y_train, y_train_pred, [0,1], 'kNN Confusion matrix (TRAIN) k='+str(optimal_k), 0.5)
	plot_cm(ax[1],  y_test, y_test_pred,   [0,1], 'kNN Confusion matrix (TEST) k='+str(optimal_k), 0.5)
	plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, 0.5)
	plt.tight_layout()
	plt.show()
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

def run_decision_tree(X_train, y_train, X_test, y_test, X_features, target_variable):
	""" run_decision_tree : Bagging algorithm 
	Args:
	Output"""
	import pygraphviz as pgv
	import pydotplus
	import collections
	from IPython.display import Image

	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	dectree = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced').fit(X_train, y_train)
	y_train_pred = dectree.predict_proba(X_train)[:,1]
	y_test_pred = dectree.predict_proba(X_test)[:,1]
	y_pred = [int(a) for a in dectree.predict(X_test)]
	num_correct = sum(int(a == ye) for a, ye in zip(y_pred, y_test))
	print("Baseline classifier using DecisionTree: %s of %s values correct." % (num_correct, len(y_test)))
	# plot learning curve
	plot_learning_curve(dectree, 'dectree learning curve', X_all, y_all,  cv= 7, n_jobs=1)
	print('Accuracy of DecisionTreeClassifier classifier on training set {:.2f}'.format(dectree.score(X_train, y_train)))
	print('Accuracy of DecisionTreeClassifier classifier on test set {:.2f}'.format(dectree.score(X_test, y_test)))
	#confusion matrix on the test data
	conf_matrix = confusion_matrix(y_test, y_pred)
	print(conf_matrix)
	print('Accuracy of DT classifier on training set {:.2f}'.format(dectree.score(X_train, y_train)))
	print('Accuracy of DT classifier on test set {:.2f}'.format(dectree.score(X_test, y_test)))
	print(classification_report(y_test, y_pred))

	#print('Features importances: {}'.format(dectree.feature_importances_))
	# indices with feature importance > 0
	idxbools= dectree.feature_importances_ > 0; idxbools = idxbools.tolist()
	idxbools = np.where(idxbools)[0]
	impfeatures = []; importances = []
	for i in idxbools:
		print('Feature:{}, importance={}'.format(X_features[i], dectree.feature_importances_[i]))
		impfeatures.append(X_features[i])
		importances.append(dectree.feature_importances_[i])
	# plot features importances
	plt.figure(figsize=(5,5))
	plot_feature_importances(dectree,importances, impfeatures)
	plt.title('Decision tree features importances')
	plt.show()
	
	#plot CM and auc
	fig,ax = plt.subplots(1,3)
	fig.set_size_inches(15,5)
	plot_cm(ax[0],  y_train, y_train_pred, [0,1], 'DecisionTreeClassifier Confusion matrix (TRAIN)', 0.5)
	plot_cm(ax[1],  y_test, y_test_pred,   [0,1], 'DecisionTreeClassifier Confusion matrix (TEST)', 0.5)
	plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, 0.5)
	plt.tight_layout()
	plt.show()
	
	# plot decision tree
	dotgraph = plot_decision_tree(dectree, X_features, target_variable)
	G=pgv.AGraph()
	G.layout()
	G.draw('adspy_temp.dot')
	# Visualization of Decision Trees with graphviz
	labelstarget = np.unique(y_train).tolist()
	dot_data = tree.export_graphviz(dectree, out_file=None, filled=True, rounded=True, feature_names=X_features) #, class_names=['NC', 'C'])
	graph = pydotplus.graph_from_dot_data(dot_data)
	colors = ('turquoise', 'orange')
	edges = collections.defaultdict(list)
	for edge in graph.get_edge_list():
		 edges[edge.get_source()].append(int(edge.get_destination()))
	for edge in edges:
		edges[edge].sort()
		for i in range(2):
			dest = graph.get_node(str(edges[edge][i]))[0]
			dest.set_fillcolor(colors[i])   
	Image(graph.create_png())
	now = datetime.datetime.now()
	now = now.strftime("%Y-%m-%d.%H:%M")
	modelname = 'decision_tree' + '_'+ now +'.png'
	modelname = os.path.join('./ML_models', modelname)
	print('Saving the decision tree: {}', modelname)
	graph.write_png(modelname)
	return dectree


def run_gradient_boosting_classifier(X_train, y_train, X_test, y_test, type_of_algo=None):
	""" run_gradient_boosting_classifier: GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions
	Args:X_train, y_train, X_test, y_test, type_of_algo=AdaBoostClassifier,GradientBoostingClassifier,XGBClassifier
	Output:
	"""
	import xgboost as xgb

	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	ratio01s= int(sum(y_train==0)/sum(y_train==1))
	sample_weight = np.array([2 if i == 1 else 0 for i in y_train])
	#https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
	n_estimators = 100; learn_r = 0.001; max_depth=5;
	# By default do GradientBoostingClassifier
	clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate =learn_r, max_depth=max_depth, random_state=0).fit(X_train, y_train, sample_weight=None)
	if type_of_algo is 'XGBClassifier':
		clf = XGBClassifier(class_weight='balanced').fit(X_train, y_train,sample_weight=None)
	elif type_of_algo is 'AdaBoostClassifier':
			clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators,algorithm="SAMME.R", learning_rate=learn_r).fit(X_train, y_train,sample_weight=None)		
	elif type_of_algo is 'GradientBoostingClassifier':
		#by default
		type_of_algo = 'GradientBoostingClassifier' #clf= GradientBoostingClassifier(n_estimators=n_estimators, learning_rate =learn_r, max_depth=max_depth, random_state=0).fit(X_train, y_train, sample_weight=sample_weight)

	#XGBmodel = XGBClassifier(class_weight='balanced').fit(X_train, y_train,sample_weight=None)
	#XGBmodel = XGBClassifier(class_weight=np.array((1,2))).fit(X_train, y_train,sample_weight=None)
	#XGBmodel = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=n_estimators,algorithm="SAMME.R", learning_rate=learn_r).fit(X_train, y_train,sample_weight=None)
	y_train_pred = clf.predict_proba(X_train)[:,1]
	y_test_pred = clf.predict_proba(X_test)[:,1]
	y_pred = [int(a) for a in clf.predict(X_test)]
	num_correct = sum(int(a == ye) for a, ye in zip(y_pred, y_test))
	print("Baseline classifier using Boosting_classifier: %s:. %s of %s  values correct." % (type_of_algo, num_correct, len(y_test)))
	# plot learning curve
	plot_learning_curve(clf, 'Boosting learning curve', X_all, y_all, cv=7, n_jobs=2)
	print('Accuracy of {} classifier on training set {:.2f}'.format(type_of_algo, clf.score(X_train, y_train)))
	print('Accuracy of {} classifier on test set {:.2f}'.format(type_of_algo, clf.score(X_test, y_test)))
	print(classification_report(y_test, y_pred))
	
	#plot confusion matrix and AUC
	fig,ax = plt.subplots(1,3)
	fig.set_size_inches(15,5)
	plot_cm(ax[0],  y_train, y_train_pred, [0,1], 'Boosting CM (TRAIN)', 0.5)
	plot_cm(ax[1],  y_test, y_test_pred,   [0,1], 'Boosting CM (TEST)', 0.5)
	plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, 0.5)
	plt.tight_layout()
	plt.show()
	
	if type_of_algo is 'XGBClassifier':
		dtrain = xgb.DMatrix(X_train, label=y_train)
		dtest = xgb.DMatrix(X_test, label=y_test)
		num_round = 5
		evallist  = [(dtest,'eval'), (dtrain,'train')]
		param = {'objective':'binary:logistic', 'silent':1, 'eval_metric': ['error', 'logloss']}
		bst = xgb.train( param, dtrain, num_round, evallist)
		y_train_pred = bst.predict(dtrain)
		y_test_pred = bst.predict(dtest)
	return clf

def run_svm_classifier(X_train, y_train, X_test, y_test, kernel=None):
	""" run_svm_classifier: is the linear classifier with the maximum margin
	Args:X_train, y_train, X_test, y_test, kernel
	Output: svm object classifier """
	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	# SVM model, cost and gamma parameters for RBF kernel. out of the box
	# class_weight='balanced' kernel poly is horrible
	# ROC in SVM with explicit weights works horribly use balanced
	#svm = SVC(cache_size=1000, kernel='rbf',class_weight={0:.4, 1:.6}).fit(X_train, y_train)
	if kernel is 'rbf':
		print('Building SVC non linear Kernel...\n')
		#SVC with RBF kernel and optimal C and gamma parameters calculated
		#kernel  ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead. 
		#gamma Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.
		#C= default 1.0,(10)gamma default ’auto’= 1/n_features=1/57=0.017 (0.03125) 
		svm = SVC(C=10.0, gamma= 0.03125, cache_size=1000, class_weight='balanced', coef0=0.0,
		decision_function_shape='ovr', degree=3, kernel=kernel,
    	max_iter=-1, probability=False, random_state=None, shrinking=True,
    	tol=0.001, verbose=False).fit(X_train, y_train)
	else:
		#C : (default=1.0)
		#‘linear’, ‘poly’, ‘rbf’(default), ‘sigmoid’, ‘precomputed’ or a callable
		kernel = 'linear' #'sigmoid'
		print('Building SVC with {} kernel...\n',kernel )
		svm = SVC(cache_size=1000, kernel=kernel,class_weight='balanced').fit(X_train, y_train)
	#Plot report CM+ROC and learning curves 	
	score_svm = svm.score(X_test, y_test)
	plot_CM_classificationreport_and_learningcurve(svm, 'SVM', X_train, X_test, y_train, y_test)
	#Check if the model generalizes well.
	# print('Calling to cross_validation_formodel to see whether the model generalizes well...\n')
	# modelCV = SVC()
	# n_splits = 7
	# results_CV = cross_validation_formodel(modelCV, X_train, y_train, n_splits=n_splits)

	# print('Plotting learning curve for CV...\n')
	# kfolds = StratifiedKFold(n_splits=7)
	# #Generate indices to split data into training and test set.
	# cv = kfolds.split(X_all,y_all)
	# title ='SVM CV Learning Curve'
	# plot_learning_curve(svm, title, X_all, y_all, cv=cv, n_jobs=1)
	pdb.set_trace()
	n_splits = 7
	print("Exhaustive search for SVM parameters to improve the out-of-the-box performance of vanilla SVM.")
	if kernel is 'linear':
		parameters = {'C':10. ** np.arange(-2,3)}
	else:
		parameters = {'C':10. ** np.arange(-2,3), 'gamma':2. ** np.arange(-5, 1)}
	grid_SVM = GridSearchCV(svm, parameters, cv=n_splits, verbose=3, n_jobs=2).fit(X_train, y_train)
	print(grid_SVM.best_estimator_)
	if kernel is 'linear':
		print("\n LVSM GridSearchCV. The best C:{} \n".format(grid_SVM.best_params_.values()[0])) 
	else:
		print("\n LVSM GridSearchCV. The best C:{}, gamma:{} \n".format(grid_SVM.best_params_.values()[0], grid_SVM.best_params_.values()[1])) 
	
	print(" SVM accuracy score with default:{} and optimal hyperparameters C, gamma):{} \n", score_svm, grid_SVM.score(X_test, y_test))
	#Plot again 
	plot_CM_classificationreport_and_learningcurve(grid_SVM, 'SVM', X_train, X_test, y_train, y_test)
	pdb.set_trace()
	# load the m odel with : my_model_loaded = joblib.load("my_model.pkl")
	return svm

def plot_CM_classificationreport_and_learningcurve(clf, clf_name, X_train, X_test, y_train, y_test):
	""" plot_CM_classificationreport_and_learningcurve: plot two figures: CM + ROC and learning curve for CV"""
	y_train_pred = clf.predict(X_train)
	y_test_pred = clf.predict(X_test)
	y_pred = [int(a) for a in clf.predict(X_test)]
	num_correct = sum(int(a == ye) for a, ye in zip(y_pred, y_test))
	print("Baseline classifier using %s: %s of %s values correct." % (clf_name, num_correct, len(y_test)))
	#confusion matrix on the test data
	conf_matrix = confusion_matrix(y_test, y_pred)
	print(conf_matrix)
	print('Accuracy of {:s} classifier on training set {:.2f}'.format(clf_name, clf.score(X_train, y_train)))
	score_svm = clf.score(X_test, y_test)
	print('Accuracy of {:s} classifier on test set {:.2f}'.format(clf_name, score_svm))
	print(classification_report(y_test, y_pred))
	
	# plot learning curves
	title = str(clf_name) + ' Learning Curve'
	cv = 7
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	plot_learning_curve(clf, title, X_all, y_all, cv= cv, n_jobs=1)
	
	#plot confusion matrix and AUC
	fig,ax = plt.subplots(1,3)
	fig.set_size_inches(15,5)
	plot_cm(ax[0], y_train, y_train_pred, [0,1], str(clf_name) + ' Confusion matrix (TRAIN)', 0.5)
	plot_cm(ax[1], y_test, y_test_pred, [0,1], str(clf_name) + ' Confusion matrix (TEST)', 0.5)
	plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, 0.5)
	plt.tight_layout()
	plt.show()


def run_naive_Bayes(X_train, y_train, X_test, y_test):
	""" run_naive_Bayes
	Args: X_train, y_train, X_test, y_test
	output:estimator"""
	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	#kfolds = StratifiedKFold(modsel)
	#cv = kfolds.split(X_all,y_all)
	#cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
	#Incremental fit on a batch of samples.
	nbayes = GaussianNB().partial_fit(X_train,y_train, np.unique(y_train))
	nbayes = GaussianNB().fit(X_train,y_train)
	y_train_pred = nbayes.predict_proba(X_train)[:,1]
	#Return probability estimates for the test vector X.
	y_test_pred = nbayes.predict_proba(X_test)[:,1]
	y_pred = [int(a) for a in nbayes.predict(X_test)]
	num_correct = sum(int(a == ye) for a, ye in zip(y_pred, y_test))
	print("Baseline classifier using Naive Bayes: %s of %s values correct." % (num_correct, len(y_test)))
	# plot learning curve
	#http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
	plot_learning_curve(nbayes, 'Naive Bayes classifier learning curve', X_all, y_all,  cv=3, n_jobs=4)	
	print('Accuracy of GaussianNB classifier on training set {:.2f}'.format(nbayes.score(X_train, y_train)))
	print('Accuracy of GaussianNB classifier on test set {:.2f}'.format(nbayes.score(X_test, y_test)))
	print(classification_report(y_test, y_pred))
	#plot_class_regions_for_classifier(nbayes,X_train, y_train, X_test, y_test, 'Gaussian naive classifier')
	
	#plot confusion matrix and AUC
	fig,ax = plt.subplots(1,3)
	fig.set_size_inches(15,5)
	plot_cm(ax[0],  y_train, y_train_pred, [0,1], 'GaussianNB Confusion matrix (TRAIN)', 0.5)
	plot_cm(ax[1],  y_test, y_test_pred,   [0,1], 'GaussianNB Confusion matrix (TEST)', 0.5)
	plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, 0.5)
	plt.tight_layout()
	plt.show()
	return nbayes


def run_votingclassifier(classifiers, X_train, y_train, X_test, y_test):
	""" run_votingclassifier ensemble methods
	Args: dictionary of classifiers objects and names"""
	from sklearn.ensemble import VotingClassifier
	import sys
	sys.setrecursionlimit(10000)
	it_clf = []
	for key in classifiers.keys():
		it_clf_name = key
		it_clf_obj = classifiers[key]
		it_clf.append(tuple((it_clf_name, it_clf_obj)))
	voting_clf = VotingClassifier(estimators=it_clf, voting='hard')
	voting_clf.fit(X_train, y_train)
	
	#print(voting_clf.__class__.__name__, accuracy_score(y_test, y_pred))

	#it_clf.append(tuple(('VotingClassifier', voting_clf)))

	for clf in it_clf:
		#print('Fitting for classifier:{}', clf[0])
		clf[1].fit(X_train, y_train)
		#print('Predicting for classifier:{}', clf[0])
		y_pred = clf[1].predict(X_test)
		print(clf[1].__class__.__name__, accuracy_score(y_test, y_pred))
	y_pred = voting_clf.predict(X_test)
	print(voting_clf.__class__.__name__, accuracy_score(y_test, y_pred))
	print(classification_report(y_test, y_pred))
	#YS: plot results 
	#http://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_probas.html
	return voting_clf

def run_randomforest(X_train, y_train, X_test, y_test, X_features, threshold=None):
	""" run_randomforest: select most imp features
	Args:
	Output: rf object"""
	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	n_estimators = 500; max_features= 10; min_samples_leaf=28; max_leaf_nodes=16
	rf = RandomForestClassifier(n_estimators=n_estimators, max_features =max_features, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf, class_weight='balanced').fit(X_train,y_train)
	y_train_pred = rf.predict_proba(X_train)[:,1]
	y_test_pred = rf.predict_proba(X_test)[:,1]
	y_pred = [int(a) for a in rf.predict(X_test)]
	num_correct = sum(int(a == ye) for a, ye in zip(y_pred, y_test))
	print("Baseline classifier using RandomForestClassifier: n_estimators = %s, %s of %s values correct." % (n_estimators, num_correct, len(y_test)))
	# plot learning curve
	plot_learning_curve(rf, 'RandomForestClassifier learning curve', X_all, y_all, cv=3, n_jobs=1)	
	print('Accuracy of RandomForestClassifier classifier on training set {:.2f}'.format(rf.score(X_train, y_train)))
	print('Accuracy of RandomForestClassifier classifier on test set {:.2f}'.format(rf.score(X_test, y_test)))
	print(classification_report(y_test, y_pred))
	
	#plot confusion matrix and AUC
	fig,ax = plt.subplots(1,3)
	fig.set_size_inches(15,5)
	plot_cm(ax[0],  y_train, y_train_pred, [0,1], 'RandomForestClassifier Confusion matrix (TRAIN)', 0.5)
	plot_cm(ax[1],  y_test, y_test_pred,   [0,1], 'RandomForestClassifier Confusion matrix (TEST)', 0.5)
	plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, 0.5)
	plt.tight_layout()
	plt.show()
	# select most important features

	idxbools= rf.feature_importances_ > 0; idxbools = idxbools.tolist()
	idxbools = np.where(idxbools)[0]
	impfeatures = []; importances = []
	for i in idxbools:
		print('Feature:{}, importance={}'.format(X_features[i], rf.feature_importances_[i]))
		impfeatures.append(X_features[i])
		importances.append(rf.feature_importances_[i])
	plt.figure(figsize=(5,5))
	plot_feature_importances(rf,importances, impfeatures)
	# Finding the Optimal hyperparameters
	#
	print("Exhaustive search for RF parameters to improve the out-of-the-box performance of vanilla RF.")
	#n_estimators = 500; max_features= 10; min_samples_leaf=28; max_leaf_nodes=16
	parameters = {'n_estimators':10 ** np.arange(1,3), 'max_features':2 ** np.arange(1, 5), 'min_samples_leaf':[8,16,32]}
	#max_features (default=”auto”)If “auto”, then max_features=sqrt(n_features).
	parameters = {'n_estimators':10 ** np.arange(1,4), 'min_samples_leaf':[8,16,32]} 

	grid_rf = GridSearchCV(rf, parameters, cv=5, verbose=3, n_jobs=-1).fit(X_train, y_train)
	y_train_pred = grid_rf.predict_proba(X_train)[:,1]
	y_test_pred = grid_rf.predict_proba(X_test)[:,1]
	y_pred = [int(a) for a in grid_rf.predict(X_test)]
	num_correct = sum(int(a == ye) for a, ye in zip(y_pred, y_test))
	print("Baseline classifier using Grid RandomForestClassifier with optimal hyperparameters: %s of %s values correct." % (num_correct, len(y_test)))
	# plot learning curve
	plot_learning_curve(grid_rf, 'RandomForestClassifier learning curve', X_all, y_all, n_jobs=1)	
	print('Accuracy of Grid-RandomForestClassifier classifier on training set {:.2f}'.format(grid_rf.score(X_train, y_train)))
	print('Accuracy of Grid-RandomForestClassifier classifier on test set {:.2f}'.format(grid_rf.score(X_test, y_test)))
	print(classification_report(y_test, y_pred))
	#plot confusion matrix and AUC
	fig,ax = plt.subplots(1,3)
	fig.set_size_inches(15,5)
	plot_cm(ax[0],  y_train, y_train_pred, [0,1], 'Grid-RandomForestClassifier Confusion matrix (TRAIN)', 0.5)
	plot_cm(ax[1],  y_test, y_test_pred,   [0,1], 'Grid-RandomForestClassifier Confusion matrix (TEST)', 0.5)
	plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, 0.5)
	plt.tight_layout()
	plt.show()

	return rf, grid_rf

def run_logreg_Lasso(X_train, y_train, X_test, y_test,cv=None):
	"""run_logreg_Lasso: Lasso normal linear regression with L1 regularization (minimize the number of features or predictors int he model)

	Args:(X_train,y_train,X_test,y_test, cv >0 (scores 0.1~0.2) for cv=0 horrible results
	Output: Lasso estimator(suboptimal method for binary classification
	Example:run_logreg_Lasso(X_train, y_train, X_test, y_test,10) """	
	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	# define the model and the hyperparameter alpha (controls the stricteness of the regularization)
	alphas = np.logspace(-6, -0.5, 10)
	# GridSearchCV does exhaustive search over specified parameter values for an estimator
	# fit of an estimator on a parameter grid and chooses the parameters to maximize the cross-validation score	
	# take train set and learn a group of Lasso models by varying the value of the alpha hyperparameter.
	#Best possible score is 1.0, lower values are worse. Unlike most other scores
	#The score method of a LassoCV instance returns the R-Squared score, which can be negative, means performing poorly
	#Estimator score method is a default evaluation criterion for the problem they are designed to solve
	#By default, the GridSearchCV uses a 3-fold cross-validation
	lasso = Lasso(random_state=0).fit(X_train, y_train)
	#GridSearchCV implements a “fit” and a “score” method
	lasso_cv = GridSearchCV(lasso, dict(alpha=alphas), cv=cv).fit(X_train, y_train)
	if cv > 0:
		lasso = lasso_cv
	#lasso is a linear estimator doesnt have .predict_proba method only predict
	y_train_pred = lasso.predict(X_train)
	y_test_pred = lasso.predict(X_test)
	#binarize 0,1 the predictions
	y_pred = [int(a) for a in lasso.predict(X_test)]
	num_correct = sum(int(a == ye) for a, ye in zip(y_pred, y_test))
	print("Baseline classifier cv=%s using LogReg_Lasso: %s of %s values correct." % (cv, num_correct, len(y_test)))
	if cv > 0:
		print("Lasso estimator cv={} results:{}".format(cv, sorted(lasso_cv.cv_results_.keys())))
		print("Mean cross-validated cv={} score of the best estimator:{}".format(cv, lasso_cv.best_score_))
		print("Estimator cv={} was chosen by the search(highest score):{} ".format(cv, lasso_cv.best_estimator_))
	# plot learning curve
	plot_learning_curve(lasso, 'lasso', X_all, y_all, n_jobs=1)
	print('A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.')
	print('Coefficient of determination R^2 of the prediction of LogReg_Lasso on training set {:.2f}'.format(lasso.score(X_train, y_train)))
	print('Coefficient of determination R^2 of the prediction of LogReg_Lasso on test set {:.2f}'.format(lasso.score(X_test, y_test)))
	#plot confusion matrix and AUC
	fig,ax = plt.subplots(1,3)
	fig.set_size_inches(15,5)
	plot_cm(ax[0],  y_train, y_train_pred, [0,1], 'LogReg_Lasso Confusion matrix (TRAIN)', 0.5)
	plot_cm(ax[1],  y_test, y_test_pred,   [0,1], 'LogReg_Lasso Confusion matrix (TEST)', 0.5)
	plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, 0.5)
	plt.tight_layout()
	plt.show()
	return lasso

def run_logistic_regression(X_train, y_train, X_test, y_test, threshold=None, explanatory_variables=None, ):
	"""run_logistic_regression: logistic regression classifier
	Args:  X_train, y_train, X_test, y_test, threshold=[0,1] for predict_proba
	Output: logreg estimator"""
	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	#print logistic regression summary using statsmodels
	logit_model_sm = sm.Logit(y_all,X_all)
	result_sm = logit_model_sm.fit()
	print(result_sm.summary(xname=explanatory_variables))
	#Check the The p-values if for most of the variables are smaller than 0.05, most of them are significant to the model.

	if threshold is None:
		threshold = 0.5
	# Create logistic regression object, class_weight='balanced':replicating the smaller class until you have 
	#as many samples as in the larger one, but in an implicit way.
	#penalty default is l2, C : default: 1.0
	#Explicit weight , more emphasis with more weight on the less populated class e.g class_weight={0:.8, 1:1}
	parameters = {'C':10.0 ** -np.arange(-5, 5)}
	#parameters = {'C':1.0 ** -np.arange(0, 1)}
	#logreg = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
	logbalanced = LogisticRegression(class_weight='balanced')
	logreg = GridSearchCV(logbalanced, parameters, cv=5, verbose=3, n_jobs=2).fit(X_train, y_train)
	C_optimal = logreg.best_params_
	print('The optimal C (inverse of strength of regularization is:{}', C_optimal)

	#logreg = LogisticRegression(class_weight={0:.2, 1:.8}).fit(X_train, y_train)
	y_train_pred = logreg.predict_proba(X_train)[:,1]
	y_test_pred = logreg.predict_proba(X_test)[:,1]
	#print("Y test predict proba:{}".format(y_test_pred))
	y_pred = [int(a) for a in logreg.predict(X_test)]
	num_correct = sum(int(a == ye) for a, ye in zip(y_pred, y_test))
	print("Baseline classifier using LogReg: %s of %s values correct." % (num_correct, len(y_test)))
	#confusion matrix on the test data
	conf_matrix = confusion_matrix(y_test, y_pred)
	print(conf_matrix)
	print('Accuracy of LogReg classifier on training set {:.2f}'.format(logreg.score(X_train, y_train)))
	print('Accuracy of LogReg classifier on test set {:.2f}'.format(logreg.score(X_test, y_test)))
	print(classification_report(y_test, y_pred))

	# plot learning curve with cv = number of KFOlds 1 leave out
	plot_learning_curve(logreg, 'LogReg', X_all, y_all, cv= 7, n_jobs=1)	
	#plot confusion matrix and AUC
	fig,ax = plt.subplots(1,3)
	fig.set_size_inches(15,5)
	plot_cm(ax[0],  y_train, y_train_pred, [0,1], 'LogisticRegression Confusion matrix (TRAIN)', threshold)
	plot_cm(ax[1],  y_test, y_test_pred,   [0,1], 'LogisticRegression Confusion matrix (TEST)', threshold)
	plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, threshold)
	plt.tight_layout()
	plt.show()
	pdb.set_trace()
	
	#Check if the model generalizes well.
	print('Calling to cross_validation_formodel to see whether the model generalizes well...\n')
	modelCV = LogisticRegression(class_weight='balanced', C=C_optimal.values()[0])
	results_CV = cross_validation_formodel(modelCV, X_train, y_train, n_splits=7)
	print('CV score is:', results_CV)
	
	return logreg

def run_multi_layer_perceptron(X_train, y_train, X_test, y_test):
	""" run_multi_layer_perceptron
	Args:
	Output: """
	#create MLP with 1 hidden layer ans 1== logistic regression,10 and 100 units 
	# solver is the algorithm to learn the weights of the network
	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	#fig, subaxes = plt.subplots(1,3, figsize=(5,15))	
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	# class_weight is not implemented in MLP, call it with balanced datasets
	threshold = 0.5
	list_of_mlps = []
	for units in [1,10,100]:
		#slphs is regulariztion by default 0.001, solver = 'lbfgs' ‘sgd’ adam 
		mlp = MLPClassifier(hidden_layer_sizes = [units], alpha = 0.01, solver = 'sgd', random_state = 0).fit(X_train, y_train)
		#list_of_mlps.append(mlp)
		y_train_pred = mlp.predict_proba(X_train)[:,1]
		y_test_pred = mlp.predict_proba(X_test)[:,1]
		y_pred = [int(a) for a in mlp.predict(X_test)]
		num_correct = sum(int(a == ye) for a, ye in zip(y_pred, y_test))
		print("Baseline classifier using mlp: %s of %s values correct." % (num_correct, len(y_test)))
		#confusion matrix on the test data
		conf_matrix = confusion_matrix(y_test, y_pred)
		print(conf_matrix)

		title = 'MLP 1 layer: {} units'.format(units)
		print('Accuracy MLP 1 hidden layer units = {}, score= {:.2f} on training'.format(units, mlp.score(X_train, y_train)))
		print('Accuracy MLP 1 hidden layer units = {}, score= {:.2f} on test'.format(units, mlp.score(X_test, y_test)))
		print(classification_report(y_test, y_pred))
		plot_learning_curve(mlp, title, X_all, y_all, cv= 7, n_jobs=1)
		#plot confusion matrix and AUC
		fig,ax = plt.subplots(1,3)
		fig.set_size_inches(15,5)
		plot_cm(ax[0],  y_train, y_train_pred, [0,1], title +' Confusion matrix (TRAIN)', threshold)
		plot_cm(ax[1],  y_test, y_test_pred,   [0,1], title + ' Confusion matrix (TEST)', threshold)
		plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, threshold)
		plt.tight_layout()
		plt.show()	

	#fig, subaxes = plt.subplots(1,3, figsize=(5,15))	
	#plot_class_regions_for_classifier_subplot(mlp, X_train, y_train, X_test, y_test, title, axis)
	#plt.tight_layout()
	
	print('\n **** Building N layers M hidden units {} **** \n ')
	#create MLP with n hidden layers with m units each
	unitsperlayer = [16, 16, 12]
	title = 'MLP NxM: {} '.format(unitsperlayer)

	mlp_layers = MLPClassifier(hidden_layer_sizes = unitsperlayer, alpha = 0.1, solver = 'lbfgs', random_state = 0).fit(X_train, y_train)
	#list_of_mlps.append(mlp_layers)
	print("Exhaustive search for regularization parameter alpha.")
	#http://scikit-learn.org/stable/modules/neural_networks_supervised.html
	parameters = {'alpha':10.0 ** -np.arange(1, 3)}
	#crashes!!
	grid = GridSearchCV(mlp_layers, parameters, cv=5, verbose=3, n_jobs=-1).fit(X_train, y_train)
	print(grid.best_estimator_)
	print("MLPClassifier GridSearchCV. The best alpha is:{}".format(grid.best_params_)) 
	print("MLPClassifier of the given test data and labels={} ", grid.score(X_test, y_test))

	print('Accuracy MLP hidden layer size = {} on training set {:.2f}'.format(unitsperlayer, grid.score(X_train, y_train)))
	print('Accuracy MLP hidden layer size = {} on test set {:.2f}'.format(unitsperlayer, grid.score(X_test, y_test)))
	y_train_pred = grid.predict_proba(X_train)[:,1]
	y_test_pred = grid.predict_proba(X_test)[:,1]
	num_correct = sum(int(a == ye) for a, ye in zip(y_pred, y_test))
	print("Baseline classifier using mlp NxM: %s of %s values correct." % (num_correct, len(y_test)))
	conf_matrix = confusion_matrix(y_test, y_pred)
	print(conf_matrix)
	print(classification_report(y_test, y_pred))

	# plot learning curve
	plot_learning_curve(grid, title, X_all, y_all, cv= 7, n_jobs=1)
	
	#plot confusion matrix and AUC
	fig,ax = plt.subplots(1,3)
	fig.set_size_inches(15,5)
	plot_cm(ax[0],  y_train, y_train_pred, [0,1], title +' Confusion matrix (TRAIN)', threshold)
	plot_cm(ax[1],  y_test, y_test_pred,   [0,1], title + ' Confusion matrix (TEST)', threshold)
	plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred, threshold)
	plt.tight_layout()
	plt.show()	
	
	print("Compute PSI(population stability index or relative entropy \n")
	psi = psi_relative_entropy(y_test_pred, y_train_pred, 10)
	print("Compute kolmogorov test\n")
	ks_two_samples_goodness_of_fit(y_test_pred,y_train_pred[:len(y_test_pred)])
	return mlp_layers
	#return list_of_mlps  #return list of MLPs

def plot_scatter_target_cond(df, preffix_longit_xandy, target_variable=None):	
	"""scatter_plot_pair_variables_target: scatter dataframe features and coloured based on target variable values
	Args:df: Pandas dataframe, preffix_longitx:preffix of longitudinal variable to plot in X axiseg dx_visita, 
	preffix_longit_y:preffix of longitudinal variable to plot in Y axis eg audi_visita, target_variable: target feature contained in dataframe 
	Example: scatter_plot_target_cond(df, 'conversion')"""
	# scatter_plot_target_cond: average and standard deviation of longitudinal status
	#selcols(prefix,year ini,year end), if we dont have longitudinal set to 1,1 if we do 1,5
	df['longit_x_avg'] = df[selcols(preffix_longit_xandy[0],1,5)].mean(axis=1)
	df['longit_y_avg'] = df[selcols(preffix_longit_xandy[1],1,5)].mean(axis=1)
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

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

def build_model_with_keras(X_train, metrics):
	""" build_model_with_keras. function called in run_keras_deep_learning
	Args:
	Outoput:compiled keras model""" 

	activation = 'relu'; optimizer = 'adam'; loss = 'binary_crossentropy'; metrics= metrics; #
	model = Sequential() 
	model.add(Dense(16, activation=activation,  kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.001), input_shape=(X_train.shape[1],)))
	model.add(Dropout(0.5))
	model.add(Dense(16,  kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.001), activation=activation))
	model.add(Dropout(0.5))
	model.add(Dense(1,  kernel_initializer='uniform', activation='sigmoid'))
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	print ("Model created! activation:{}, optimizer:{}, loss:{}, metrics:{}".format(activation,optimizer,loss,metrics))
	return model
def plot_keras_metrics(history, msgtitle):
	""" plot_keras_metrics
	Args:history, 
	Output:
	"""
	#plot training and validation loss/accuracy
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	recall = history.history['recall']
	val_recall = history.history['val_recall']
	precision = history.history['precision']
	val_precision = history.history['val_precision']

	plt.figure(figsize=(15,5))
	#msgtitle ='Learning metrics Deep Network (no k-fold)'
	plt.suptitle(msgtitle)
	plt.subplot(2, 2, 2)
	epochs = range(1, len(loss) + 1)
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	#plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()

	plt.subplot(2, 2, 1)
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	#plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()

	plt.subplot(2, 2, 3)
	plt.plot(epochs, recall, 'bo', label='Training recall')
	plt.plot(epochs, val_recall, 'b', label='Validation recall')
	plt.title('Training and validation recall')
	plt.xlabel('Epochs')
	plt.ylabel('recall')
	plt.legend()
	plt.show()

	plt.subplot(2, 2, 4)
	plt.plot(epochs, precision, 'bo', label='Training precision')
	plt.plot(epochs, val_precision, 'b', label='Validation precision')
	plt.title('Training and validation precision')
	plt.xlabel('Epochs')
	plt.ylabel('precision')
	plt.legend()
	plt.show()

def run_keras_deep_learning(X_train, y_train, X_test, y_test):
	""" run_keras_deep_learning: deep network classifier using keras
	Args: X_train, y_train, X_test, y_test
	Output:
	Remember to activate the virtual environment source ~/git...code/tensorflow/bin/activate"""
 	from keras import callbacks
	import tensorflow as tf
	precision = as_keras_metric(tf.metrics.precision)
	recall = as_keras_metric(tf.metrics.recall)
	#https://stackoverflow.com/questions/43076609/how-to-calculate-precision-and-recall-in-keras

	input_samples = X_train.shape[0]
	input_dim = X_train.shape[1]
	# set apart the 10 % of the training set for validation
	x_val = X_train[:int(input_samples*0.1)]
	partial_x_train = X_train[int(input_samples*0.1):]
	y_val = y_train[:int(input_samples*0.1)]
	partial_y_train = y_train[int(input_samples*0.1):]
	# Deep network model run_keras_deep_learning
	metrics=['accuracy', precision, recall]
	model = build_model_with_keras(X_train, metrics)
	# Evaluating model (no K-fold xvalidation) 
	#bl = BatchLogger() oibluy for GPU, training way too long in cpu, only for .fit with validation_split
	callbacks = [callbacks.TensorBoard(log_dir='logs_tensorboard',histogram_freq=1, embeddings_freq=1,)]  
	num_epochs = 20
	# the more batch size more meory used
	batch_size = 48 
	number_of_iterations = input_samples/batch_size 
	print('Number of iterations:{}', number_of_iterations, ' for {} epochs:',num_epochs, ' .Total: {}', num_epochs*number_of_iterations )
	#history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=16, \
	#	verbose=2, validation_split=0.2, callbacks=callbacks)
	class_weight = {0:1., 1:7.} #to treat every instance of class 1 as 10 instances of class 0
	history = model.fit(partial_x_train, partial_y_train, epochs=num_epochs, batch_size=batch_size, \
		verbose=2, validation_data=(x_val, y_val), class_weight = class_weight) #'auto'
	evaluationmodel = model.evaluate(X_test, y_test)
	model_loss = evaluationmodel[0]; model_acc = evaluationmodel[1]; model_prec = evaluationmodel[2]; model_recall = evaluationmodel[3]
	metricsused = history.history.keys()
	print('Metrics used:{}',metricsused )
	#['acc', 'loss', 'recall', 'precision', 'val_acc', 'val_recall', 'val_precision', 'val_loss']
	predictions = model.predict(X_test)
	converters = np.sum(predictions >0.5)
	print('Number of converters : {} / {} ', np.sum(converters), predictions.shape[0])
	print("Deep Network Loss={}, accuracy={}, precision={} ".format(model_loss, model_acc, model_prec))
	#compare accuraccy with  a purely random classifier 
	y_test_copy = np.copy(y_test)
	np.random.shuffle(y_test_copy)
	hits_array = np.array(y_test) == np.array(y_test_copy)
	chance_acc = float(np.sum(hits_array)) / len(y_test)
	print("\nModel accuracy ={:.3f} Dummy model accuracy={:.3f}".format(model_acc,chance_acc))
	#print("Model predictions ={}".format(model.predict(X_test).ravel()))
	plot_model(model, show_shapes=True, to_file='dn_model_nocv.png')
	plot_keras_metrics(history, msgtitle ='Learning metrics Deep Network (no k-fold)')
	#K-fold
	k = 4
	print('Running Network lerarning for  K={}-fold cross-validation', k)
	num_val_samples = X_train.shape[0] // k
	all_scores = []; acc_history = []; loss_history= []; precision_history = []; recall_history = []
	for i in range(k):
		print("Processing fold #",i)
		val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]
		val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
		partial_train_data = np.concatenate([X_train[:i * num_val_samples], X_train[(i + 1) * num_val_samples:]],axis=0)
		partial_train_targets = np.concatenate([y_train[:i * num_val_samples], y_train[(i + 1) * num_val_samples:]],axis=0)  
		#build model, returns a compiled model
		model = build_model_with_keras(X_train, metrics)
		plot_model(model, show_shapes=True, to_file='dn_model_cv.png')
		#evaluate the network (trains the model)
		history = model.fit(partial_train_data,partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=1, verbose=1,class_weight = class_weight)
 		#evaluates model in validation data
		evaluationFolds = model.evaluate(val_data, val_targets, verbose=0)
		model_loss = evaluationFolds[0]; model_acc = evaluationFolds[1]; model_prec = evaluationFolds[2]; model_recall = evaluationFolds[3]
		all_scores.append(evaluationFolds)
		#keep a record of how well the model does at each epoch, save the per-epoch validation score log.
		history.history.keys() # metrics being monitored during validation [u'acc', u'loss', u'val_acc', u'val_loss']
		acc_history.append(history.history['val_acc'])
		loss_history.append(history.history['val_loss'])
		precision_history.append(history.history['val_precision'])
		recall_history.append(history.history['val_recall'])
	
	print("all_scores validation, mean={} std=({})".format(np.mean(all_scores),np.std(all_scores)))	
	average_accuracy_history = [np.mean([x[i] for x in acc_history]) for i in range(num_epochs)]
	average_loss_history = [np.mean([x[i] for x in loss_history]) for i in range(num_epochs)]
	average_precision_history = [np.mean([x[i] for x in precision_history]) for i in range(num_epochs)]
	average_recall_history = [np.mean([x[i] for x in recall_history]) for i in range(num_epochs)]
	#plot k-fold cross validation loss/accuracy
	plt.figure(figsize=(15,5))
	msgtitle = str(k) + '-fold cross validation Deep Network'
	plt.suptitle(msgtitle)
	plt.subplot(2, 2, 1)
	plt.plot(range(1, len(average_accuracy_history) + 1), average_accuracy_history)
	plt.ylabel('Validation Acc')
	plt.subplot(2, 2, 2)
	plt.plot(range(1, len(average_loss_history) + 1), average_loss_history)
	plt.ylabel('Validation Loss')
	plt.subplot(2, 2, 3)
	plt.plot(range(1, len(average_precision_history) + 1), average_precision_history)
	plt.xlabel('Epochs')
	plt.ylabel('Validation Precision')
	plt.subplot(2, 2, 4)
	plt.plot(range(1, len(average_recall_history) + 1), average_recall_history)
	plt.xlabel('Epochs')
	plt.ylabel('Validation Recall')
	plt.show()
	#test_score = history.evaluate(X_test, y_test)
	return model

def _joint_probabilities_constant_sigma(D, sigma):
	P = np.exp(-D**2/2 * sigma**2)
	P /= np.sum(P, axis=1)
	return P

def run_tSNE_manifold_learning(X_train, y_train, X_test, y_test):
	""" run_tSNE_manifold_learning: manifold learning to visualize high dimensional data space
	Args:X_train, y_train, X_test, y_test
	Outputs: """
	# Pairwise distances between all data points
	import sklearn
	import matplotlib.patches as mpatches
	print("Training set set dimensions: X=", X_train.shape, " y=", y_train.shape)
	print("Test set dimensions: X_test=", X_test.shape, " y_test=", y_test.shape)
	X_all = np.concatenate((X_train, X_test), axis=0)
	y_all = np.concatenate((y_train, y_test), axis=0)
	D = pairwise_distances(X_all, squared=True)
	# Similarity with constant sigma.
	sigma = .002
	P_constant = _joint_probabilities_constant_sigma(D, sigma)
	# Similarity with variable sigma.
	P_binary = _joint_probabilities(D, 30., False)
	# The output of this function needs to be reshaped to a square matrix.
	P_binary_s = squareform(P_binary)
	#display the distance matrix of the data points D, and the similarity matrix
	#with both a constant P_constant and variable sigma P_constant_s.
	plt.figure(figsize=(12, 4))
	pal = sns.light_palette("blue", as_cmap=True)
	plt.subplot(131)
	#plot only every 10th element
	#plt.imshow(D[::10, ::10], interpolation='none', cmap=pal)
	plt.imshow(D, interpolation='none', cmap=pal)
	plt.axis('on')
	plt.title("Distance matrix", fontdict={'fontsize': 14})
	plt.subplot(132)
	plt.imshow(P_constant, interpolation='none', cmap=pal)
	plt.axis('off')
	plt.title("$p_{j|i}$ (constant $\sigma$=" + str(sigma) +")", fontdict={'fontsize': 14})

	plt.subplot(133)
	plt.imshow(P_binary_s, interpolation='none', cmap=pal)
	plt.axis('off')
	plt.title("$p_{j|i}$ (variable $\sigma$)", fontdict={'fontsize': 14})
	plt.savefig('images/similarity-generated.png', dpi=120)
	#Calculate the map pint matrix Q using KL and gradient descent to minimize the score
	# This list will contain the positions of the map points at every iteration
	positions = []
	def _gradient_descent(objective, p0, it, n_iter, n_iter_check=1,n_iter_without_progress=30,
		momentum=0.5, learning_rate=1000.0, min_gain=0.01,
		min_grad_norm=1e-7, min_error_diff=1e-7, verbose=0,
		args=[],kwargs=None):
		# The documentation of this function can be found in scikit-learn's code.
		p = p0.copy().ravel()
		update = np.zeros_like(p)
		gains = np.ones_like(p)
		error = np.finfo(np.float).max
		best_error = np.finfo(np.float).max
		best_iter = 0

		for i in range(it, n_iter):
			# We save the current position.
			positions.append(p.copy())
			new_error, grad = objective(p, *args)
			error_diff = np.abs(new_error - error)
			error = new_error
			grad_norm = np.linalg.norm(grad)
			if error < best_error:
				best_error = error
				best_iter = i
			elif i - best_iter > n_iter_without_progress:
				break
			if min_grad_norm >= grad_norm:
				break
			if min_error_diff >= error_diff:
				break
			inc = update * grad >= 0.0
			dec = np.invert(inc)
			gains[inc] += 0.05
			gains[dec] *= 0.95
			np.clip(gains, min_gain, np.inf)
			grad *= gains
			update = momentum * update - learning_rate * grad
			p += update
		return p, error, i

	sklearn.manifold.t_sne._gradient_descent = _gradient_descent
	X_proj =TSNE(perplexity=5, random_state=0).fit_transform(X_all)
	plt.figure(figsize=(9,9))
	
	for i in range(X_proj.shape[0]):
		if y_all[i] < 1:
			plt.scatter(X_proj[i,0],X_proj[i,1], color='b', alpha=0.5, label='conversion: NO')
		else:
			plt.scatter(X_proj[i,0],X_proj[i,1], color='r', alpha=0.5, label='conversion: YES')
	plt.tight_layout()
	plt.title('tSNE map point')
	red_patch = mpatches.Patch(color='r', label='converters')
	blue_patch = mpatches.Patch(color='b', label='Non converters')
	plt.legend(handles=[red_patch, blue_patch])
	plt.axis('off')
	plt.show()
	X_iter = np.dstack(position.reshape(-1, 2) for position in positions)
	# create an animation using MoviePy.


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

def calculate_top_features_contributing_class(clf, features, numbertop=None):
	""" calculate_top_features_contributing_class: print the n features contributing the most to class labels for a fitted estimator.
	Args: estimator, features lits and number of features"""
	if numbertop is None:
		numbertop = 10
	print("The estimator is:{} \n".format(clf))
	for i in range(0, clf.best_estimator_.coef_.shape[0]):
		toplist = np.argsort(clf.best_estimator_.coef_[i])[-numbertop:]
	print("the top {} features indices contributing to the class labels are:{}".format(numbertop, toplist))
	if features is not None:
		print("\tand the top {} features labels contributing to the class labels are:{} \n".format(numbertop, operator.itemgetter(*toplist)(features)))
	
def run_load_csv(csv_path = None):
	""" load csv database, load csv file and print summary
	Args: csv_path
	Output: dataset pandas dataframe"""
	if csv_path is None:
		csv_path = "/Users/jaime/vallecas/data/scc/SCDPlus_IM_21052018.csv"
		csv_path = "/Users/jaime/vallecas/data/scc/socioeconomics_29052018.csv" #socioeconomics dataset Dataset_29052018
		csv_path = "/Users/jaime/vallecas/data/scc/Dataset_29052018.csv"
		csv_path = "/Users/jaime/vallecas/data/BBDD_vallecas/Proyecto Vallecas_5_visitas_mayo 2018.csv"
		csv_path = "/Users/jaime/vallecas/data/BBDD_vallecas/Proyecto Vallecas_N_visitas_14_junio_2018.csv"
	dataset = pd.read_csv(csv_path) #, sep=';')
	print('Loaded the csv file: ', csv_path, '\n')
	#summary of data
	print("Number of Rows=", dataset.shape[0])
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
	"""run_statistical_tests ANOVA test
	Args:dataset, feature_label, target_label
	Output: test results """
	#chi_square_of_df_cols(dataset, feature_label[0], feature_label[1])
	anova_result = anova_test(dataset, feature_label, target_label)
	ttest_result = t_test(dataset, feature_label, target_label)
	#chi_result  = chi_square(dataset, feature_label)
	dict_results = {'anova': anova_result, 'ttest': ttest_result} #, 'chi':chi_result}
	return dict_results

def plot_grid_pairs(df, features):
	""" plot_grid_pairs plot paurs of features with seaborn.pairplot 
	Args: dataframe, list of features
	"""
	#https://seaborn.pydata.org/tutorial/axis_grids.html
	feature_x = features[6]
	feature_y = features[0]
	feature_z = features[8] #conditional
 	#Draw conditional plots of segmented data using factorplot and FacetGrid
	sns.factorplot(data=df, x=feature_x, y=feature_y)
	sns.factorplot(data=df, x=feature_x, y=feature_y, col=feature_z)

	#kernel density estimate KDE plot
	g = sns.FacetGrid(df, col=feature_z)
	g.map(sns.distplot, feature_x) 
	#scatter plot
	g.map(plt.scatter, feature_x, feature_y)
	#draw regression plot
	g.map(plt.scatter,feature_x, feature_y)  
	#PairGrid to showing the interactions between variables
	g = sns.FacetGrid(df[[feature_x,feature_y]], col="tabac", size=4, aspect=.5)
	g.map(sns.barplot, feature_x, feature_y)
 
	g = sns.pairplot(df[[feature_x,feature_y]],diag_kind="hist")
	g.map_upper(sns.regplot) 
	g.map_lower(sns.residplot) 
	g.map_diag(plt.hist) 
	for ax in g.axes.flat:
		plt.setp(ax.get_xticklabels(), rotation=45) 
	g.add_legend() 
	g.set(alpha=0.5)
	#view both a joint distribution and its marginals at once
	sns.jointplot(feature_x, feature_y, data=df[features], kind='kde')
	g = sns.JointGrid(x=feature_x, y=feature_y, data=df[features]) 
	g.plot_joint(sns.regplot, order=2) 
	g.plot_marginals(sns.distplot)



def plot_histogram_pair_variables(dataset, feature_label=None):
	"""" histogram_plot_pair_variables: plot 2 histotgrams one for each variable
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

def build_connectedGraph_correlation_matrix(corr_df, threshold=None, corr_target=None):
	""" build_connectedGraph_correlation_matrix: requires package pip install pygraphviz. Plot only connected compoennts, omit isolated nodes
	Args:A is the dataframe correlation matrix
	Output:G
	"""
	import string
	# extract corr matrix fro the dataframe
	A_df = corr_df.as_matrix()
	node_names = corr_df.keys().tolist()
	if threshold is None:
		threshold = mean(corr_matrix)
	A = np.abs(A_df) > threshold
	# delete isolated nodes
	connected_nodes = []
	for nod in range(0,len(corr_target)):
		if sum(A[nod,:])>1:
			connected_nodes.append(nod)	
	row_idx = np.array([connected_nodes])
	A = A[np.ix_(row_idx.ravel(),row_idx.ravel())]
	fig, ax = plt.subplots()
	
	labels = {}
	for idx,val in enumerate(node_names):
		labels[idx] = val
		if labels[idx].endswith('_visita1'):
			labels[idx] = labels[idx][:len(labels[idx])-len('_visita1')]
		# plot label and correlation with the target 
		if corr_target is not None:
			labels[idx] = labels[idx]+ '\n' + `'{0:.2g}'.format(corr_target[idx])`

	G = nx.from_numpy_matrix(A)
	#G.remove_nodes_from(nx.isolates(G))
	pos=nx.spring_layout(G)
	nx.draw_networkx_nodes(G, pos)
	nx.draw_networkx_edges(G, pos)
	labels = dict((key,value) for key, value in labels.iteritems() if key in connected_nodes)
	labels_connected = {}; counter = 0
	for item in labels: 
		labels_connected[counter] = labels[item]
		counter+=1
	nx.draw_networkx_labels(G, pos, labels_connected, font_size=7)
	# plt.title('Binary Graph from correlation matrix{}'.format(node_names))
	plt.title('Binary Graph, threshold={0:.3g}'.format(threshold))
	return G
	pdb.set_trace()

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

def chi_square(df, col1):
	""" Calculate a one-way chi square test (scipy.stats.chisquare(f_obs, f_exp=None, ddof=0, axis=0)[source])
	This test is invalid when the observed or expected frequencies in each category are too small. 
	A typical rule is that all of the observed and expected frequencies should be at least 5.
	When just f_obs is given, it is assumed that the expected frequencies are uniform and given by the mean of the observed frequencies.
	Args:The chi square test tests the null hypothesis that the categorical data has the given frequencies.
	Output:"""
	from scipy.stats import chisquare
	res_chi =  chisquare([df[col1]])


def normal_gaussian_test(rv, rv_name =None, method=None):
	""" normal_gaussian_test: Test isf a dfistribution fllows a Normal Gaussian fistribution 
	Args: rv: random variable we need to know its distribution. 
	'shapiro' Shapiro-Wilk tests if a random sample came from a normal distribution (null hypothesis: data 
	is normally distributed). Use shapiro only if size < 5000
	'kolmogorov': Kolmogorov–Smirnov tests if a sample distribution fits the CDF of another distribution (eg Gaussian).
	As for shapiro, null hypothesis is the sample distribution is identical to the other distribution (Gaussian). 
	If p < .05 we can reject the null
	"""
	#It is completely possible that for p > 0.05 and the data does not come from a normal population. 
	#Failure to reject could be from the sample size being too small to detect the non-normality
	#With a large sample size of 1000 samples or more, a small deviation from normality (some noise in the sample) may be concluded as significant and reject the null.
	# The two best tools are the histogram and Q-Q plot. Potentially using the tests as additional indicators.
	import pylab
	p_threshold = 0.05
	[t_shap, p_shap] = stats.shapiro(rv)
	
	if p_shap < p_threshold:
		print('Shapiro-Wilk test: Reject null hypothesis that sample comes from Gaussian distribution \n')
	else:
		print('Shapiro-Wilk test:DO NOT Reject null hypothesis that sample comes from Gaussian distribution \n')
		print('Likely sample comes from Normal distribution. But the failure to reject could be because of the sample size:{}\n', rv.shape[0])
	print('Shapiro-Wilk test: t statistic:{} and p-value:{}',t_shap, p_shap)
	#quantile-quantile (QQ) plot
	#If the two distributions being compared are from a common distribution, the points in from QQ plot the points in 
	#the plot will approximately lie on a line, but not necessarily on the line y = x		
	sm.qqplot(rv, loc = rv.mean(), scale = rv.std(), line='s')
	plt.title('Shapiro-Wilk: '+ rv_name+ ' . Shapiro p-value='+ str(p_shap))
	pylab.show()
	#The Kolmogorov–Smirnov tests if a sample distribution fits a cumulative distribution function (CDF) of are referenced distribution
	[t_kol, p_kol] = stats.kstest(rv, 'norm', args=(rv.mean(), rv.std()))
	if p_kol < p_threshold:
		print('Kolmogorov–Smirnov: Reject null hypothesis that sample comes from Gaussian distribution \n')
	else:
		print('Kolmogorov–Smirnov: DO NOT Reject null hypothesis that sample comes from Gaussian distribution \n')
		print('Likely sample comes from Normal distribution. But the failure to reject could be because of the sample size:{}\n', rv.shape[0])
	#Comparing CDF for KS test
	print('Kolmogorov test: t statistic:{} and p-value:{}',t_kol, p_kol)
	length = len(rv)
	plt.figure(figsize=(12, 7))
	plt.plot(np.sort(rv), np.linspace(0, 1, len(rv), endpoint=False))
	plt.plot(np.sort(stats.norm.rvs(loc=rv.mean(), scale=rv.std(), size=len(rv))), np.linspace(0, 1, len(rv), endpoint=False))
	plt.legend('top right')
	plt.legend(['Data', 'Theoretical Gaussian Values'])
	plt.title('Comparing CDFs for KS-Test: '+ rv_name + ' . KS p-value='+str(p_kol))
		

def ks_two_samples_goodness_of_fit(sample1, sample2):
	"""ks_two_samples_goodness_of_fit: Perform a Kolmogorov-Smirnov two sample test that two data samples come from the same distribution. 
	Note that we are not specifying what that common distribution is.  
	Args: sample1, sample2 continuous distribution
	Output:
	"""
	print("Kolmogorov smirnoff test: H0 2 independent samples are drawn from the same continuous distribution")
	ks_stat, ks_pvalue = stats.ks_2samp(sample1, sample2)
	print("[KS statistic={:.3f}, p-value{:.3f}]".format(ks_stat,ks_pvalue))
	if ks_pvalue < 0.01:
		print("We reject H0: y_pred and y_test come from the same continuous distribution")
	else:
		print("We can't reject H0: y_pred and y_test come from the same continuous distribution")
		

def psi_relative_entropy(bench, comp, group):
	""" psi_relative_entropy: population stability index, metric to measure goodness of fit, is a measure of the stability
	of a variable iover time. It is a symmetric version of KL. PSI <0.1 is stable, 0.1-0.25 needs monitoring,
	 >0.25 stabilty of the variable is suspect. 
	 To create a PSI, we need to select a benchmark. Normally a build sample is treated as a benchmark. 
	 We create 10 or 20 bins from the benchmark. Then we compare the target against the benchmark
	 https://qizeresearch.wordpress.com/2013/11/20/population-stability-index-psi/
	 Args:bench list with the sample , comp list with the distrib target, group is the number of bins
	 Output:psi
	 Example: psi_relative_entropy([], [],  20)

	"""
	from math import floor, log
	ben_len = len(bench)
	comp  =comp[:ben_len]
	comp_len = len(comp) 
	bench.sort()
	comp.sort()
	psi_cut = []
	n=int(floor(ben_len/group))
	for i in range(1,group):
		lowercut=bench[(i-1)*n+1]
		if i!=group:
			uppercut=bench[(i*n)]
			ben_cnt=n
		else:
			uppercut=bench[-1]
			ben_cnt=ben_len-group*(n-1)
	comp_cnt = len([i for i in comp if i > lowercut and i<=uppercut])
	ben_pct=(ben_cnt+0.0)/ben_len
	comp_pct=(comp_cnt+0.0)/comp_len
	psi_cut.append((ben_pct-comp_pct)*log(ben_pct/comp_pct))
	psi=sum(psi_cut)
	print("PSI ={}".format(psi))
	return psi
	

def anova_test(df, feature=None, target_label=None):
	""" The one-way analysis of variance (ANOVA) is used to determine whether there are any statistically significant differences between 
	the means of three or more independent (unrelated) groups.
	ANOVA test for pandas data set and features.  one-way ANOVA is an omnibus test statistic and cannot tell you which specific groups 
	were statistically significantly different from each other, only that at least two groups were
	Example: anova_test(dataset, features = [dataset.keys()[1], dataset.keys()[2]]) 
	MSwithin = SSwithin/DFwithin"""
	#from scipy import stats
	#F, p = stats.f_oneway(dataset[features[0]], dataset[features[1]], dataset[features[1]])
	#print("ANOVA test groups", features, ". F =", F, " p =", p)
	import scipy.stats as ss
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

def t_test(dataset, feature1, feature2):
	""" t test: Calculate the T-test for the means of two independent samples of scores"""
	from scipy.stats import ttest_ind
	print('Calculating scipy.stats.ttest_ind for features:', feature1, ' and ', feature2, ' \n')
	print('It is a two-sided test for the null hypothesis that 2 independent samples have identical expected values. ') 
	#by default It assumes equal variance ttest_ind(dataset[feature1], dataset[feature2], equal_var=True)
	# For unequal variance is called Welch's test
	#Student's t-test assumes that the two populations have normal distributions and with equal variances. 
	#Welch's t-test is designed for unequal variances, but the assumption of normality is maintained.
	# Welch's t-test is an approximate solution to the Behrens–Fisher problem.
	tstat_student, pval_student = ttest_ind(dataset[feature1], dataset[feature2])
	#print('Computing Welchs ttest, equal variance is not assumed')
	#tstat_welch, pval_welch = ttest_ind(dataset[feature1], dataset[feature2], equal_var=False)


def print_feature_importances(learner, features):
	""" print_fearure_importances only for elarners that take feature_importances_
	XGB, random forest, decision tree (NO SVC, GridSearchCV, mlp_estimator, Bayes, Keras)"""
	features_imp = pd.DataFrame(learner.feature_importances_, index=features)
	features_imp  = features_imp.sort_values(by=0, ascending=False)
	print("The most important features are {}".format(features_imp.head(20)))

def detect_multicollinearities(df, target, cols=None):
	""" detect_multicollinearities 
	Args:
	Output: """
	f =plt.figure(figsize=(8, 4))
	sns.countplot(df[target], palette='RdBu')
	cols.append(target)
	df = df[cols].dropna(axis=0)
	# count number of obvs in each class

	nonconverters, converters = df[target].value_counts()
	print('Number of nonconverters: ', nonconverters)
	print('Number of converters : ', converters)

	print(" Generate a scatter plot matrix with the mean columns...\n")# 
	g = sns.pairplot(data=df.dropna(),hue = target, palette='RdBu')
	dir_images = '/Users/jaime/github/code/tensorflow/production/images'
	filenametosav = cols[0] + '_scatter.png'
	g.savefig(os.path.join(dir_images, filenametosav))

	#pdb.set_trace()
	print(" Generate and visualize the correlation matrix...\n")
	corr = df.corr().round(3)
	# Mask for the upper triangle
	mask = np.zeros_like(corr, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True
	f, ax = plt.subplots(figsize=(20, 20))
	# Define custom colormap
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	# Draw the heatmap
	sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
	plt.tight_layout()
	dir_images = '/Users/jaime/github/code/tensorflow/production/images'
	filenametosav = cols[0] + '_corr.png'
	f.savefig(os.path.join(dir_images, filenametosav))
	# plot joint distributions of each feature against the target
	run_jointdibutions= True
	if run_jointdibutions is True:
		print('Plotting joint distributions for: ', cols, '\n')
		feature_y = target
		for i in cols[:-1]:
			feature_x = i
			dfjoints = df[[feature_x,feature_y]].dropna()
			plot_jointdistributions(dfjoints, feature_x, feature_y)
			print('\n     Closing all figures.   ....\n')

def plot_jointdistributions(dfjoints, feature_x, feature_y, feature_x2=None):
	""" plot_jointdistributions
	Args: dfjoints (pandas dataframe), feature_x, feature_y, feature_x2 (for scatter only)
	Output:"""
	#dfjoints = dataframe[['scd_visita1', 'conversion']].dropna()
	#dfjoints = dataframe[[dict_features['diet_keto'][3], 'conversion']].dropna() 
	#https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html
	dir_images = '/Users/jaime/github/code/tensorflow/production/images'
	print("Plot Kernel Density Estimation (KDE) of ", feature_x, " and ", feature_y)
	
	sns_plot =sns.jointplot(dfjoints[feature_x], dfjoints[feature_y],dfjoints, kind='kde')
	sns_plot.savefig(os.path.join(dir_images, 'joint_' + feature_x + feature_y + '.png'))
	print("Combine Histograms and KDE can be combined using distplot for: ", feature_x, " and ", feature_y)
	f = plt.figure(figsize=(6,6))
	ax =sns.distplot(dfjoints[feature_x])
	ax =sns.distplot(dfjoints[feature_y]);
	f.savefig(os.path.join(dir_images, 'distplot_' + feature_x + feature_y + '.png'))
	print("Joint distribution and the marginal distributions together for: ", feature_x, " and ", feature_y)

	with sns.axes_style('white'):
		f = sns.jointplot(feature_x, feature_y, dfjoints, kind='reg') #kind='hex' 'kde'
		f.savefig(os.path.join(dir_images, 'jointplot_' + feature_x + feature_y + '.png'))
	f = plt.figure(figsize=(6,6))
	sns.kdeplot(dfjoints[feature_x][dfjoints[feature_y]==0], label='NonConv', shade=True)
	sns.kdeplot(dfjoints[feature_x][dfjoints[feature_y]==1], label='Conv', shade=True)
	f.savefig(os.path.join(dir_images, 'kdeplot_' + feature_x + feature_y + '.png'))
	plt.xlabel(feature_x);	
	f = plt.figure(figsize=(6,6))
	sns_plot = sns.violinplot(feature_y, feature_x, data=dfjoints,
               palette=["lightblue", "lightpink"]);	
	f.savefig(os.path.join(dir_images, 'violinplot_' + feature_x + feature_y + '.png'))
	#scatter plot feature_x1, featue_x2 regplot, which will automatically fit a linear regression to the data:
	if feature_x2 is not None: 
		g = sns.lmplot(feature_x, feature_x2, col=feature_y, data=dfjoints, markers=".", scatter_kws=dict(color='c'))
		g.map(plt.axhline, y=0.1, color="k", ls=":");
		g.set_axis_labels(feature_x,feature_x2)
		g.savefig(os.path.join(dir_images, 'lmplot_' + feature_x + feature_y + '.png'))
	
def recursive_feature_elimination(X, y, nbofbestfeatures, explanatory_features):
	""" recursive_feature_elimination (RFE) using a logisticregression model, RFE tries to select features by
	recursively considering smaller and smaller sets of features. 
	RFE select features by recursively considering smaller and smaller sets of features
	Args:X,y
	Output:"""
	from sklearn.feature_selection import RFE
	logreg = LogisticRegression()
	#nbofbestfeatures = 20
	rfe = RFE(logreg, nbofbestfeatures)
	rfe = rfe.fit(X, y )
	print(rfe.support_)
	print(rfe.ranking_)
	best_logreg_features = []
	print('Creating the list of ', nbofbestfeatures,' most important features\n')
	for i in range(0, len(rfe.support_.tolist())):
		if rfe.support_.tolist()[i] == True:
			best_logreg_features.append(explanatory_features[i])
	return best_logreg_features

def cross_validation_formodel(modelCV, X_train, y_train, n_splits):
	""" cross_validation_formodel"""
	kfold = model_selection.KFold(n_splits=n_splits, random_state=7)
	scoring = 'accuracy'
	#Evaluate a score by cross-validation, results= scores : of scores of the estimator for each run of the cv
	#array of float, shape=(len(list(cv)),)
	results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
	print("%d-fold cross validation average accuracy: %.3f" % (n_splits, results.mean()))

	# skf = StratifiedKFold(n_splits=n_splits)
	# #skf.get_n_splits(X, y)
	# print(skf)  
	# for train_index, test_index in skf.split(X_train, y_train):
	# 	print("TRAIN:", train_index, "TEST:", test_index)
	# 	X_train, X_test = X_train[train_index-1], X_train[test_index-1]
	# 	y_train, y_test = y_train[train_index-1], y_train[test_index-1]
	# 	modelCV.fit(X_train, y_train)
	# 	confusion_matrix(y_test, modelCV.predict(X_test))

	return results


def meritxell_eda():
	""" """
	cvs2load = '/Users/jaime/vallecas/data/BBDD_vallecas/20180531_meritxell.csv'
	dataframe = run_load_csv(cvs2load)
	cleanup_column_names(dataframe, {}, True)	
	target_variable = '5dcl'
	explanatory_features = ['1glu',  'visita_1_sue_rec', 'visita_1_sue_noc','1ortostatismo', '1trat', '1dcl']
	explanatory_features = ['5glu',  'visita_5_sue_rec', 'visita_5_sue_noc', '5ortostatismo', '5trat', '5dcl']

	dataframe = dataframe[explanatory_features]
	dataframe.dropna(axis=0, inplace=True)
	#plot_histograma_bygroup_categorical(dataframe[explanatory_features], target_variable=target_variable)
	feature_x = explanatory_features[1]
	feature_y = target_variable
	plot_histogram_pair_variables(dataframe, [feature_x, target_variable] )
	plot_histograma_bygroup(dataframe, target_variable)
	plot_histograma_bygroup_target(dataframe, target_variable)

	dfjoints = dataframe[[feature_x, feature_y]].dropna()
	plot_jointdistributions(dfjoints, feature_x, feature_y)
	# To plot scatter feature_x and feature_x2 uncomment
	# #feature_x2 = 'alcar' 
	# #dfjoints = dataframe[[feature_x,feature_y, feature_x2]].dropna() 
	# #plot_jointdistributions(dfjoints, feature_x, feature_y, feature_x2)
	# dfjoints = dataframe[[feature_x,feature_y]].dropna()
	#plot_jointdistributions(dfjoints, feature_x, feature_y)
	
	explanatory_features = [['5glu',  'visita_5_sue_noc', 'visita_5_sue_rec', '5ortostatismo', '5trat']]
	for cols in explanatory_features:
		print("Calculating multicolinearities for Group feature: ", cols, ' \n')
	 	#features = dict_features[cols]
	 	features = cols
	 	print('Detect collinearities dcl and ', features)
	 	#detect_multicollinearities calls to plot_jointdistributions
	 	detect_multicollinearities(dataframe, target_variable, features)

def model_interpretation(estimator, X_train, X_test, explanatory_features):
	""" model_interpretation"""
	from skater.core.explanations import Interpretation
	from skater.model import InMemoryModel
	from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
	
	interpreter = Interpretation(X_test, feature_names=explanatory_features)
	proba = estimator.predict_proba
	target_names = estimator.classes_
	model = InMemoryModel(estimator.predict_proba, examples = X_train)
	print('Ploting importance calling to: interpreter.feature_importance.plot_feature_importance...')
	plots = interpreter.feature_importance.plot_feature_importance(model, ascending=True)
	interpreter.partial_dependence.plot_partial_dependence(explanatory_features[12],model, grid_resolution=50,with_variance=True, figsize = (6, 4))
	# Partial dependence plots  of most important feature
	# explainer = LimeTabularExplainer(X_train, feature_names=explanatory_features,discretize_continuous=True, class_names=['0', '1'])
	# # explain prediction for data point no conversor or conversor, i.e. label 1
	# exp = explainer.explain_instance(X_test[0], estimator.predict_proba).show_in_notebook()
	# p = interpreter.partial_dependence.plot_partial_dependence(['a02'], model, grid_resolution=50, with_variance=True, figsize = (6, 4))
	# fig, axs = interpreter.partial_dependence.plot_partial_dependence(estimator, X_test, 0) 

def income_charts(dataframe):
	#pie chart for distritos/barrios
	plt.subplot(1, 2, 1)
	distritos = dataframe['distrito'].dropna()
	distritos_list = distritos.unique()
	dist_rentas = {}
	for distri in distritos_list:
		dist_rentas[distri] = np.sum(distritos.str.find(distri) == 0)
	plt.pie([v for v in dist_rentas.values()], labels=[k for k in dist_rentas.keys()],autopct=None)
	plt.title('Home residency of volunteers')
	
	plt.subplot(1, 2, 2)
	rentas = dataframe['nivelrenta'].dropna()
	rentas_list = rentas.unique()
	dist_nivelrentas = {}
	for rent in rentas_list:
		dist_nivelrentas[rent] = np.sum(rentas == rent)
		print('level', rent, ' = ', dist_nivelrentas[rent] )
	patches, texts = plt.pie([v for v in dist_nivelrentas.values()], labels=[k for k in dist_nivelrentas.keys()],autopct=None)
	plt.title('Income level based in home residency')
	labels= [ 'Low', 'Medium','High']
	plt.legend(patches, labels, loc="best")
	plt.show()

def save_the_model(clf, clf_name, path=None):
	"""
	"""
	print('CLF params are:', clf.get_params())
	now = datetime.datetime.now()
	now = now.strftime("%Y-%m-%d.%H:%M")

	modelname = clf_name + '_'+ now +'.pkl'
	modelname = os.path.join(path, modelname)
	print('Saving the model with joblib...')
	joblib.dump(clf, modelname)
	print('To load the model: my_model_loaded = joblib.load("my_model.pkl") ')

def socioeconomics_infographics():
	""" socioeconomics_infographics: EDA for wealth, diet etc
	"""
	plt.close('all')
	
	dataframe = run_load_csv('/Users/jaime/vallecas/data/scc/Dataset_29052018.csv')	
	target_variable = 'conversionmci'
	# Feature Selection : cosmetic name changing and select input and output 
	print('Calling for cosmetic cleanup (all lowercase, /, remove blanks) e.g. cleanup_column_names(df,rename_dict={},do_inplace=True)') 
	cleanup_column_names(dataframe, {}, True)
	#combine features to create familiar_ad and physical exercise
	dataframe = combine_features(dataframe)
	#copy dataframe with the cosmetic changes e.g. Tiempo is now tiempo
	dataframe_orig = dataframe.copy()
	#replace -1 by 0 
	dataframe['conversionmci'].replace(-1,0, inplace=True)
	plot_histograma_bygroup_categorical(dataframe_orig, target_variable=target_variable)
	#(4) Descriptive analytics: plot scatter and histograms
	#longit_xy_scatter = ['scd_visita', 'gds_visita'] #it works for longitudinal
	#plot_scatter_target_cond(dataframe_orig, ['renta', 'anos_escolaridad'], target_variable=target_variable)
	#features_to_plot = ['scd_visita1', 'gds_visita1'] 
	plot_histogram_pair_variables(dataframe, ['renta', 'anos_escolaridad'] )
	# #plot 1 histogram by grouping values of one continuous feature 
	plot_histograma_bygroup(dataframe, 'renta')
	# # plot one histogram grouping by the value of the target variable
	plot_histograma_bygroup_target(dataframe, target_variable)
	#multicollinearity
	feature_x = 'renta'
	feature_y = 'anos_escolaridad'
	dfjoints = dataframe_orig[[feature_x, feature_y]].dropna()
	#plot_jointdistributions(dfjoints, feature_x, feature_y)
	# To plot scatter feature_x and feature_x2 uncomment
	# #feature_x2 = 'alcar' 
	# #dfjoints = dataframe[[feature_x,feature_y, feature_x2]].dropna() 
	# #plot_jointdistributions(dfjoints, feature_x, feature_y, feature_x2)
	# dfjoints = dataframe[[feature_x,feature_y]].dropna()
	plot_jointdistributions(dfjoints, feature_x, feature_y)
	feature_y = 'mmse_visita1'
	
	
	#plot_jointdistributions(dataframe_orig[[feature_x, feature_y]].dropna(), feature_x, 'mmse_visita1')



	#Detect multicollinearities
	#cols_list = [['scd_visita1', 'gds_visita1']] to multicollinearity of a list
	#cols_list = [['scd_visita1', 'edadinicio_visita1', 'tpoevol_visita1', 'peorotros_visita1', 'preocupacion_visita1', 'eqm06_visita1', 'eqm07_visita1', 'eqm81_visita1', 'eqm82_visita1', 'eqm83_visita1', 'eqm84_visita1', 'eqm85_visita1', 'eqm86_visita1', 'eqm09_visita1', 'eqm10_visita1', 'act_aten_visita1', 'act_orie_visita1', 'act_mrec_visita1', 'act_memt_visita1', 'act_visu_visita1', 'act_expr_visita1', 'act_comp_visita1', 'act_ejec_visita1', 'act_prax_visita1', 'act_depre_visita1', 'act_ansi_visita1', 'act_apat_visita1', 'gds_visita1', 'stai_visita1', 'eq5dmov_visita1', 'eq5dcp_visita1', 'eq5dact_visita1', 'eq5ddol_visita1', 'eq5dans_visita1', 'eq5dsalud_visita1', 'eq5deva_visita1', 'relafami_visita1', 'relaamigo_visita1', 'relaocio_visita1', 'rsoled_visita1', 'valcvida_visita1', 'valsatvid_visita1', 'valfelc_visita1']]
	cols_list = [['scd_visita1', 'gds_visita1']]
	cols_list = [['mmse_visita1',	'anos_escolaridad', 'renta', 'reloj_visita1','fcsrtrl1_visita1','animales_visita1', 'dietaproteica', 'dietagrasa', 'dietasaludable']]


	#for cols in cols_list:
	#for cols in dict_features.keys():
	for cols in cols_list:
	 	print("Calculating miulticolinearities for Group feature: ", cols, ' \n')
	 	#features = dict_features[cols]
	 	features = cols
	 	print('Detect collinearities conversionmci and ', features)
	 	#detect_multicollinearities calls to plot_jointdistributions
	 	detect_multicollinearities(dataframe, 'conversionmci', features)




if __name__ == "__name__":
	#print(Parallel(n_jobs=2)(parallel_func() for _ in range(3)))  # forgot delayed around parallel_func here
	main()