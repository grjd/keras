# -*- coding: utf-8 -*-
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
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from random import randint
import seaborn as sns

def main():
	dataset = run_load_csv()
	dataset = run_preprocessing(dataset)
	pdb.set_trace()
	feature_label = run_histogram_and_scatter(dataset)
	run_correlation_matrix(dataset,feature_label)
	run_descriptive_stats(dataset,feature_label)
	#run_svm()
	#run_networks()

def run_naive_Bayes(dataset, features=None):
	from plot_learning_curve import plot_learning_curve
	from sklearn.naive_bayes import GaussianNB
	title = "Learning Curves (Naive Bayes)"
	# Cross validation with 100 iterations to get smoother mean test and train
	# score curves, each time with 20% data randomly selected as a validation set.
	features =['Visita_1_EQ5DMOV', 'Visita_1_EQ5DCP', 'years_school', 'SCD_v1']
	df = dataset.fillna(method='ffill')
	X_all = df[features]
	y_all = df['Conversion']
	
	cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
	estimator = GaussianNB()
	y_pred = estimator.fit(X_all.values, y_all.values).predict(X_all.values)
	print("Number of mislabeled points out of a total %d points : %d" % (X_all.values.shape[0],(y_all.values != y_pred).sum()))
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
	# SVM model, cost and gamma parameters for RBF kernel
	svm = SVC(cache_size=1000, kernel='rbf')
	parameters = {'C':10. ** np.arange(5,10), 'gamma':2. ** np.arange(-5, -1)}
	kfolds = StratifiedKFold(5)
	cv = kfolds.split(X,y)
	title =  "Learning Curves (SVM)"
	plot_learning_curve(svm, title, X_all.values, y_all.values, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
	# Exhaustive search over specified parameter values for an estimator.
	grid = GridSearchCV(svm, parameters, cv=cv, verbose=3, n_jobs=2)
	grid.fit(X, y)
	print("predicting")
	print("score: ", grid.score(X_test, y_test))
	print(grid.best_estimator_)



def run_preprocessing(dataset, feature_labels=None):
	""" transform str to int categorical features into numeric ones and scaling features"""
	from sklearn import preprocessing
	
	if feature_labels is None:
		feature_labels = dataset.keys() #'Visita_4_EQM01', row 380 'ERROR' . 'years_school', 'Tiempo',
		feature_labels_str =[  'Visita_1_EQ5DMOV', 'Visita_1_EQ5DCP', 'Visita_1_EQ5DACT', 'Visita_1_EQ5DDOL', 'Visita_1_EQ5DANS', 'Visita_1_EQ5DSALUD', 'Visita_1_EQ5DEVA', 'Visita_1_ALFRUT', 'Visita_1_ALCAR', 'Visita_1_ALPESBLAN', 'Visita_1_ALPESZUL', 'Visita_1_ALAVES', 'Visita_1_ALACEIT', 'Visita_1_ALPAST', 'Visita_1_ALPAN', 'Visita_1_ALVERD', 'Visita_1_ALLEG', 'Visita_1_ALEMB', 'Visita_1_ALLACT', 'Visita_1_ALHUEV', 'Visita_1_ALDULC', 'Visita_1_HSNOCT', 'Visita_1_RELAFAMI', 'Visita_1_RELAAMIGO', 'Visita_1_RELAOCIO', 'Visita_1_RSOLED', 'Visita_1_A01', 'Visita_1_A02', 'Visita_1_A03', 'Visita_1_A04', 'Visita_1_A05', 'Visita_1_A06', 'Visita_1_A07', 'Visita_1_A08', 'Visita_1_A09', 'Visita_1_A10', 'Visita_1_A11', 'Visita_1_A12', 'Visita_1_A13', 'Visita_1_A14', 'Visita_1_EJFRE', 'Visita_1_EJMINUT', 'Visita_1_VALCVIDA', 'Visita_1_VALSATVID', 'Visita_1_VALFELC', 'Visita_1_SDESTCIV', 'Visita_1_SDHIJOS', 'Visita_1_NUMHIJ', 'Visita_1_SDVIVE', 'Visita_1_SDECONOM', 'Visita_1_SDRESID', 'Visita_1_SDTRABAJA', 'Visita_1_SDOCUPAC', 'Visita_1_SDATRB', 'Visita_1_HTA', 'Visita_1_HTA_INI', 'Visita_1_GLU', 'Visita_1_LIPID', 'Visita_1_LIPID_INI', 'Visita_1_TABAC', 'Visita_1_TABAC_INI', 'Visita_1_TABAC_FIN', 'Visita_1_SP', 'Visita_1_COR', 'Visita_1_COR_INI', 'Visita_1_ARRI', 'Visita_1_CARD', 'Visita_1_CARD_INI', 'Visita_1_TIR', 'Visita_1_ICTUS',  'Visita_1_ICTUS_INI', 'Visita_1_ICTUS_SECU', 'Visita_1_DEPRE', 'Visita_1_DEPRE_INI', 'Visita_1_DEPRE_NUM', 'Visita_1_ANSI', 'Visita_1_ANSI_NUM', 'Visita_1_ANSI_TRAT', 'Visita_1_TCE', 'Visita_1_TCE_NUM', 'Visita_1_TCE_INI', 'Visita_1_TCE_CON', 'Visita_1_SUE_DIA',  'Visita_1_SUE_CON', 'Visita_1_SUE_MAN', 'Visita_1_SUE_SUF', 'Visita_1_SUE_PRO', 'Visita_1_SUE_RON', 'Visita_1_SUE_MOV', 'Visita_1_SUE_RUI', 'Visita_1_SUE_HOR', 'Visita_1_SUE_DEA', 'Visita_1_SUE_REC', 'Visita_1_EDEMMAD','Visita_1_EDEMPAD', 'Visita_1_PABD', 'Visita_1_PESO', 'Visita_1_TALLA', 'Visita_1_AUDI', 'Visita_1_VISU', 'Visita_1_IMC','Visita_1_GLU_INI','Visita_1_TABAC_CANT','Visita_1_ARRI_INI', 'Visita_1_ICTUS_NUM', 'Visita_1_DEPRE_TRAT','Visita_1_ANSI_INI', 'Visita_1_TCE_SECU', 'Visita_1_SUE_NOC']
		feature_labels_categorical = ['Visita_1_Lat']
	for f in range(len(feature_labels_str)):
		dataset[feature_labels_str[f]].dropna(inplace=True)
		if isinstance(dataset[feature_labels[f]].values[0], basestring):
			print("str column!!",feature_labels_str[f], "\n" )
			pd.to_numeric(dataset[feature_labels_str[f]])
		print("scaling for: ",feature_labels_str[f], " ...")	
		dataset[feature_labels_str[f]] = preprocessing.scale(dataset[feature_labels_str[f]])	
	return dataset			
	

def run_load_csv(csv_path = None):
	""" load csv database"""
	if csv_path is None:
		csv_path = "/Users/jaime/vallecas/data/scc/sccplus-24012018.csv"
	dataset = pd.read_csv(csv_path) #, sep=';')
	#print "CSV loaded. The feature names are:", dataset.keys()
	print(dataset.info())
	print(dataset.describe())
	# check if some element is null return dataframe of Booleans
	pd.isnull(dataset)
	return dataset
	# to add an additional features of dummy data
	#x=[randint(0,1) for p in range(0,dataset['lactate'].shape[0])]
	#dataset['me_gusta'] = x
	# delete the feature
	#del dataset['me_gusta']

def run_descriptive_stats(dataset,feature_label):
	#chi_square_of_df_cols(dataset, feature_label[0], feature_label[1])
	anova_test(dataset, [feature_label[1], feature_label[3]])


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