""" descriptive_stats.py load csv and plor perform basic statistics nothing really that excel could do as well
"""

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
from sklearn.model_selection import StratifiedKFold
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
	feature_label = run_histogram_and_scatter(dataset)
	run_correlation_matrix(dataset,feature_label)
	run_descriptive_stats()
	#run_svms()
	#run_networks()
def run_load_csv(csv_path = None):
	""" load csv database"""
	if csv_path is None:
		csv_path = "/Users/jaime/vallecas/data/scc/sccplus-24012018.csv"
	dataset = pd.read_csv(csv_path) #, sep=';')
	print "CSV loaded. The feature names are: \n", dataset.keys()
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
	g.set(xticklabels=[])
	g = sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values, vmin =-1, vmax=1, center=0,annot=True)
	#ax.set_xlabel("Features")
	#ax.set_ylabel("Features")
	ax.set_title(cmethod + " correlation")
	g.set(xticklabels=[])
	g.set_yticklabels(feature_label, rotation=30)
	plt.show()

def chi_square_of_df_cols(df, col1, col2):
	df_col1, df_col2 = df[col1], df[col2]
	obs = np.array(df_col1, df_col2 ).T
def anova_test(dataset, features=None):
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
	DB_PATH = "/Users/jaime/vallecas/data/scc/foranovatest.csv"
	df = pd.read_csv(DB_PATH)
	print("Calculating ANOVA for features:\n",df.ix[0])
	if features is None:
		features = ['age', 'years_education'] #, 'mmse','FLUISEM2']
	ctrl = df[features[0]].where(df['scc_group']=='NC').dropna()
	sccs = df[features[0]].where(df['scc_group']=='SCD-P').dropna()
	print("Number of control subjects=",len(ctrl.notnull()))
	print("Number of scc subjects=",len(sccs.notnull()))
	for idx, val in enumerate(features):
		expl_control_var = features[idx] + ' ~ scc_group'
		mod = ols(expl_control_var, data=df).fit()
		aov_table = sm.stats.anova_lm(mod, typ=2)
		print("ANOVA table feature:", val) 
		print(aov_table)
		#Create a boxplot
		df.boxplot(features[idx], by='scc_group', figsize=(12, 8))

def ancova_test(dataset, features=None):
	""" ACNOVA test controlling for features that may have a relationship with the dependent variable"""
	print("Calculating ANCOVA for features:\n",df.ix[0])

def t_test(dataset, features=None):
	""" t test"""
	print("Calculating t test for features:\n",df.ix[0])

def load_groups(path=None):
	# load the two gropus to compare from a csv file
	if path is None:
		#assign the location of the csv file
		DB_WEB = "/Users/jaime/vallecas/data/surrogate_bcpa/postmortem-JG-AR-17122017.csv"
	print("Loading gropus from csv file: :\n",path)
	dataset = pd.read_csv(DB_WEB) #, sep=';')
	print("CSV loaded. The feature names are: ", dataset.keys())


if __name__ = "__name__":
	main()