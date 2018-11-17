import sys

from sklearn.externals import joblib
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score

import decision_tree as dt
#import naive_bayes as nb
import linear_svc as ls

"""
Read command line arguments
"""
ds_option = sys.argv[1]
alg_option = sys.argv[2:]

"""
Set file path parameters
"""
### Input
fp_train  = "../data/ds" + ds_option + "/ds" + ds_option + "Train.csv"
fp_val    = "../data/ds" + ds_option + "/ds" + ds_option + "Val.csv"
### Output
fp_dt_out = "../output/ds" + ds_option + "Test-dt.csv"
fp_nb_out = "../output/ds" + ds_option + "Test-nb.csv"
fp_ls_out = "../output/ds" + ds_option + "Test-3.csv" ### Placeholder

fp_dt_mdl = "../models/ds" + ds_option + "/dt_mdl.pkl" 
fp_nb_mdl = "../models/ds" + ds_option + "/nb_mdl.pkl"
fp_ls_mdl = "../models/ds" + ds_option + "/l_svc_mdl.pkl" ### Placeholder

"""
Read dataset
"""
with open(fp_train, 'r') as file:
	train_data = [line.split(',') for line in file.read().split('\n')]
	if not train_data[-1][0]:
		train_data = train_data[0:-1]

train_data = [[int(element) for element in row] for row in train_data]
#get rid of possible newline character at end of list
if(train_data[-1] == ['']):
    train_data.pop()

train_features = np.array([d[:-1] for d in train_data], dtype=np.int32)
train_labels = np.array([d[-1] for d in train_data], dtype=np.int32)
training = {
	"data": train_data, 
	"features": train_features, 
	"labels": train_labels
}

with open(fp_val, 'r') as file:
	val_data = [line.split(',') for line in file.read().split('\n')]
	if not val_data[-1][0]:
		val_data = val_data[0:-1]

val_data = [[int(element) for element in row] for row in val_data]
#get rid of newline character at end of list
if(val_data[-1] == ['']):
    val_data.pop()

val_features =  np.array([d[:-1] for d in val_data], dtype=np.int32)
val_labels = np.array([d[-1] for d in val_data], dtype=np.int32)
validation = {
	"data": val_data,
	"features": val_features,
	"labels": val_labels
}

"""
Run training algorithms
"""
if "dt" in alg_option:
	### Run decision tree
	dt_clf, dt_predicted = dt.decision_tree_training(training, validation)

	### Save predictions
	with open(fp_dt_out, 'w') as file:
		for i in range(len(dt_predicted)):
			file.write('%d,%d\n' % (i + 1, dt_predicted[i]))

	### Save model
	joblib.dump(dt_clf, fp_dt_mdl)

	###Display accuracy
	accuracy = accuracy_score(val_labels, dt_predicted)
	print("The training accuracy of Decision Tree was {}".format(accuracy))

if "nb" in alg_option:
	### Run Naive Bayes
	nb_clf, nb_predicted = nb.naive_bayes(training, validation)

	### Save predictions
	with open(fp_nb_out, 'w') as file:
		for i in range(len(nb_predicted)):
			file.write('%d,%d\n' % (i + 1, nb_predicted[i]))

	### Save model
	joblib.dump(nb_clf, fp_nb_mdl)

	###Display accuracy
	accuracy = accuracy_score(val_labels, nb_predicted)
	print("The training accuracy of Naive Bayes was {}".format(accuracy))

if "ls" in alg_option:
	### Run Placeholder
	ls_clf, ls_predicted = ls.linear_svc_training(training, validation)

	### Save predictions
	with open(fp_ls_out, 'w') as file:
		for i in range(len(ls_predicted)):
			file.write('%d,%d\n' % (i + 1, ls_predicted[i]))

	### Save model
	joblib.dump(ls_clf, fp_ls_mdl)

	###Display accuracy
	accuracy = accuracy_score(val_labels, ls_predicted)
	print("The training accuracy of LINEAR SVC was {}".format(accuracy))