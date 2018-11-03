import sys
import pickle
from sklearn import tree
import decision_tree as dt
import naive_bayes as nb

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
fp_ph_out = "../output/ds" + ds_option + "Test-ph.csv" ### Placeholder

fp_dt_mdl = "../models/dt_mdl.pkl"
fp_nb_mdl = "../models/nb_mdl.pkl"
fp_ph_mdl = "../models/ph_mdl.pkl" ### Placeholder

"""
Read dataset
"""
with open(fp_train, 'r') as file:
	train_data = [line.split(',') for line in file.read().split('\n')]
	if not train_data[-1][0]:
		train_data = train_data[0:-1]

train_data = [[int(element) for element in row] for row in train_data]
train_features = [d[:-1] for d in train_data]
train_labels = [d[-1] for d in train_data]
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
val_features = [d[:-1] for d in val_data]
val_labels = [d[-1] for d in val_data]
validation = {
	"data": val_data,
	"features": val_features,
	"labels": val_labels
}

"""
Run algorithms
"""
if "dt" in alg_option:
	### Run decision tree
	dt_predicted = dt.decision_tree(training, validation)
	### Save predictions
	with open(fp_dt_out, 'w') as file:
		for i in range(len(dt_predicted)):
			file.write('%d,%d\n' % (i + 1, dt_predicted[i]))
	### Save model
	
if "nb" in alg_option:
	### Run Naive Bayes
	nb_predicted = nb.naive_bayes(training, validation)
	### Save predictions
	with open(fp_nb_out, 'w') as file:
		for i in range(len(nb_predicted)):
			file.write('%d,%d\n' % (i + 1, nb_predicted[i]))
	### Save model
if "3" in alg_option:
	### Run Placeholder
	ph_predicted = placeholder(training, validation)
	### Save predictions
	with open(fp_ph_out, 'w') as file:
		for i in range(len(ph_predicted)):
			file.write('%d,%d\n' % (i + 1, ph_predicted[i]))
	### Save model


