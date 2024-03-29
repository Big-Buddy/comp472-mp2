import sys
import csv

from sklearn.externals import joblib
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import decision_tree as dt
import naive_bayes as nb
import svc as ls

def main():
	while(True):
		"""
		Choose data set to work on
		"""
		ds_input = input("\nWhich data set would you like to use?\n"
						 "1. Data set 1\n"
						 "2. Data set 2\n")
		if(ds_input == '1'):
			ds_option = '1'
		else:
			ds_option = '2'


		"""
		Set file path parameters
		"""
		### Input
		fp_train  = "../data/ds" + ds_option + "/ds" + ds_option + "Train.csv"
		fp_val    = "../data/ds" + ds_option + "/ds" + ds_option + "Val.csv"
		fp_test   = "../data/ds" + ds_option + "/ds" + ds_option + "Test.csv"

		### Validation Output
		fp_dt_val_out = "../output/ds" + ds_option + "Val-dt.csv"
		fp_nb_val_out = "../output/ds" + ds_option + "Val-nb.csv"
		fp_ls_val_out = "../output/ds" + ds_option + "Val-3.csv"

		### Test Output
		fp_dt_test_out = "../output/ds" + ds_option + "Test-dt.csv"
		fp_nb_test_out = "../output/ds" + ds_option + "Test-nb.csv"
		fp_ls_test_out = "../output/ds" + ds_option + "Test-3.csv"

		### Saved Models
		fp_dt_mdl = "../models/ds" + ds_option + "/dt_mdl.pkl" 
		fp_nb_mdl = "../models/ds" + ds_option + "/nb_mdl.pkl"
		fp_ls_mdl = "../models/ds" + ds_option + "/ls_mdl.pkl"


		train_test_input = input("Would you like to train or test? \n"
								"1. Train\n"
								"2. Test\n")
		
		alg_input = input("Which algorithm would you like to use?\n"
						"1. Decision Tree\n"
						"2. Naive Bayes\n"
						"3. SVC\n")
		
		if(alg_input == '1'):
			alg_option = 'dt'
		elif(alg_input == '2'):
			alg_option = 'nb'
		else:
			alg_option = 'ls'

		####################################################### TRAINING ########################################################
		if(train_test_input == '1'):
			"""
			Read training & validation datasets
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
			#get rid of possible newline character at end of list
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
				print("Training Decision Tree...")

				### Run decision tree
				dt_clf, dt_predicted = dt.decision_tree_training(training, validation, ds_option)

				### Save predictions to output file
				with open(fp_dt_val_out, 'w') as file:
					for i in range(len(dt_predicted)):
						file.write('%d,%d\n' % (i + 1, dt_predicted[i]))

				### Save model
				joblib.dump(dt_clf, fp_dt_mdl)

				### Display scores
				print("Scores for Decision Tree on ds{}:".format(ds_option))
				display_scores(dt_predicted, val_labels, alg_option, ds_option)


			if "nb" in alg_option:
				print("Training Naive Bayes...")

				### Run Naive Bayes
				nb_clf, nb_predicted = nb.naive_bayes_training(training, validation, ds_option)

				### Save predictions to output file
				with open(fp_nb_val_out, 'w') as file:
					for i in range(len(nb_predicted)):
						file.write('%d,%d\n' % (i + 1, nb_predicted[i]))

				### Save model
				joblib.dump(nb_clf, fp_nb_mdl)

				### Display scores
				print("Scores for Naive Bayes on ds{}:".format(ds_option))
				display_scores(nb_predicted, val_labels, alg_option, ds_option)

				
			if "ls" in alg_option:
				print("Training SVC...")

				### Run Placeholder
				ls_clf, ls_predicted = ls.svc_training(training, validation, ds_option)

				### Save predictions to output file
				with open(fp_ls_val_out, 'w') as file:
					for i in range(len(ls_predicted)):
						file.write('%d,%d\n' % (i + 1, ls_predicted[i]))

				### Save model
				joblib.dump(ls_clf, fp_ls_mdl)

				### Display scores
				print("Scores for SVC on ds{}:".format(ds_option))
				display_scores(ls_predicted, val_labels, alg_option, ds_option)
	


		################################################## TESTING ########################################################
		else:
			"""
			Read test dataset
			"""
			with open(fp_test, 'r') as file:
				test_data = [line.split(',') for line in file.read().split('\n')]

			#get rid of possible newline character at end of list
			if(test_data[-1] == ['']):
				test_data.pop()
				
			test_data = [[int(element) for element in row] for row in test_data]


			#Convert to numpy array
			testing = np.array([d for d in test_data], dtype=np.int32)

			"""
			Run testing algorithms
			"""
			if "dt" in alg_option:
				print("Testing Decision Tree...")

				### Load model
				model = joblib.load(fp_dt_mdl)

				### Run decision tree
				dt_predicted = dt.decision_tree_testing(model, testing)

				### Save predictions to output file
				with open(fp_dt_test_out, 'w') as file:
					for i in range(len(dt_predicted)):
						file.write('%d,%d\n' % (i + 1, dt_predicted[i]))
				print("Predictions have been saved to " + fp_dt_test_out)

			if "nb" in alg_option:
				print("Testing Naive Bayes...")

				### Load model
				model = joblib.load(fp_nb_mdl)

				### Run Naive Bayes
				nb_predicted = nb.naive_bayes_testing(model, testing)

				### Save predictions to output file
				with open(fp_nb_test_out, 'w') as file:
					for i in range(len(nb_predicted)):
						file.write('%d,%d\n' % (i + 1, nb_predicted[i]))
				
				print("Predictions have been saved to " + fp_nb_test_out)
	
			if "ls" in alg_option:
				### Load model
				model = joblib.load(fp_ls_mdl)

				### Run Placeholder
				ls_predicted = ls.svc_testing(model, testing)

				### Save predictions
				with open(fp_ls_test_out, 'w') as file:
					for i in range(len(ls_predicted)):
						file.write('%d,%d\n' % (i + 1, ls_predicted[i]))
				print("Predictions have been saved to " + fp_ls_test_out)



def display_scores(predicted_labels, true_labels, model_name, dataset):
	### Generate file name for confusion matrix
	confusion_matrix_out = "../matrix/ds" + dataset + "-maxtrix-" + model_name + ".csv"
	
	### Calculate scores
	accuracy = accuracy_score(true_labels, predicted_labels)
	precision = precision_score(true_labels, predicted_labels, average='weighted')
	recall = recall_score(true_labels, predicted_labels, average='weighted')

	### Display scores
	print("Accuracy: {}".format(accuracy))
	print("Precision (weighted average): {}".format(precision))
	print("Recall (weighted average): {}".format(recall))

	### Generate confusion matrix
	if dataset == '1':
		matrix_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
			  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
			  20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
			  30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
			  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
			  50]	
	else:
		matrix_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

	confusion = confusion_matrix(true_labels, predicted_labels, labels=matrix_labels)
	confusion_list = (confusion.astype(int)).tolist()
	
	### Add row and column labels to confusion list
	confusion_list.insert(0, matrix_labels)
	for index, row in enumerate(confusion_list):
		if index == 0:
			row.insert(0, '')
		else:
			row.insert(0, matrix_labels[index])

	### Save confusion matrix
	with open(confusion_matrix_out, 'w', newline='') as myfile:
		wr = csv.writer(myfile)
		wr.writerows(confusion_list)



if __name__ == "__main__":
    main()
