import sys
import pickle
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

import decision_tree as dt
import naive_bayes as nb

def main():
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
	fp_test   = "../data/ds" + ds_option + "/ds" + ds_option + "Test.csv"

	### Validation Output
	fp_dt_val_out = "../output/ds" + ds_option + "Val-dt.csv"
	fp_nb_val_out = "../output/ds" + ds_option + "Val-nb.csv"
	fp_ph_val_out = "../output/ds" + ds_option + "Val-3.csv" ### Placeholder

	### Test Output
	fp_dt_test_out = "../output/ds" + ds_option + "Test-dt.csv"
	fp_nb_test_out = "../output/ds" + ds_option + "Test-nb.csv"
	fp_ph_test_out = "../output/ds" + ds_option + "Test-3.csv" ### Placeholder

	### Saved Models
	fp_dt_mdl = "../models/ds" + ds_option + "/dt_mdl.pkl" 
	fp_nb_mdl = "../models/ds" + ds_option + "/nb_mdl.pkl"
	fp_ph_mdl = "../models/ds" + ds_option + "/ph_mdl.pkl" ### Placeholder


	user_choice = input("Would you like to train or test? \n" +
						"0. Train\n"
						"1. Test\n")

####################################################### TRAINING ########################################################
	if(user_choice == '0'):
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
			print("Training Decision Tree...\n")

			### Run decision tree
			dt_clf, dt_predicted = dt.decision_tree_training(training, validation)

			### Save predictions to output file
			with open(fp_dt_val_out, 'w') as file:
				for i in range(len(dt_predicted)):
					file.write('%d,%d\n' % (i + 1, dt_predicted[i]))

			### Save model
			joblib.dump(dt_clf, fp_dt_mdl)

			###Display accuracy
			accuracy = accuracy_score(val_labels, dt_predicted)
			print("The training accuracy of Decision Tree was {}".format(accuracy))

		if "nb" in alg_option:
			print("Training Naive Bayes...\n")

			### Run Naive Bayes
			nb_clf, nb_predicted = nb.naive_bayes_training(training, validation)

			### Save predictions to output file
			with open(fp_nb_val_out, 'w') as file:
				for i in range(len(nb_predicted)):
					file.write('%d,%d\n' % (i + 1, nb_predicted[i]))

			### Save model
			joblib.dump(nb_clf, fp_nb_mdl)

			###Display accuracy
			accuracy = accuracy_score(val_labels, nb_predicted)
			print("The training accuracy of Naive Bayes was {}".format(accuracy))

		if "3" in alg_option:
			print("Training Placeholder...\n")

			### Run Placeholder
			ph_clf, ph_predicted = placeholder(training, validation)

			### Save predictions to output file
			with open(fp_ph_val_out, 'w') as file:
				for i in range(len(ph_predicted)):
					file.write('%d,%d\n' % (i + 1, ph_predicted[i]))

			### Save model
			joblib.dump(ph_clf, fp_ph_mdl)

			###Display accuracy
			accuracy = accuracy_score(val_labels, ph_predicted)
			print("The training accuracy of PLACEHOLDER was {}".format(accuracy))


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
			print("Testing Decision Tree...\n")

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
			print("Testing Naive Bayes...\n")

			### Load model
			model = joblib.load(fp_nb_mdl)

			### Run Naive Bayes
			nb_predicted = nb.naive_bayes_testing(model, testing)

			### Save predictions to output file
			with open(fp_nb_test_out, 'w') as file:
				for i in range(len(nb_predicted)):
					file.write('%d,%d\n' % (i + 1, nb_predicted[i]))
			
			print("Predictions have been saved to " + fp_nb_test_out)
	

if __name__ == "__main__":
    main()	
	