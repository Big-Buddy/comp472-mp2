from sklearn import tree

ds1_parameters = {
    "max_depth": 20, 		#int or None
	"criterion": 'entropy' 	#'gini' or 'entropy'
}

ds2_parameters = {
    "max_depth": 25,		#int or None
	"criterion": 'gini' 	#'gini' or 'entropy'
}

def decision_tree_training(training, validation, dataset):
	if (dataset == '1'):
		classifier = tree.DecisionTreeClassifier(
			criterion=ds1_parameters["criterion"], 
			max_depth=ds1_parameters["max_depth"]
		)
	else:
		classifier = tree.DecisionTreeClassifier(
			criterion=ds2_parameters["criterion"], 
			max_depth=ds2_parameters["max_depth"]
		)
	
	classifier.fit(training["features"], training["labels"])
	validation_predicted = classifier.predict(validation["features"])

	return classifier, validation_predicted

def decision_tree_testing(model, testing):
	testing_predicted = model.predict(testing)
	return testing_predicted
