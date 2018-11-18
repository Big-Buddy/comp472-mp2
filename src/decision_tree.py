from sklearn import tree

def decision_tree_training(training, validation):
	classifier = tree.DecisionTreeClassifier()
	classifier.fit(training["features"], training["labels"])

	validation_predicted = classifier.predict(validation["features"])

<<<<<<< HEAD
	return classifier, validation_predicted
=======
	return classifier, validation_predicted

def decision_tree_testing(model, testing):
	testing_predicted = model.predict(testing)
	return testing_predicted
>>>>>>> f19ff0a796cd11adea77ed49a3df11d7acccd8e3
