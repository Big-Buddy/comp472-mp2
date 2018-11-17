from sklearn import tree

def decision_tree_training(training, validation):
	classifier = tree.DecisionTreeClassifier()
	classifier.fit(training["features"], training["labels"])

	validation_predicted = classifier.predict(validation["features"])

	return classifier, validation_predicted