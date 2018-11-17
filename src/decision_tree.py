from sklearn import tree
from sklearn.metrics import accuracy_score

def decision_tree_training(training, validation):
	classifier = tree.DecisionTreeClassifier()
	classifier.fit(training["features"], training["labels"])

	validation_predicted = classifier.predict(validation["features"])

	return validation_predicted