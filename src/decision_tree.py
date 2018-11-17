from sklearn import tree
from sklearn.metrics import accuracy_score

def decision_tree(training, validation):
	classifier = tree.DecisionTreeClassifier()
	classifier.fit(training["features"], training["labels"])

	validation_predicted = classifier.predict(validation["features"])
	accuracy_score = accuracy_score(validation["labels"], validation_predicted)

	return validation_predicted