from sklearn import svm

def linear_svc_training(training, validation):
	classifier = svm.SVC(kernel='linear')
	classifier.fit(training["features"], training["labels"])

	validation_predicted = classifier.predict(validation["features"])

	return classifier, validation_predicted

def linear_svc_testing(model, testing):
    testing_predicted = model.predict(testing)
    return testing_predicted