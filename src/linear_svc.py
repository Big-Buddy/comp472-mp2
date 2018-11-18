from sklearn import svm

ds1_parameters = {
    "kernel": 'linear'
}

ds2_parameters = {
    "kernel": 'linear'
}

def linear_svc_training(training, validation, dataset):
	if(dataset == '1'):
		classifier = svm.SVC(kernel=ds1_parameters["kernel"])
	else:
		classifier = svm.SVC(kernel=ds2_parameters["kernel"])
		
	classifier.fit(training["features"], training["labels"])
	validation_predicted = classifier.predict(validation["features"])

	return classifier, validation_predicted

def linear_svc_testing(model, testing):
    testing_predicted = model.predict(testing)
    return testing_predicted