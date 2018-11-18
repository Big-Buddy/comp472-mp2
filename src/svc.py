from sklearn import svm

ds1_parameters = {
    "kernel": 'poly',
    "degree": 8,
    "gamma": 'scale'
}

ds2_parameters = {
    "kernel": 'poly',
    "degree": 10,
    "gamma": 'scale'
}

def svc_training(training, validation, dataset):
	if(dataset == '1'):
		classifier = svm.SVC(kernel=ds1_parameters["kernel"], degree=ds1_parameters["degree"], gamma=ds1_parameters["gamma"])
	else:
		classifier = svm.SVC(kernel=ds2_parameters["kernel"], degree=ds2_parameters["degree"], gamma=ds1_parameters["gamma"])
		
	classifier.fit(training["features"], training["labels"])
	validation_predicted = classifier.predict(validation["features"])

	return classifier, validation_predicted

def svc_testing(model, testing):
    testing_predicted = model.predict(testing)
    return testing_predicted