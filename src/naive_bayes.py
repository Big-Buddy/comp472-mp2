from sklearn import naive_bayes

ds1_parameters = {
    "alpha": 0.5
}

ds2_parameters = {
    "alpha": 0.1
}

def naive_bayes_training(training, validation, dataset):
    if(dataset == '1'):
        classifier = naive_bayes.BernoulliNB(alpha=ds1_parameters["alpha"])
    else:
        classifier = naive_bayes.BernoulliNB(alpha=ds2_parameters["alpha"])

    classifier.fit(training["features"], training["labels"])
    validation_predicted = classifier.predict(validation["features"])

    return classifier, validation_predicted

def naive_bayes_testing(model, testing):
    testing_predicted = model.predict(testing)
    return testing_predicted

