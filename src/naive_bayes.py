from sklearn import naive_bayes

def naive_bayes_training(training, validation):
    classifier = naive_bayes.BernoulliNB(alpha=0.5)
    classifier.fit(training["features"], training["labels"])

    validation_predicted = classifier.predict(validation["features"])

    return classifier, validation_predicted

def naive_bayes_testing(model, testing):
    testing_predicted = model.predict(testing)
    return testing_predicted
    