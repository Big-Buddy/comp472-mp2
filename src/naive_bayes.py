from sklearn import naive_bayes
from sklearn.metrics import accuracy_score

import numpy as np

def naive_bayes_training(training, validation):
    classifier = naive_bayes.BernoulliNB(alpha=0.5)
    classifier.fit(training["features"], training["labels"])

    validation_predicted = classifier.predict(validation["features"])

    return validation_predicted

