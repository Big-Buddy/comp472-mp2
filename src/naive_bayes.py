from sklearn import naive_bayes
from sklearn.metrics import accuracy_score

import numpy as np

def naive_bayes_training(training, validation):
    ########MODEL TRAINING##########################
    classifier = naive_bayes.BernoulliNB(alpha=0.5)
    classifier.fit(training["features"], training["labels"])

    validation_predicted = classifier.predict(validation["features"])

    return validation_predicted

"""
#############TRAINING DATA###################
with open('data/ds1/ds1Train.csv', 'r') as file:
    train_data = [line.split(',') for line in file.read().split('\n')]

#get rid of newline character at end of list
if(train_data[-1] == ['']):
    train_data.pop()

train_features = np.array([d[:-1] for d in train_data], dtype=np.int32)
train_labels = np.array([d[-1] for d in train_data], dtype=np.int32)


#########VALIDATION DATA ########################
with open('data/ds1/ds1Val.csv', 'r') as file:
    validation_data = [line.split(',') for line in file.read().split('\n')]

#get rid of newline character at end of list
if(validation_data[-1] == ['']):
    validation_data.pop()

validation_features = np.array([d[:-1] for d in validation_data], dtype=np.int32)
validation_labels = np.array([d[-1] for d in validation_data], dtype=np.int32)

"""
